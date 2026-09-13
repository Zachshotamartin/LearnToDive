"""Articulated MuJoCo diving physics for the v11 native trainer.

``training_v3/diver.xml`` is this trainer's model; the published browser
runtime still uses ``public/physics/diver.xml`` (v8) and is not wired to this
trainer. Units are kg, m, s and rad. Native coordinates: X forward, Y lateral,
Z up; the water surface is at Z = -height. The only controller commands are
bounded joint position servos; there is no root wrench. Board spring contact
transfers takeoff momentum physically.

:class:`Arena` integrates a batch of divers and keeps the judge-side
measurements (takeoff, flight rotation, water entry). Every per-environment
array is part of the physical state that recovery snapshots and exact
checkpoints copy; the scratch buffers named in :data:`SCRATCH` are not.
"""
from pathlib import Path

import mujoco
import numpy as np

from entry_faults import ENTRY_BUDGETS
from mujoco.rollout import Rollout

from geometry import (
    ACTION_HIGH, ACTION_LOW, ACTUATOR_MAP, ANGULAR_VELOCITY, BOARD_CONTACTS, FOOT_CONTACT_FOUND, FOOT_GEOMS,
    FOREARM_GEOMS, GEOM_COUNT, GEOM_POSES, GEOM_VELOCITIES, HAND_CONTACT_FOUND, HAND_GEOMS, LEG_GEOMS, SHIN_GEOMS,
    SHOULDER_GEOMS, THIGH_GEOMS, TRUNK_GEOMS, board_forces, decoded_action, entry_geometry, framed_angles, quat_up,
    stand_forces, tuck_geometry, wrap,
)
from positions import position_qualities
from stance import SKILLS, STATE_SPEC, initial_state
from water import body_water, immersion

__all__ = ['Arena', 'ACTION_LOW', 'ACTION_HIGH', 'ACTUATOR_MAP', 'STATE_SPEC', 'SKILLS', 'initial_state',
           'quat_up', 'framed_angles', 'board_forces', 'DT', 'SUBSTEPS', 'TRAINING_SOURCES', 'TARGET_RATE']

ROOT = Path(__file__).resolve().parents[1]
DT = .02
SUBSTEPS = 10
ENTRY_WINDOW = 1.2   # seconds after first contact in which the whole body must submerge
TIMEOUT = 3.6        # seconds without water contact before a dive is abandoned
HOP_SPEED = .25      # upward centre-of-mass speed at contact loss that counts as a hop
RELEASE_AIR_TIME = .04
ASSIST_IMPULSE = .1  # N s of non-foot board contact that invalidates a takeoff
ASSIST_FORCE = 15    # N of non-foot board contact that invalidates a takeoff
TARGET_RATE = np.array([14, 14, 14, 10, 10, 8, 8, 10, 4])  # rad/s servo target slew; legs must outrun a jump
TRAINING_SOURCES = ('environment.py', 'engine.py', 'geometry.py', 'stance.py', 'positions.py', 'water.py', 'rules.py', 'judge.py', 'training_reward.py', 'entry_faults.py', 'motor_objective.py', 'motor_curriculum.py', 'recovery_curriculum.py', 'evaluation.py', 'model_selection.py', 'assessment_stats.py',
                    'policy.py', 'losses.py', 'train.py', 'reference_policies.py', 'difficulty.json', 'diver.xml')
CONTROL_SPEC = 3904  # CTRL | XFRC_APPLIED | EQ_ACTIVE | MOCAP_POS | MOCAP_QUAT for MuJoCo 3.13
ATHLETE = slice(1, 16)               # the fifteen athlete geoms in the model
BOARD_FOUND = slice(150, 210, 4)     # 'found' flag of every board contact sensor
TORSO_BODY = 3
SCRATCH = ('roll_state', 'roll_sensor', 'ctrl', 'qadr', 'vadr')
POOL_X = (-.44, 11.34)
POOL_HALF_WIDTH = 3.56
ATHLETE_MASS = 70.


class Arena:
    """Batched diver physics with takeoff, flight and entry measurements."""

    def __init__(self, n=128, seed=123, threads=8, skills=None, heights=(3, 10), training=True, minimum_heights=None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.training = training
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'training_v3/diver.xml'))
        workers = min(threads, n)
        self.pool = Rollout(nthread=workers)
        self.datas = [mujoco.MjData(self.model) for _ in range(workers)]
        self.resetdata = mujoco.MjData(self.model)
        self.ns = mujoco.mj_stateSize(self.model, STATE_SPEC)
        self.skill_ids = list(range(len(SKILLS))) if skills is None else skills
        self.heights = heights
        self.minimum_heights = minimum_heights or [skill['height'] for skill in SKILLS]
        self.qadr = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        self.vadr = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]
        self.diverged_events = 0
        self._allocate(n)
        self.reset(np.arange(n))

    def _allocate(self, n):
        model = self.model
        zeros = lambda *shape, dtype=float: np.zeros((n, *shape), dtype)
        self.state = zeros(self.ns)
        self.sensors = zeros(model.nsensordata)
        self.targets = zeros(9)
        self.goals = np.tile([.5, 0, 0, 0.], (n, 1))
        self.headfirst = np.ones(n, bool)
        self.armstand = zeros(dtype=bool)
        self.platform = zeros(dtype=bool)
        self.disturbance = zeros()
        self.disturbance_time = zeros()
        self.height = zeros()
        self.skill = zeros(dtype=int)
        # Orientation bookkeeping: unwrapped swing and twist paths.
        self.theta = zeros()
        self.twist = zeros()
        self.prev_pitch = zeros()
        self.prev_twist = zeros()
        self.max_lateral = zeros()
        # Takeoff bookkeeping.
        self.departure_com = zeros()
        self.apex_com = zeros()
        self.takeoff_vertical_speed = zeros()
        self.takeoff_angular_momentum = zeros(3)
        self.phase_theta = zeros()
        self.departure_pitch = zeros()
        self.departure_theta = zeros()
        self.departure_twist = zeros()
        self.last_foot_contact = zeros(2)
        self.foot_departure_gap = zeros()
        self.takeoff_tilt_invalid = zeros(dtype=bool)
        self.air_clear_time = zeros()
        self.air_theta = zeros()
        self.air_twist = zeros()
        self.release_theta = zeros()
        self.release_twist = zeros()
        self.release_time = zeros()
        self.released = zeros(dtype=bool)
        self.board_invalid = zeros(dtype=bool)
        self.board_impulse = zeros()
        self.board_peak = zeros()
        self.stand_impulse = zeros()
        self.stand_peak = zeros()
        self.preparation_bounces = zeros(dtype=int)
        self.foot_side_contact = zeros(dtype=bool)
        self.rotated_recontact = zeros(dtype=bool)
        # Flight position assessment.
        self.position_sums = zeros(4)
        self.motor_position_sums = zeros(4)
        self.motor_position_weight = zeros()
        self.position_ticks = zeros()
        # Water entry bookkeeping.
        self.entry_time = np.full(n, np.nan)
        self.full_entry_time = np.full(n, np.nan)
        self.first_state = zeros(self.ns)
        self.first_sensors = zeros(model.nsensordata)
        self.first_geometry = zeros(dtype=int)
        self.entry_max_angle = zeros()
        self.entry_min_form = np.ones(n)
        self.entry_limbs_valid = np.ones(n, bool)
        self.entry_geometry_valid = np.ones(n, bool)
        self.entry_worst_geometry = zeros(9)
        self.entry_crossed = zeros(dtype=bool)
        self.entry_omega = zeros()
        self.surface_finished = zeros(GEOM_COUNT, dtype=bool)
        self.surface_group_loss = zeros(6)
        self.entry_fault_losses = zeros(len(ENTRY_BUDGETS))
        self.surface_lateral = zeros()
        self.above_water = zeros()
        self.water_fraction = zeros()
        self.returns = zeros()
        self.practice = zeros(dtype=bool)
        self.banked = zeros(dtype=bool)
        # Scratch buffers reused every step.
        self.roll_state = np.empty((n, SUBSTEPS, self.ns))
        self.roll_sensor = np.empty((n, SUBSTEPS, model.nsensordata))
        self.ctrl = np.zeros((n, SUBSTEPS, model.nu + model.nbody * 6 + 8))

    def close(self):
        self.pool.close()

    # ------------------------------------------------------------------ reset
    def reset(self, ids, contexts=None):
        """Place the divers ``ids`` in fresh standing starts.

        ``contexts`` may override skill, height, disturbance and stance
        parameters per environment; random draws are made in a fixed order so
        that the environment stream is reproducible.
        """
        for j, i in enumerate(ids):
            context = {} if contexts is None else contexts[j]
            skill = int(self.rng.choice(self.skill_ids))
            lowest = max(self.heights[0], self.minimum_heights[skill])
            height = self.rng.uniform(lowest, self.heights[1])
            if self.training:
                self.rng.random()  # reserved draw kept so the v11 environment stream is unchanged
            skill = context.get('skill', skill)
            height = context.get('height', height)
            self.skill[i] = skill
            self.height[i] = height
            self.disturbance[i] = context.get('disturbance', 0)
            self.disturbance_time[i] = context.get('disturbanceTime', self.rng.uniform(.65, 1.05))
            stance = self._stance_parameters(context)
            state, sensors, targets = initial_state(self.model, self.resetdata, height, skill, **stance)
            self.state[i] = state
            self.sensors[i] = sensors
            self.targets[i] = targets
            self._clear(i, state)
        self.water_measurements()
        return self.sensors

    def _stance_parameters(self, context):
        draw = self.rng.uniform
        training = self.training
        return dict(
            lean=context.get('lean', draw(.02, .08) if training else .04),
            preload=context.get('preload', draw(-.20, -.16) if training else -.18),
            hip=context.get('hip', draw(.08, .2) if training else .12),
            knee=context.get('knee', draw(.15, .3) if training else .2),
            toe_over=context.get('toeOver', draw(0, .06) if training else .03),
            platform=bool(context.get('platform', False)))

    def _clear(self, i, state):
        quaternion = state[5:9][None]
        up = quat_up(quaternion)[0]
        self.prev_pitch[i] = np.arctan2(up[0], up[2])
        self.prev_twist[i] = framed_angles(quaternion)[1][0]
        for name in ('theta', 'twist', 'max_lateral', 'departure_com', 'apex_com', 'takeoff_vertical_speed', 'takeoff_angular_momentum',
                     'phase_theta', 'departure_pitch', 'departure_theta', 'departure_twist', 'last_foot_contact',
                     'foot_departure_gap', 'takeoff_tilt_invalid', 'air_clear_time', 'air_theta', 'air_twist',
                     'release_theta', 'release_twist', 'release_time', 'released', 'board_invalid', 'board_impulse',
                     'board_peak', 'stand_impulse', 'stand_peak', 'preparation_bounces', 'foot_side_contact',
                     'rotated_recontact', 'position_sums', 'position_ticks', 'motor_position_sums', 'motor_position_weight', 'first_geometry', 'entry_max_angle',
                     'entry_worst_geometry', 'entry_crossed', 'entry_omega', 'surface_finished',
                     'surface_group_loss', 'entry_fault_losses', 'surface_lateral', 'returns', 'practice', 'banked'):
            getattr(self, name)[i] = 0
        self.entry_time[i] = np.nan
        self.full_entry_time[i] = np.nan
        self.entry_min_form[i] = 1
        self.entry_limbs_valid[i] = True
        self.entry_geometry_valid[i] = True

    # ------------------------------------------------------------------ water
    def water_measurements(self):
        geoms = self.sensors[:, GEOM_POSES].reshape(self.n, GEOM_COUNT, 7)
        fraction, extent, _ = immersion(self.model.geom_type[ATHLETE], self.model.geom_size[ATHLETE],
                                        geoms[:, :, :3], geoms[:, :, 3:], -self.height[:, None])
        self.above_water = np.max(geoms[:, :, 2] + extent + self.height[:, None], axis=1)
        masses = self.model.body_mass[self.model.geom_bodyid[ATHLETE]]
        self.water_fraction = np.sum(fraction * masses, axis=1) / ATHLETE_MASS

    def water_rollout(self, ids):
        """Integrate submerged divers one substep at a time with fresh water forces.

        Rollout's sensor output is pre-integration, so a disposable second
        substep obtains the forward-updated sensors of each state; only the
        first physical transition is kept. This avoids stale velocity feedback.
        """
        model = self.model
        count = len(ids)
        state = self.state[ids].copy()
        sensor = self.sensors[ids].copy()
        states = np.empty((count, SUBSTEPS, self.ns))
        sensors = np.empty((count, SUBSTEPS, model.nsensordata))
        after = np.empty_like(sensors)
        control = self.ctrl[ids, :2].copy()
        bodies = model.geom_bodyid[ATHLETE]
        for k in range(SUBSTEPS):
            geoms = sensor[:, GEOM_POSES].reshape(count, GEOM_COUNT, 7)
            velocity = sensor[:, GEOM_VELOCITIES].reshape(count, GEOM_COUNT, 6)
            result = body_water(model.geom_type[ATHLETE], model.geom_size[ATHLETE], geoms[:, :, :3], geoms[:, :, 3:],
                                geoms[:, :, :3], velocity[:, :, :3], velocity[:, :, 3:], model.body_mass[bodies],
                                -self.height[ids, None])
            force = np.zeros((count, model.nbody, 6))
            force[:, bodies, :3] = result['force']
            force[:, bodies, 3:] = result['torque']
            control[:, :, model.nu:model.nu + model.nbody * 6] = force.reshape(count, 1, model.nbody * 6)
            stepped, sensed = self.pool.rollout([model] * count, self.datas, state, control, skip_checks=True,
                                                control_spec=CONTROL_SPEC, nstep=2, state=np.empty((count, 2, self.ns)),
                                                sensordata=np.empty((count, 2, model.nsensordata)))
            states[:, k] = stepped[:, 0]
            sensors[:, k] = sensed[:, 0]
            after[:, k] = sensed[:, 1]
            state = stepped[:, 0].copy()
            sensor = sensed[:, 1].copy()
        return states, sensors, after

    def track_entry_batch(self, ids, state, sensor):
        """Score each anatomical group while it crosses the surface.

        Once a group is fully submerged, later movement cannot worsen that
        group's entry.
        """
        if len(ids) == 0:
            return
        geoms = sensor[:, GEOM_POSES].reshape(len(ids), GEOM_COUNT, 7)
        fraction, _, _ = immersion(self.model.geom_type[ATHLETE], self.model.geom_size[ATHLETE], geoms[:, :, :3],
                                   geoms[:, :, 3:], -self.height[ids, None])
        active = (fraction > 0) & ~self.surface_finished[ids]
        geometry = entry_geometry(sensor)
        hands = np.any(active[:, HAND_GEOMS], axis=1)
        legs = np.any(active[:, LEG_GEOMS], axis=1)
        trunk = np.any(active[:, TRUNK_GEOMS], axis=1)
        feet = active[:, FOOT_GEOMS]
        hand_parts = active[:, HAND_GEOMS]
        shins = active[:, SHIN_GEOMS]
        thighs = active[:, THIGH_GEOMS]
        shoulders = active[:, SHOULDER_GEOMS]
        forearms = active[:, FOREARM_GEOMS]
        knee_cross = np.any(thighs | shins, axis=1)
        ankle_cross = np.any(shins | feet, axis=1)
        toe_cross = np.any(feet, axis=1)
        metrics = np.concatenate([geometry['footLineAngles'], geometry['handAxisAngles'],
                                  np.stack([geometry[key] for key in ['handSeparation', 'handHeightGap', 'kneeGap',
                                                                      'ankleGap', 'toeGap']], axis=1)], axis=1)
        mask = np.concatenate([feet, hand_parts, np.stack([hands, hands, knee_cross, ankle_cross, toe_cross], axis=1)],
                              axis=1)
        self.entry_worst_geometry[ids] = np.maximum(self.entry_worst_geometry[ids], np.where(mask, metrics, 0))
        self.entry_crossed[ids] |= geometry['crossedLegs'] & legs
        head = self.headfirst[ids]
        angle = np.degrees(np.arccos(np.clip(np.where(head, -1, 1) * quat_up(state[:, 5:9])[:, 2], -1, 1)))
        self.entry_max_angle[ids] = np.maximum(self.entry_max_angle[ids], np.where(trunk | legs, angle, 0))
        joint = state[:, 1 + self.qadr]
        hips = joint[:, [0, 3]]
        knees = joint[:, [1, 4]]
        pitch = joint[:, [6, 9]] - np.where(head, 3.05, 0)[:, None]
        roll = joint[:, [7, 10]] - head[:, None] * np.array([-.3, .3])
        elbows = joint[:, [8, 11]]
        # Inspect arm placement from first water contact until the corresponding
        # segment is submerged, so raising the arms after the head enters cannot
        # erase the fault. Post-submersion movement cannot add entry penalties.
        arm_mask = ~self.surface_finished[ids][:, SHOULDER_GEOMS]
        elbow_mask = ~self.surface_finished[ids][:, FOREARM_GEOMS]
        hand_mask = ~self.surface_finished[ids][:, HAND_GEOMS]
        hand_window = np.any(hand_mask, axis=1) & head
        individual = np.stack([
            np.sum(1.25 * hips ** 2 * thighs, axis=1),
            np.sum(1.25 * knees ** 2 * shins, axis=1),
            np.sum(.325 * pitch ** 2 * arm_mask, axis=1),
            np.sum(.65 * roll ** 2 * arm_mask, axis=1),
            np.sum(.5 * elbows ** 2 * elbow_mask, axis=1),
            np.sum(.55 * geometry['footLineAngles'] ** 2 * feet, axis=1),
            np.sum(1.5 * np.maximum(geometry['handAxisAngles'] - np.pi / 12, 0) ** 2 * hand_mask, axis=1) * head,
            4 * np.maximum(geometry['handSeparation'] - .07, 0) ** 2 * hand_window,
            20 * geometry['handHeightGap'] ** 2 * hand_window,
            12 * np.maximum(geometry['kneeGap'] - .17, 0) ** 2 * knee_cross,
            12 * np.maximum(geometry['ankleGap'] - .13, 0) ** 2 * ankle_cross,
            12 * np.maximum(geometry['toeGap'] - .11, 0) ** 2 * toe_cross,
            4 * geometry['crossedLegs'] * toe_cross,
        ], axis=1)
        self.entry_fault_losses[ids] = np.maximum(self.entry_fault_losses[ids], individual)
        losses = np.stack([
            np.sum(1.25 * hips ** 2 * thighs, axis=1),
            np.sum(1.25 * knees ** 2 * shins, axis=1) + 12 * np.maximum(geometry['kneeGap'] - .17, 0) ** 2 * knee_cross,
            np.sum(.55 * geometry['footLineAngles'] ** 2 * feet, axis=1)
            + 12 * (np.maximum(geometry['ankleGap'] - .13, 0) ** 2 * ankle_cross
                    + np.maximum(geometry['toeGap'] - .11, 0) ** 2 * toe_cross)
            + 4 * geometry['crossedLegs'] * toe_cross,
            np.sum((.325 * pitch ** 2 + .65 * roll ** 2) * shoulders, axis=1),
            np.sum(.5 * elbows ** 2 * forearms, axis=1),
            np.sum(1.5 * np.maximum(geometry['handAxisAngles'] - np.pi / 12, 0) ** 2 * hand_parts, axis=1)
            + (4 * np.maximum(geometry['handSeparation'] - .07, 0) ** 2 + 20 * geometry['handHeightGap'] ** 2) * hands,
        ], axis=1)
        losses[:, -1] *= head
        self.surface_group_loss[ids] = np.maximum(self.surface_group_loss[ids], losses)
        # A heavy-tailed mapping keeps a slope toward clean form even for a badly bent entry.
        self.entry_min_form[ids] = 1 / (1 + self.surface_group_loss[ids].sum(axis=1))
        limbs = np.all((~thighs | (np.abs(hips) < .45)) & (~shins | (np.abs(knees) < .5))
                       & (~shoulders | ((np.abs(pitch) < .6) & (np.abs(roll) < .5)))
                       & (~forearms | (np.abs(elbows) < .5)), axis=1)
        hand_ok = (np.all(~hand_parts | (geometry['handAxisAngles'] < np.pi / 6), axis=1)
                   & (~hands | ((geometry['handSeparation'] < .12) & (geometry['handHeightGap'] < .04))))
        leg_ok = (np.all(~feet | (geometry['footLineAngles'] < np.radians(20)), axis=1)
                  & (~knee_cross | (geometry['kneeGap'] < .19)) & (~ankle_cross | (geometry['ankleGap'] < .16))
                  & (~toe_cross | ((geometry['toeGap'] < .15) & ~geometry['crossedLegs'])))
        self.entry_geometry_valid[ids] &= (~head | hand_ok) & leg_ok
        self.entry_limbs_valid[ids] &= limbs
        # Sensor pairs are [framelinvel, frameangvel] per geom. Lateral means sideways
        # relative to the line of flight (world y, rule 10.5.1), in metres per second.
        velocity = sensor[:, GEOM_VELOCITIES].reshape(len(ids), GEOM_COUNT, 6)[:, :, :3]
        lateral = np.max(np.where(active, np.abs(velocity[:, :, 1]), 0), axis=1)
        self.surface_lateral[ids] = np.maximum(self.surface_lateral[ids], lateral)
        self.surface_finished[ids] |= fraction >= 1
        complete = np.all(fraction == 1, axis=1)
        fresh = complete & ~np.isfinite(self.full_entry_time[ids])
        self.full_entry_time[ids[fresh]] = state[fresh, 0]

    # ------------------------------------------------------------------- step
    def step(self, action, auto_reset=True):
        """Advance every diver by one 20 ms control tick.

        Returns the raw sensor batch, the small shape-agnostic regularizer
        reward, the done flags and one measurement dict per finished dive.
        The trainer's observation is assembled by :mod:`environment`.
        """
        started_wet = np.isfinite(self.entry_time)
        old_state = self.state.copy()
        old_t = old_state[:, 0].copy()
        old_theta = self.theta.copy()
        old_twist = self.twist.copy()
        old_invalid = self.board_invalid.copy()
        self._apply_controls(action, old_t, started_wet)
        states, sensors, wet, wet_after = self._integrate(started_wet)
        geoms = sensors[:, :, GEOM_POSES].reshape(self.n, SUBSTEPS, GEOM_COUNT, 7)
        water = self._register_first_contact(states, sensors, geoms, old_state, started_wet)
        self._finish_scored_windows(wet, states, sensors, wet_after)
        self.state[:] = states[:, -1]
        self.sensors[:] = sensors[:, -1]
        if len(wet):
            self.sensors[wet] = wet_after[:, -1]
        fresh = np.flatnonzero(water)
        self.track_entry_batch(fresh, self.state[fresh], self.sensors[fresh])
        self.water_measurements()
        pitches, twists, laterals, ups = self._orientations(states)
        self.max_lateral = np.maximum(self.max_lateral, laterals.max(axis=1) * (~started_wet))
        trajectory_theta = old_theta[:, None] + np.cumsum(wrap(np.diff(np.concatenate([self.prev_pitch[:, None], pitches], axis=1), axis=1)), axis=1)
        twist_path = old_twist[:, None] + np.cumsum(wrap(np.diff(np.concatenate([self.prev_twist[:, None], twists], axis=1), axis=1)), axis=1)
        self._update_takeoff(states, sensors, geoms, old_t, started_wet, pitches, ups, trajectory_theta, twist_path)
        self.theta = np.where(started_wet, old_theta, trajectory_theta[:, -1])
        self.twist = np.where(started_wet, old_twist, twist_path[:, -1])
        self.prev_pitch = pitches[:, -1]
        self.prev_twist = twists[:, -1]
        contacted = np.isfinite(self.entry_time)
        done = self._finished(contacted, old_state)
        self._assess_flight(contacted)
        reward = self._regularizers(started_wet, contacted, old_invalid)
        info = [self._episode_info(i, contacted[i]) for i in np.flatnonzero(done)]
        if auto_reset and done.any():
            self.reset(np.flatnonzero(done))
        return self.sensors, reward.astype(np.float32), done, info

    def _apply_controls(self, action, old_t, started_wet):
        model = self.model
        desired = decoded_action(action)
        self.targets += np.clip(desired - self.targets, -TARGET_RATE * DT, TARGET_RATE * DT)
        self.ctrl[:, :, :model.nu] = self.targets[:, None, ACTUATOR_MAP]
        self.ctrl[:, :, model.nu:] = 0
        window = (old_t >= self.disturbance_time - 1e-8) & (old_t < self.disturbance_time + .12 - 1e-8)
        push = self.disturbance * (window & self.released & (~started_wet))
        self.ctrl[:, :, model.nu + TORSO_BODY * 6] = push[:, None]
        offset = model.nu + model.nbody * 6
        self.ctrl[:, :, offset] = self.platform[:, None]      # rigid platform equality
        self.ctrl[:, :, offset + 3] = -self.height[:, None]   # pool mocap z
        self.ctrl[:, :, offset + 4] = 1                       # pool mocap quaternion w

    def _integrate(self, started_wet):
        states, sensors = self.roll_state, self.roll_sensor
        dry = np.flatnonzero(~started_wet)
        wet = np.flatnonzero(started_wet)
        wet_after = None
        if len(dry):
            model = self.model
            stepped, sensed = self.pool.rollout([model] * len(dry), self.datas, self.state[dry], self.ctrl[dry],
                                                skip_checks=True, control_spec=CONTROL_SPEC, nstep=SUBSTEPS,
                                                state=np.empty((len(dry), SUBSTEPS, self.ns)),
                                                sensordata=np.empty((len(dry), SUBSTEPS, model.nsensordata)))
            states[dry] = stepped
            sensors[dry] = sensed
        if len(wet):
            states[wet], sensors[wet], wet_after = self.water_rollout(wet)
        return states, sensors, wet, wet_after

    def _vertical_extents(self, geoms):
        quaternions = geoms[:, :, :, 3:]
        w, x, y, z = np.moveaxis(quaternions, -1, 0)
        row = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1)
        sizes = self.model.geom_size[ATHLETE]
        extent = np.sqrt(np.sum((row * sizes) ** 2, axis=-1))
        for j, kind in enumerate(self.model.geom_type[ATHLETE]):
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent[:, :, j] = sizes[j, 0]
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent[:, :, j] = sizes[j, 0] + sizes[j, 1] * np.abs(row[:, :, j, 2])
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                extent[:, :, j] = np.sum(np.abs(row[:, :, j]) * sizes[j], axis=-1)
        return extent

    def _register_first_contact(self, states, sensors, geoms, old_state, started_wet):
        """Freeze each newly wet diver at exactly its first sampled contact state."""
        extent = self._vertical_extents(geoms)
        in_pool = ((geoms[:, :, :, 0] >= POOL_X[0]) & (geoms[:, :, :, 0] <= POOL_X[1])
                   & (np.abs(geoms[:, :, :, 1]) <= POOL_HALF_WIDTH))
        lowest = np.min(np.where(in_pool, geoms[:, :, :, 2] - extent, np.inf), axis=2)
        contact_steps = lowest <= -self.height[:, None]
        water = np.any(contact_steps, axis=1) & (~started_wet)
        for i in np.flatnonzero(water):
            k = int(np.argmax(contact_steps[i]))
            self.first_geometry[i] = int(np.argmin(np.where(in_pool[i, k], geoms[i, k, :, 2] - extent[i, k], np.inf))) + 1
            entry_state = old_state[i].copy() if k == 0 else states[i, k - 1].copy()
            states[i, k:] = entry_state
            sensors[i, k:] = sensors[i, k].copy()
            self.entry_time[i] = entry_state[0]
            self.first_state[i] = entry_state
            self.first_sensors[i] = sensors[i, k]
        return water

    def _finish_scored_windows(self, wet, states, sensors, wet_after):
        """Track only the scored interval; later underwater motion is not judged."""
        if not len(wet):
            return
        tracking = ~np.isfinite(self.full_entry_time[wet])
        for k in range(SUBSTEPS):
            active = tracking & (states[wet, k, 0] <= self.entry_time[wet] + ENTRY_WINDOW + 1e-9)
            self.track_entry_batch(wet[active], states[wet[active], k], wet_after[active, k])
            for j, i in enumerate(wet):
                submerged = np.isfinite(self.full_entry_time[i]) and states[i, k, 0] >= self.full_entry_time[i] - 1e-9
                expired = states[i, k, 0] >= self.entry_time[i] + ENTRY_WINDOW - 1e-9
                if tracking[j] and (submerged or expired):
                    tracking[j] = False
                    states[i, k:] = states[i, k].copy()
                    sensors[i, k:] = wet_after[j, k].copy()
                    wet_after[j, k:] = wet_after[j, k].copy()

    def _orientations(self, states):
        quaternions = states[:, :, 5:9].reshape(-1, 4)
        ups = quat_up(quaternions).reshape(self.n, SUBSTEPS, 3)
        pitch, twist, lateral = framed_angles(quaternions)
        return (pitch.reshape(self.n, SUBSTEPS), twist.reshape(self.n, SUBSTEPS),
                lateral.reshape(self.n, SUBSTEPS), ups)

    def _update_takeoff(self, states, sensors, geoms, old_t, started_wet, pitches, ups, trajectory_theta, twist_path):
        """Board and stand contact bookkeeping at every physics substep.

        Ground state comes only from actual athlete-to-board contacts, never
        self-touch. Non-foot assistance, a rotated recontact, staggered feet
        and a departure past horizontal invalidate the dive.
        """
        forces = board_forces(sensors)
        support = stand_forces(sensors)
        physical_dt = np.diff(np.concatenate([old_t[:, None], states[:, :, 0]], axis=1), axis=1)
        for k in range(SUBSTEPS):
            active = (physical_dt[:, k] > 0) & (~started_wet)
            allowed = np.zeros((self.n, GEOM_COUNT), bool)
            allowed[:, FOOT_GEOMS] = ~self.armstand[:, None]
            allowed[:, HAND_GEOMS] = self.armstand[:, None]
            assisting = np.where(allowed, 0, forces[:, k])
            self.board_impulse += np.sum(assisting, axis=1) * physical_dt[:, k]
            self.board_peak = np.maximum(self.board_peak, np.max(assisting, axis=1) * active)
            self.stand_impulse += np.sum(support[:, k], axis=1) * physical_dt[:, k]
            self.stand_peak = np.maximum(self.stand_peak, np.max(support[:, k], axis=1) * active)
            # A foot pressing the side or underside of the board is not a takeoff surface.
            foot_low = geoms[:, k, FOOT_GEOMS, 2] < states[:, k, 1, None] + .04
            self.foot_side_contact |= np.any(foot_low & (forces[:, k, FOOT_GEOMS] > ASSIST_FORCE), axis=1) & active & ~self.armstand
            grounded = np.sum(sensors[:, k, BOARD_FOUND], axis=1) > 0
            foot_contacts = np.where(self.armstand[:, None], sensors[:, k, HAND_CONTACT_FOUND] > 0,
                                     sensors[:, k, FOOT_CONTACT_FOUND] > 0)
            self.last_foot_contact = np.where(foot_contacts & active[:, None], states[:, k, 0, None], self.last_foot_contact)
            recontact = self.released & grounded & active
            # A preparation bounce is a two-footed hop (rule 8.6.5.2): contact regained
            # after an upward departure. Contact flicker while falling is not a hop.
            self.preparation_bounces += recontact & (self.takeoff_vertical_speed > HOP_SPEED)
            self.rotated_recontact |= recontact & (np.where(self.armstand, -ups[:, k, 2], ups[:, k, 2]) < .87)
            self.board_invalid |= ((self.board_impulse > ASSIST_IMPULSE) | (self.board_peak > ASSIST_FORCE)
                                   | (self.stand_impulse > ASSIST_IMPULSE) | (self.stand_peak > ASSIST_FORCE)
                                   | self.rotated_recontact | self.foot_side_contact)
            landed = grounded & active
            self.air_clear_time[landed] = 0
            self.departure_com[landed] = sensors[landed, k, 2]
            self.apex_com[landed] = sensors[landed, k, 2]
            self.released[landed] = False
            self.max_lateral[landed] = 0
            first_clear = (self.air_clear_time == 0) & (~grounded) & active
            self.departure_com[first_clear] = sensors[first_clear, k, 2]
            self.apex_com[first_clear] = sensors[first_clear, k, 2]
            self.takeoff_vertical_speed[first_clear] = sensors[first_clear, k, 5]
            self.takeoff_angular_momentum[first_clear] = sensors[first_clear, k, 6:9]
            self.departure_pitch[first_clear] = pitches[first_clear, k]
            self.departure_theta[first_clear] = trajectory_theta[first_clear, k]
            self.departure_twist[first_clear] = twist_path[first_clear, k]
            self.apex_com = np.where((~grounded) & active, np.maximum(self.apex_com, sensors[:, k, 2]), self.apex_com)
            self.air_clear_time += (~grounded) * physical_dt[:, k]
            newly = (~self.released) & (self.air_clear_time >= RELEASE_AIR_TIME - 1e-8) & (states[:, k, 0] > .08) & active
            self.released[newly] = True
            self.release_time[newly] = states[newly, k, 0] - self.air_clear_time[newly] + .002
            self.release_theta[newly] = self.departure_theta[newly]
            self.release_twist[newly] = self.departure_twist[newly]
            self.foot_departure_gap[newly] = np.abs(self.last_foot_contact[newly, 0] - self.last_foot_contact[newly, 1])
            # Leaving the board past horizontal means the athlete rolled off the edge; a
            # steep but earlier departure is a poor takeoff, scored by the judge, not a failure.
            departure = np.where(self.armstand, np.abs(np.abs(self.departure_pitch) - np.pi), np.abs(self.departure_pitch))
            self.takeoff_tilt_invalid |= newly & (departure > np.pi / 2)
            self.board_invalid |= self.takeoff_tilt_invalid | (newly & (self.foot_departure_gap > .04 + 1e-8))
            self.air_theta = np.where(started_wet, self.air_theta,
                                      np.where(self.released, trajectory_theta[:, k] - self.release_theta, 0))
            start_pitch = np.where(self.armstand, wrap(self.departure_pitch - np.pi), self.departure_pitch)
            self.phase_theta = np.where(started_wet, self.phase_theta, np.where(self.released, self.air_theta + start_pitch, 0))
            self.air_twist = np.where(started_wet, self.air_twist,
                                      np.where(self.released, twist_path[:, k] - self.release_twist, 0))
            self.max_lateral = np.maximum(self.max_lateral, np.abs(ups[:, k, 1]) * self.released * (~started_wet))

    def _finished(self, contacted, old_state):
        # MuJoCo silently resets an unstable world to its XML pose. An athlete cannot
        # move a metre in one 20 ms step, so such a jump ends the dive as a failure.
        jumped = np.linalg.norm(self.state[:, 2:5] - old_state[:, 2:5], axis=1) > 1.
        self.diverged_events += int(jumped.sum())
        elapsed = self.state[:, 0]
        return (np.isfinite(self.full_entry_time) | (contacted & (elapsed - self.entry_time >= ENTRY_WINDOW - 1e-9))
                | ((~contacted) & (elapsed >= TIMEOUT)) | (~np.isfinite(self.state).all(axis=1)) | jumped)

    def _assess_flight(self, contacted):
        """Accumulate position quality inside the declared rotation window."""
        sensors = self.sensors
        joint = self.state[:, 1 + self.qadr]
        hip, knee = joint[:, 0], joint[:, 1]
        omega = np.linalg.norm(sensors[:, ANGULAR_VELOCITY], axis=1)
        self.entry_omega = np.where(contacted, np.maximum(self.entry_omega, omega), self.entry_omega)
        quality = position_qualities(hip, knee, np.max(tuck_geometry(sensors)[0], axis=1))
        turns = self.goals[:, 0]
        progress = self.phase_theta / np.where(np.abs(turns) > 1e-6, turns * 2 * np.pi, 1)
        assess = self.released & ~contacted & (((progress > .15) & (progress < .8)) | (turns == 0))
        self.position_sums += quality * assess[:, None]
        self.position_ticks += assess
        # Learning feedback is available even when rotation is wrong. Entry
        # preparation gradually replaces the flight shape, rather than requiring
        # the athlete to remain tucked while entering the water.
        from motor_objective import entry_weight
        weight = self.released * ~contacted * (1 - entry_weight(self.above_water, sensors[:, 5]))
        self.motor_position_sums += quality * weight[:, None]
        self.motor_position_weight += weight

    def _regularizers(self, started_wet, contacted, old_invalid):
        """Shape-agnostic small physical regularizers; no desired rotation or twist."""
        velocity = self.state[:, 1 + self.model.nq:]
        geometry = entry_geometry(self.sensors)
        airborne = self.released * ~contacted
        reward = -.0005 * np.mean((velocity[:, self.vadr] / 12) ** 2, axis=1)
        reward -= .001 * np.sum(geometry['footLineAngles'] ** 2, axis=1) * airborne
        reward -= .002 * np.maximum(geometry['ankleGap'] - .13, 0) * airborne
        reward -= .2 * (self.board_invalid & ~old_invalid)
        return np.where(started_wet, 0, reward)

    def _episode_info(self, i, contacted):
        state = self.first_state[i] if contacted else self.state[i]
        sensors = self.first_sensors[i] if contacted else self.sensors[i]
        sign = -1 if self.headfirst[i] else 1
        first_angle = float(np.degrees(np.arccos(np.clip(sign * quat_up(state[5:9][None])[0, 2], -1, 1))))
        worst = self.entry_worst_geometry[i]
        arm_reference = 3.05 if self.headfirst[i] else 0
        lean = np.abs(np.abs(self.departure_pitch[i]) - np.pi) if self.armstand[i] else np.abs(self.departure_pitch[i])
        return dict(
            index=int(i), rotation=float(self.phase_theta[i] / (2 * np.pi)), twist=float(self.air_twist[i] / (2 * np.pi)),
            boardInvalid=bool(self.board_invalid[i]), water=bool(contacted),
            fullEntryComplete=bool(np.isfinite(self.full_entry_time[i])), firstGeometry=int(self.first_geometry[i]),
            x=float(sensors[0]), firstContactAngle=first_angle, entryAngle=float(max(first_angle, self.entry_max_angle[i])),
            entryFaultLosses={name: float(self.entry_fault_losses[i, j]) for j, name in enumerate(ENTRY_BUDGETS)},
            form=float(self.entry_min_form[i]), entryGeometryValid=bool(self.entry_geometry_valid[i]),
            entryLimbsValid=bool(self.entry_limbs_valid[i]),
            entryGeometryWorst=dict(footLineAngles=np.degrees(worst[:2]).tolist(), handAxisAngles=np.degrees(worst[2:4]).tolist(),
                                    **{key: float(worst[4 + j]) for j, key in enumerate(['handSeparation', 'handHeightGap', 'kneeGap', 'ankleGap', 'toeGap'])},
                                    crossedLegs=bool(self.entry_crossed[i])),
            entryArmPositionValid=bool(np.max(np.abs(state[1 + self.qadr[[6, 9]]] - arm_reference)) < .8),
            positionQualities=(self.position_sums[i] / max(1, self.position_ticks[i])).tolist(),
            motorPositionQualities=(self.motor_position_sums[i] / max(1e-8, self.motor_position_weight[i])).tolist(),
            ascent=float(max(0, self.apex_com[i] - self.departure_com[i])), preparationBounces=int(self.preparation_bounces[i]),
            surfaceLateralSpeed=float(self.surface_lateral[i]), entryAngularSpeed=float(self.entry_omega[i]),
            maxLateral=float(self.max_lateral[i]), takeoffVerticalSpeed=float(self.takeoff_vertical_speed[i]),
            takeoffAngularMomentum=self.takeoff_angular_momentum[i].tolist(),
            departureLean=float(np.degrees(lean)),
            time=float(self.state[i, 0]))
