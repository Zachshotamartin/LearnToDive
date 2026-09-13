"""On-policy reusable motor practice alongside autonomous full dives.

Practice tasks contain no prescribed dive numbers or motion trajectories.
Randomized airborne resets are explicitly practice, never scored as full dives.
The same torque-limited actor learns both; evaluation uses only real board starts.
All task goals, readiness, statistics and RNG state are checkpointed.
"""
import copy
import mujoco
import numpy as np

from engine import STATE_SPEC
from geometry import ACTUATOR_MAP, framed_angles, quat_up, ANGULAR_VELOCITY
from motor_objective import live_errors
from stance import orientation

TASKS = ('full', 'takeoff', 'shape', 'aerial', 'entry')
EXTRA_OBSERVATIONS = 10  # five task indicators, hip/knee goals, three-axis up goal
HORIZONS = np.array([0, 1.4, 1., 1., 3.])


class MotorCurriculum:
    def __init__(self, base, enabled=True):
        self.base = base
        self.enabled = bool(enabled and base.training)
        self.n = base.n
        self.task = np.zeros(self.n, int)
        self.shape_goal = np.zeros((self.n, 2))
        self.up_goal = np.zeros((self.n, 3))
        self.up_goal[:, 2] = -1
        self.age = np.zeros(self.n)
        self.task_returns = np.zeros(self.n)
        self.previous_cost = np.zeros(self.n)
        self.visits = np.zeros(5, int)
        self.successes = np.zeros(5, int)
        self.readiness = np.zeros(5)
        self.level = np.zeros(5)
        self.error_sum = np.zeros(5)
        self.last_completed = []
        self.assign(np.arange(self.n))

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def observation_size(self):
        return self.base.observation_size + EXTRA_OBSERVATIONS

    @property
    def practice(self):
        return self.base.practice | (self.task != 0)

    def assign(self, ids):
        for i in ids:
            self.task[i] = 0
            if self.enabled:
                # Full-from-board rollouts remain at least 25% from the start.
                share = max(.2, .75 * (1 - float(self.readiness[1:].min())))
                if self.base.rng.random() < share:
                    weights = .15 + 1 - self.readiness[1:]
                    self.task[i] = 1 + self.base.rng.choice(4, p=weights / weights.sum())
            self.age[i] = self.task_returns[i] = self.previous_cost[i] = 0
            self.shape_goal[i] = [(0, 0), (1.5, 0), (1.4, 2)][self.base.rng.integers(3)] if self.task[i] == 2 else (0, 0)
            self.up_goal[i] = [0, 0, -1 if self.task[i] == 4 else 1]
            if self.task[i] == 3:
                theta = self.base.rng.uniform(-np.pi, np.pi)
                self.up_goal[i] = [np.sin(theta), 0, np.cos(theta)]

    def observe(self):
        obs = self.base.observe()
        # The autoregressive action history must remain the final nine columns.
        goals = np.column_stack([np.eye(5)[self.task], self.shape_goal / 3, self.up_goal])
        goals[self.task == 0, 5:] = 0  # Full dives use only their own declared intent.
        return np.column_stack([obs[:, :-9], goals, obs[:, -9:]]).astype(np.float32)

    def initialize_task(self, i):
        task = self.task[i]
        if not task:
            return
        e, rng = self.base.physics, self.base.rng
        self.base.practice[i] = True
        if task == 1:
            return  # Real foot/hand contact; no injected takeoff velocity.
        d = mujoco.MjData(e.model)
        mujoco.mj_setState(e.model, d, e.state[i], STATE_SPEC)
        level = self.level[task]
        pitch = np.arctan2(self.up_goal[i, 0], self.up_goal[i, 2])
        if task == 2:
            pitch = 0.
        pitch += rng.uniform(-.15 - .5 * level, .15 + .5 * level)
        d.qpos[4:8] = orientation(pitch, 0)
        d.qpos[1:4] = [3, 0, (-e.height[i] + rng.uniform(2.2, 3.2)) if task == 4 else 4]
        d.qvel[:] = 0
        if task == 4:
            d.qvel[3] = -rng.uniform(.5, 2 + 2 * level)
        if task in (3, 4):
            d.qvel[4:7] = rng.uniform(-.2 - 2 * level, .2 + 2 * level, 3)
        target = np.array([rng.uniform(0, .7), rng.uniform(0, .9), 1.2,
                           rng.uniform(0, 3.05), rng.uniform(0, 3.05), -.2, .2,
                           rng.uniform(0, 1.), .04])
        if task == 2:
            target[:2] = rng.uniform(0, [1.7, 2.2])
        d.qpos[e.qadr] = target[ACTUATOR_MAP]
        d.ctrl[:] = target[ACTUATOR_MAP]
        mujoco.mj_forward(e.model, d)
        mujoco.mj_getState(e.model, d, e.state[i], STATE_SPEC)
        e.sensors[i] = d.sensordata
        e.targets[i] = target
        e.released[i] = True
        e.armstand[i] = False
        e.headfirst[i] = task == 4
        e.departure_com[i] = e.apex_com[i] = d.sensordata[2]
        e.prev_pitch[i], e.prev_twist[i], _ = [x[0] for x in framed_angles(e.state[i, 5:9][None])]
        e.departure_pitch[i] = e.prev_pitch[i]
        # No practice trajectory or reward ever enters the full-dive archive.
        e.banked[i] = True

    def costs(self):
        e = self.base.physics
        q = e.state[:, 1 + e.qadr]
        errors = live_errors(e)
        up = quat_up(e.state[:, 5:9])
        gap = np.mean(np.abs(q[:, :2] - self.shape_goal) / [.8, 1.], axis=1)
        orientation_error = np.arccos(np.clip(np.sum(up * self.up_goal, axis=1), -1, 1))
        speed = np.linalg.norm(e.sensors[:, ANGULAR_VELOCITY], axis=1)
        shape = gap + .15 * errors['toes'] + .1 * errors['legs']
        aerial = orientation_error + .1 * speed
        entry = (errors['alignment'] + .5 * errors['hip'] + errors['knee']
                 + 1.5 * errors['shoulderPitch'] + errors['elbow'] + .5 * errors['shoulderRoll']
                 + errors['toes'] + errors['hands'] + errors['legs'])
        rise = np.maximum(0, e.apex_com - e.departure_com)
        takeoff = (np.maximum(0, 1 - rise / .3) + .5 * np.maximum(0, 1 - e.takeoff_vertical_speed / 2.8)
                   + 2 * e.board_invalid)
        # Armstand practice rewards a genuine clear release, not an impossible
        # standing-jump target from a hand-supported starting pose.
        takeoff = np.where(e.armstand, (~e.released).astype(float) + 2 * e.board_invalid, takeoff)
        return np.choose(self.task, [np.zeros(self.n), takeoff, shape, aerial, entry]).clip(0, 20)

    def step(self, choices, actions):
        self.last_completed = []
        chosen = self.base.choosing.copy()
        tasks = self.task.copy()
        previous = self.costs()
        self.base.recovery_allowed[:] = tasks == 0
        _, reward, done, rows = self.base.step(choices, actions)
        finished = {row['index']: row for row in rows}
        for i in np.flatnonzero(chosen):
            self.initialize_task(i)
        cost = self.costs()
        self.age += .02 * ~chosen
        skill = tasks != 0
        # Fixed practice duration removes an incentive to prolong a good pose or
        # terminate early to evade accumulated error.
        timed = skill & ~chosen & (self.age >= HORIZONS[tasks] - 1e-8)
        ended = skill & (done | timed)
        for i in np.flatnonzero(skill):
            if chosen[i]:
                reward[i] = 0
                continue
            if done[i]:
                m = finished[i]['measurements']
                if tasks[i] == 4:
                    cost[i] = (finished[i]['entryAngle'] / 45 + sum(np.log1p(v) for v in m['entryFaultLosses'].values())
                               + 3 * (not m['fullEntryComplete']))
                elif tasks[i] == 1:
                    cost[i] = (max(0, 1 - m['ascent'] / .3) + .5 * max(0, 1 - m['takeoffVerticalSpeed'] / 2.8)
                               + 2 * m['boardInvalid']) if finished[i]['category'] != 6 else 2 * m['boardInvalid'] + float(not m['water'])
                else:
                    cost[i] = 20.  # An early crash cannot improve a motor task.
            # Direct task error plus a progress term. Terminal cancellation keeps
            # the progress term from adding a second final-pose reward.
            reward[i] = -.02 * cost[i] + previous[i] - (.995 * cost[i] if not ended[i] else 0)
            if ended[i]:
                reward[i] -= 2 * cost[i]
                if done[i] and tasks[i] in (2, 3):
                    reward[i] -= 20 * max(0, HORIZONS[tasks[i]] - self.age[i])
                threshold = [0, .7, .2, .25, .8][tasks[i]]
                success = cost[i] < threshold
                task = tasks[i]
                self.last_completed.append(dict(index=int(i), task=int(task), error=float(cost[i]), success=bool(success)))
                self.visits[task] += 1
                self.successes[task] += success
                self.readiness[task] += .03 * (success - self.readiness[task])
                self.error_sum[task] += cost[i]
                if self.visits[task] >= 64 and self.readiness[task] > .65:
                    self.level[task] = min(1., self.level[task] + .01)
        self.task_returns += reward * skill
        # A practice declaration, score or used-dive mask must never become the
        # starting history of a subsequent autonomous competition routine.
        if ended.any():
            self.base.reset(np.flatnonzero(ended), new_routine=True)
        done |= ended
        if done.any():
            self.assign(np.flatnonzero(done))
        # Motor practice cannot inflate competition metrics or choose champions.
        rows = [row for row in rows if tasks[row['index']] == 0]
        return self.observe(), reward, done, rows

    def metrics(self):
        return {TASKS[t]: dict(episodes=int(self.visits[t]), successes=int(self.successes[t]),
                              readiness=float(self.readiness[t]), level=float(self.level[t]),
                              meanError=float(self.error_sum[t] / max(1, self.visits[t]))) for t in range(1, 5)}

    def state_dict(self):
        arrays = {k: v.copy() for k, v in vars(self).items() if isinstance(v, np.ndarray)}
        return dict(base=self.base.state_dict(), curriculum=arrays, enabled=self.enabled)

    def load_state_dict(self, state):
        if state['enabled'] != self.enabled:
            raise ValueError('Cannot resume with changed motor curriculum')
        self.base.load_state_dict(state['base'])
        for k, v in state['curriculum'].items():
            getattr(self, k)[:] = copy.deepcopy(v)
