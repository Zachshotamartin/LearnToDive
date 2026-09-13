"""Self-declared diving routines on top of the physics arena.

One policy selects a declaration for each round, then drives real joint
servos. The trainer never selects dive targets or supplies motion
demonstrations; the only physical priors are the standing and armstand
starting poses, and recovery snapshots originate from this learner's own
dives.
"""
import copy

import numpy as np

from engine import ACTION_HIGH, ACTION_LOW, SCRATCH, Arena as Physics, board_forces, framed_angles, initial_state
from geometry import FOOT_GEOMS
from judge import judge
from training_reward import reward_components, MODES
from rules import CODE_INDEX, CODES, DIVES, POSITIONS, legal_mask, validate_declaration
from stance import balanced_armstand
from recovery_curriculum import RecoveryArchive, recovery_stage
from motor_objective import potential_components, shaped_reward

PLATFORM_HEIGHTS = [5, 7.5, 10]
SPRINGBOARD_HEIGHTS = [1, 3]
SPRINGBOARD_PRELOAD = -.05123  # board deflection under a standing athlete, metres
SUPPORT_FORCE = 15             # newtons under a foot that count as standing on the board
AUXILIARY_RATE = .25           # fraction of training routines that practise one weak group
ROUTINE_LENGTH = 6
BANK_LIMIT = 96
POSITION_INDEX = {'C': 0, 'B': 1, 'D': 2, 'A': 3}
BACK_SKILL = 3                 # engine skill whose stance faces away from the water
FORWARD_SKILL = 0


def rows(engine):
    """Every per-environment array of the physics arena; scratch buffers excluded."""
    return {name: value for name, value in vars(engine).items()
            if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == engine.n and name not in SCRATCH}


class Arena:
    """A batch of routines: six declared dives per athlete, or one auxiliary practice dive."""

    def __init__(self, n=64, seed=1, threads=4, training=True, practice=.35, reward_mode="v12", recovery_mode="recent"):
        if reward_mode not in MODES:
            raise ValueError("Unknown reward mode")
        self.reward_mode = reward_mode
        if recovery_mode not in ("recent", "progress"):
            raise ValueError("Unknown recovery curriculum")
        self.recovery_mode = recovery_mode
        self.recovery_archive = RecoveryArchive()
        self.perturb = False
        self.n = n
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.practice_rate = practice
        self.physics = Physics(n, seed, threads, skills=[FORWARD_SKILL], heights=(1, 10), minimum_heights=[1] * 6,
                               training=training)
        # The normalized action that reproduces the evaluation stance; a policy whose
        # initial mean holds still is the most neutral prior, not a prescribed motion.
        _, _, stance = initial_state(self.physics.model, self.physics.resetdata, 10, FORWARD_SKILL)
        self.initial_action = (2 * (stance - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW) - 1).astype(np.float32)
        self.previous_actions = np.zeros((n, 9), np.float32)
        self.choosing = np.ones(n, bool)
        self.declaration = np.zeros(n, int)
        self.group = np.ones(n, int)
        self.apparatus = np.zeros(n, int)
        self.height = np.zeros(n)
        self.used = np.zeros((n, len(CODES)), bool)
        self.auxiliary = np.zeros(n, bool)
        self.round = np.zeros(n, int)
        self.schedule = np.zeros((n, ROUTINE_LENGTH), int)
        self.routine_points = np.zeros(n)
        self.returns = np.zeros(n)
        self.practice = np.zeros(n, bool)
        self.recovery_allowed = np.ones(n, bool)
        self.bank = [[] for _ in range(ROUTINE_LENGTH)]
        self.banked_stages = np.zeros((n, 3), bool)
        self.selected_recovery = np.full(n, -1, dtype=int)
        self.attempts = np.zeros(ROUTINE_LENGTH)
        self.clean = np.zeros(ROUTINE_LENGTH)
        self.interactions = 0
        self.potential_names = ('takeoff', 'flightPosition', 'entryAlignment', 'entryHip', 'entryKnee',
                                'entryShoulderPitch', 'entryShoulderRoll', 'entryElbow', 'entryToes', 'entryHands', 'entryLegs')
        self.phase_previous = np.zeros((n, len(self.potential_names)))
        self.phase_totals = np.zeros_like(self.phase_previous)
        self.reset(np.arange(n), new_routine=True)
        self.observation_size = self.observe().shape[-1]

    def close(self):
        self.physics.close()

    # ------------------------------------------------------------ routines
    def reset(self, ids, new_routine=False):
        for i in ids:
            if new_routine or self.round[i] >= self.rounds(i):
                self.new_routine(i)
            self.group[i] = self.schedule[i, self.round[i]]
            self.choosing[i] = True
            self.returns[i] = 0
            self.practice[i] = False
            self.banked_stages[i] = False
            self.selected_recovery[i] = -1
            self.previous_actions[i] = 0
            self.phase_previous[i] = 0
            self.phase_totals[i] = 0
            self.stand(i, FORWARD_SKILL)
        return self.observe()

    def rounds(self, i):
        return 1 if self.auxiliary[i] else ROUTINE_LENGTH

    def new_routine(self, i):
        """Draw apparatus, height and the order of the six dive groups."""
        rng = self.rng
        self.apparatus[i] = int(rng.integers(2))
        self.height[i] = float(rng.choice(PLATFORM_HEIGHTS) if self.apparatus[i] else rng.choice(SPRINGBOARD_HEIGHTS))
        groups = [1, 2, 3, 4, 5, 6] if self.apparatus[i] else [1, 2, 3, 4, 5, int(rng.integers(1, 6))]
        self.schedule[i] = rng.permutation(groups)
        self.auxiliary[i] = self.training and rng.random() < AUXILIARY_RATE
        if self.auxiliary[i]:
            # Practise the group with the lowest clean rate more often.
            weights = .2 + 1 - self.clean / np.maximum(1, self.attempts)
            weights *= np.array([1, 1, 1, 1, 1, self.apparatus[i]])
            self.schedule[i, 0] = int(rng.choice(np.arange(1, 7), p=weights / weights.sum()))
        self.round[i] = 0
        self.used[i] = False
        self.routine_points[i] = 0

    def stand(self, i, skill):
        """Place athlete ``i`` in the standing start for its apparatus and height."""
        platform = bool(self.apparatus[i])
        context = dict(skill=skill, height=self.height[i], preload=0 if platform else SPRINGBOARD_PRELOAD,
                       disturbance=0, platform=platform)
        self.physics.reset([i], [context])
        self.physics.platform[i] = platform
        self.physics.armstand[i] = False

    def apparatus_name(self, i):
        return 'platform' if self.apparatus[i] else 'springboard'

    def mask(self):
        return np.stack([legal_mask(int(self.group[i]), self.apparatus_name(i), self.height[i],
                                    np.asarray(CODES)[self.used[i]]) for i in range(self.n)])

    # ---------------------------------------------------------- observation
    def observe(self):
        e = self.physics
        q = e.state[:, 1:1 + e.model.nq]
        v = e.state[:, 1 + e.model.nq:]
        s = e.sensors
        # No ballistic future estimate, desired angular speed, or externally assigned
        # maneuver. Intent is visible only after the policy has committed to it.
        intent = np.zeros((self.n, 9))
        for i in np.flatnonzero(~self.choosing):
            d = DIVES[self.declaration[i]]
            intent[i, :5] = [d['sign'] * d['turns'] / 5, d['twists'] / 5, d['back'], d['armstand'], d['direction'] / 4]
            intent[i, 5 + POSITIONS.index(d['position'])] = 1
        # Foot contact flags and the board spring state are physical sensors the
        # athlete has (pressure under the feet, the board moving); balance and the
        # press need them. Nothing here is a future estimate or an assigned target.
        support = np.concatenate([(board_forces(s)[:, FOOT_GEOMS] > SUPPORT_FORCE).astype(float),
                                  q[:, 0:1] * 4, v[:, 0:1] / 3], axis=1)
        progress = np.stack([e.phase_theta / (2 * np.pi), e.air_twist / (2 * np.pi), e.height / 10, e.state[:, 0] / 4,
                             e.released, e.platform, e.armstand, e.water_fraction, e.above_water / 10, self.choosing,
                             self.round / 6], axis=1)
        observation = np.concatenate([q[:, 4:8], v[:, 1:7] / 10, q[:, e.qadr] / 3, v[:, e.vadr] / 15, e.targets / 3,
                                      s[:, :3] / [4, 2, 10], s[:, 3:6] / 10, progress, support,
                                      np.eye(6)[self.group - 1], intent, self.used.astype(float),
                                      self.previous_actions], axis=1)
        return observation.clip(-10, 10).astype(np.float32)

    # ---------------------------------------------------------- declaration
    def declare(self, i, choice):
        d = validate_declaration(choice, int(self.group[i]), self.apparatus_name(i), self.height[i],
                                 np.asarray(CODES)[self.used[i]])
        e = self.physics
        self.declaration[i] = choice
        e.goals[i] = [d['sign'] * d['turns'], d['twists'], d['back'], POSITION_INDEX[d['position']]]
        # Legal starting orientation follows the agent's chosen dive. There is no
        # modification to root state after launch.
        self.stand(i, BACK_SKILL if d['back'] else FORWARD_SKILL)
        e.armstand[i] = d['armstand']
        e.headfirst[i] = d['headfirst']
        e.height[i] = self.height[i]
        if d['armstand']:
            e.sensors[i], e.targets[i] = balanced_armstand(e.model, e.resetdata, e.state[i], d['back'])
            e.prev_pitch[i], e.prev_twist[i], _ = [x[0] for x in framed_angles(e.state[i, 5:9][None])]
        if self.perturb:
            e.disturbance[i] = 30 if i % 2 else -30
            e.disturbance_time[i] = .75
        self.choosing[i] = False
        self.maybe_practice(i, choice)

    def maybe_practice(self, i, choice):
        """Occasionally restart from a banked mid-air snapshot of the same declaration.

        Snapshots come only from this learner's real dives with the same
        declaration, apparatus and height; a target is never assigned.
        """
        if not self.recovery_allowed[i]:
            return
        g = self.group[i] - 1
        rate = self.clean[g] / max(1, self.attempts[g])
        probability = self.practice_rate * (1 - rate)
        if self.recovery_mode == 'progress':
            if self.training and self.rng.random() < probability:
                selected = self.recovery_archive.choose(g, choice, self.height[i], self.apparatus[i], self.rng)
                if selected is not None:
                    for name, value in selected['snapshot']['physics'].items():
                        getattr(self.physics, name)[i] = value
                    self.selected_recovery[i] = selected['id']
                    self.practice[i] = True
            return
        eligible = [x for x in self.bank[g] if x['declaration'] == choice and x['height'] == self.height[i]
                    and x['apparatus'] == self.apparatus[i]]
        if self.training and eligible and self.rng.random() < probability:
            snapshot = eligible[int(self.rng.integers(len(eligible)))]
            for name, value in snapshot['physics'].items():
                getattr(self.physics, name)[i] = value
            self.practice[i] = True

    def bank_snapshot(self, i):
        """Bank a descending, airborne, not yet wet state as a future recovery start."""
        e = self.physics
        snapshot = dict(declaration=int(self.declaration[i]), height=float(self.height[i]),
                        apparatus=int(self.apparatus[i]), physics={k: v[i].copy() for k, v in rows(e).items()})
        if self.recovery_mode == 'progress':
            ratio = float(e.above_water[i] / max(.1, self.height[i]))
            stage = recovery_stage(ratio)
            features = [ratio, e.sensors[i, 5] / 10, e.phase_theta[i] / (2 * np.pi), e.air_twist[i] / (2 * np.pi)]
            self.recovery_archive.add(self.group[i] - 1, snapshot, features, stage)
            self.banked_stages[i, stage] = True
            return
        bank = self.bank[self.group[i] - 1]
        bank.append(snapshot)
        if len(bank) > BANK_LIMIT:
            bank.pop(0)
        e.banked[i] = True

    def bankable(self, i, chosen):
        e = self.physics
        if self.recovery_mode == 'progress':
            stage = recovery_stage(e.above_water[i] / max(.1, self.height[i]))
            return (self.training and i not in chosen and not self.practice[i] and e.released[i]
                    and not np.isfinite(e.entry_time[i]) and e.sensors[i, 5] < 0
                    and e.above_water[i] > .3 and not self.banked_stages[i, stage])
        return (self.training and i not in chosen and not self.practice[i] and not e.banked[i] and e.released[i]
                and not np.isfinite(e.entry_time[i]) and e.sensors[i, 5] < 0 and .3 < e.above_water[i] < 3)

    # ----------------------------------------------------------------- step
    def step(self, choices, actions):
        e = self.physics
        chosen = np.flatnonzero(self.choosing)
        # Declare before any integration. Masked actions during declaration do not
        # contribute motor likelihoods or move the body.
        for i in chosen:
            self.declare(i, int(choices[i]))
        frozen = {k: v[chosen].copy() for k, v in rows(e).items()}
        _, reward, done, info = e.step(actions, auto_reset=False)
        self.previous_actions[:] = actions
        self.previous_actions[chosen] = 0
        for k, value in frozen.items():
            getattr(e, k)[chosen] = value
        reward[chosen] = 0
        done[chosen] = False
        info = [x for x in info if x['index'] not in chosen]
        self.interactions += self.n
        for i in range(self.n):
            if self.bankable(i, chosen):
                self.bank_snapshot(i)
        if self.reward_mode == 'phase-dense':
            parts = potential_components(self)
            current = np.column_stack([parts[name] for name in self.potential_names])
            shaped = shaped_reward(self.phase_previous, current, done[:, None])
            self.phase_previous[:] = np.where(done[:, None], 0., current)
            self.phase_totals += shaped
            reward += shaped.sum(axis=1)
        completed = [self.complete(m, reward) for m in info]
        self.returns += reward
        for record in completed:
            record['return'] = float(self.returns[record['index']])
        if done.any():
            self.reset(np.flatnonzero(done))
        return self.observe(), reward, done, completed

    def complete(self, m, reward):
        """Judge one finished dive, credit its terminal reward and advance the routine."""
        i = m['index']
        d = DIVES[self.declaration[i]]
        m['positionQuality'] = m['positionQualities'][POSITIONS.index(d['position'])]
        m['motorPositionQuality'] = m['motorPositionQualities'][POSITIONS.index(d['position'])]
        score = judge(self.declaration[i], self.apparatus_name(i), self.height[i], m)
        components = reward_components(score, m, self.reward_mode)
        reward[i] += components["total"]
        score["trainingRewardComponents"] = components
        if self.reward_mode == 'phase-dense':
            score['potentialShaping'] = dict(zip(self.potential_names, self.phase_totals[i].tolist()))
        g = self.group[i] - 1
        if self.practice[i] and self.recovery_mode == 'progress':
            # This affects only reset sampling, never the competition or training reward.
            quality = max(0., 1 - score['entryAngle'] / 90) * m['positionQuality']
            quality /= 1 + score['rotationError'] + score['twistError']
            self.recovery_archive.observe(g, self.selected_recovery[i], quality)
        if not self.practice[i]:
            self.attempts[g] += 1
            self.clean[g] += score['clean']
        self.routine_points[i] += score['points']
        self.used[i, CODE_INDEX[d['code']]] = True
        self.round[i] += 1
        return dict(**score, measurements=m, practice=bool(self.practice[i]), height=float(self.height[i]),
                    apparatus=self.apparatus_name(i), routinePoints=float(self.routine_points[i]),
                    routineFinished=bool(self.round[i] == ROUTINE_LENGTH and not self.auxiliary[i]),
                    auxiliaryPractice=bool(self.auxiliary[i]), index=i)

    # ------------------------------------------------------------ persistence
    def state_dict(self):
        return dict(physics={k: v.copy() for k, v in rows(self.physics).items()},
                    arrays={k: v.copy() for k, v in vars(self).items() if isinstance(v, np.ndarray)},
                    bank=copy.deepcopy(self.bank), recoveryArchive=self.recovery_archive.state_dict(), rng=self.rng.bit_generator.state,
                    physicsRng=self.physics.rng.bit_generator.state, interactions=self.interactions)

    def load_state_dict(self, state):
        for k, v in state['physics'].items():
            getattr(self.physics, k)[:] = v
        for k, v in state['arrays'].items():
            getattr(self, k)[:] = v
        self.bank = copy.deepcopy(state['bank'])
        self.recovery_archive.load_state_dict(state['recoveryArchive'])
        self.rng.bit_generator.state = state['rng']
        self.physics.rng.bit_generator.state = state['physicsRng']
        self.interactions = state['interactions']
