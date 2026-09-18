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
from motor_objective import live_errors, live_takeoff_legality, rotation_direction_cost, entry_weight, takeoff_legality
from stance import orientation
from rules import DIVES, IDS, in_practice_scope
from environment import HISTORY

TASKS = ('full', 'takeoff', 'shape', 'aerial', 'entry')
ENTRY_POSE = np.array([0, 0, 1.2, 3.05, 3.05, -.3, .3, 0, .06])   # straight, arms overhead: the pose an entry needs
ENTRY_HEIGHT = (2.5, 3.5)       # metres above the water at which the entry task starts
LEVEL_UP = .01                  # difficulty step after a well-mastered block of attempts
LEVEL_DOWN = .005               # difficulty step back while the task is being lost
MASTERED = .65                  # readiness above which the task widens
LOST = .2                       # readiness below which the task narrows
BLOCK = 64                      # attempts before difficulty may move
EXTRA_OBSERVATIONS = 10  # five task indicators, hip/knee goals, three-axis up goal
HORIZONS = np.array([0, 1.4, 1., 1., 3.])
# The takeoff task asks for a small hop first and the competition takeoff at
# full mastery; the level widens on measured success like the other tasks.
TAKEOFF_RISE = (.1, .3)        # metres of rise at level 0 and level 1
TAKEOFF_SPEED = (1.2, 2.8)     # vertical departure speed in m/s at level 0 and level 1
BOARD_INVALID_COST = 2.       # charged in proportion to how illegal the takeoff is, not as a binary fault
FAST_RATE = .1                # success-rate estimates whose disagreement measures learning progress
SLOW_RATE = .02
PRACTICE_FLOOR = .15          # every practised skill keeps a share, so none can be starved
PROGRESS_SCALE = 4.
MINIMUM_PRACTICE = .2
# The engine invalidates a takeoff that leaves past horizontal as a whole; the
# practice cost also charges the lean itself so that leaving more upright is
# always worth something before the takeoff becomes valid.
TAKEOFF_LEAN_WEIGHT = .5
TAKEOFF_LEAN_SCALE = np.pi / 2


def takeoff_targets(level):
    level = np.clip(level, 0., 1.)
    return (TAKEOFF_RISE[0] + (TAKEOFF_RISE[1] - TAKEOFF_RISE[0]) * level,
            TAKEOFF_SPEED[0] + (TAKEOFF_SPEED[1] - TAKEOFF_SPEED[0]) * level)


def takeoff_cost(rise, speed, illegality, level, lean=0.):
    """Shortfall against the rung's rise and speed targets, the departure lean and how illegal the takeoff was."""
    rise_target, speed_target = takeoff_targets(level)
    return (np.maximum(0, 1 - np.asarray(rise, dtype=float) / rise_target)
            + .5 * np.maximum(0, 1 - np.asarray(speed, dtype=float) / speed_target)
            + TAKEOFF_LEAN_WEIGHT * np.clip(np.abs(np.asarray(lean, dtype=float)) / TAKEOFF_LEAN_SCALE, 0, 2)
            + BOARD_INVALID_COST * np.asarray(illegality, dtype=float))


def takeoff_lean(e):
    """Radians from vertical: the body's pitch until it leaves the board, then the pitch it left with."""
    return np.where(e.released, np.abs(e.departure_pitch), np.abs(e.prev_pitch))


class MotorCurriculum:
    def __init__(self, base, enabled=True, direction_practice=False, goal_practice=False):
        self.base = base
        self.direction_practice = bool(direction_practice)
        self.base.direction_practice = self.direction_practice
        self.goal_practice_enabled = bool(goal_practice)
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
        # Two success-rate estimates at different rates; their disagreement is
        # how fast the task is still changing, which is what practice follows.
        self.fast = np.zeros(5)
        self.slow = np.zeros(5)
        self.level = np.zeros(5)
        self.error_sum = np.zeros(5)
        self.direction_visits = np.zeros(4, int)
        self.direction_successes = np.zeros(4, int)
        self.direction_assignments = np.zeros(4, int)
        self.direction_readiness = np.zeros(4)
        self.assigned_goal = np.zeros(self.n, bool)
        self.goal_visits = np.zeros(len(DIVES), int)
        self.goal_mastery = np.zeros(len(DIVES))
        self.goal_progress = np.zeros(len(DIVES))
        self.last_completed = []
        self.assign(np.arange(self.n))

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def observation_size(self):
        return self.base.observation_size + EXTRA_OBSERVATIONS

    @property
    def practice(self):
        return self.base.practice | (self.task != 0) | self.assigned_goal

    def assign(self, ids):
        for i in ids:
            self.task[i] = 0
            self.assigned_goal[i] = False
            if self.enabled:
                # Full-from-board rollouts remain at least 25% from the start.
                initial_share = .5 if self.goal_practice_enabled else .75
                skills = np.array([1, 2, 4] if self.goal_practice_enabled else [1, 2, 3, 4])
                # Practise where the success rate is still moving. Allocating by
                # failure instead let a task that could not be won at all consume
                # half of every rollout for tens of millions of steps.
                progress = self.learning_progress()[skills]
                share = max(MINIMUM_PRACTICE, min(initial_share, initial_share * PROGRESS_SCALE * float(progress.max())))
                if self.base.rng.random() < share:
                    # Stationary aerial stabilization conflicts with full rotating dives.
                    # Keep its slot for old checkpoints and standalone skill evals.
                    weights = PRACTICE_FLOOR + PROGRESS_SCALE * progress
                    self.task[i] = self.base.rng.choice(skills, p=weights / weights.sum())
                elif self.goal_practice_enabled:
                    # At least 20% autonomous routines remain. Full-target
                    # practice gets 30–48%, motor fundamentals get 20–50%.
                    self.assigned_goal[i] = self.base.rng.random() < .6
            self.age[i] = self.task_returns[i] = self.previous_cost[i] = 0
            self.shape_goal[i] = [(0, 0), (1.5, 0), (1.4, 2)][self.base.rng.integers(3)] if self.task[i] == 2 else (0, 0)
            self.up_goal[i] = [0, 0, -1 if self.task[i] == 4 else 1]
            if self.task[i] == 3:
                theta = self.base.rng.uniform(-np.pi, np.pi)
                self.up_goal[i] = [np.sin(theta), 0, np.cos(theta)]
            if self.direction_practice and self.task[i] == 1:
                # Balance facing/direction combinations within the current mastery
                # scope; the actor still chooses its own legal declaration. No
                # maneuver or control path is given.
                directions = self.practised_directions()
                fewest = self.direction_assignments[directions].min()
                candidates = [d for d in directions if self.direction_assignments[d] == fewest]
                direction = int(self.base.rng.choice(candidates))
                self.direction_assignments[direction] += 1
                self.base.group[i] = direction + 1
                self.base.schedule[i, self.base.round[i]] = direction + 1

    def learning_progress(self):
        """How fast each task's success rate is still changing, mastered or not."""
        return np.abs(self.fast - self.slow)

    def practised_directions(self):
        """Takeoff directions (groups 1 to 4, zero-based) inside the base arena's current scope."""
        scope = getattr(self.base, 'scope', None)
        return [d for d in range(4) if scope is None or d + 1 in scope['groups']]

    def choose_practice_goal(self, i):
        legal = np.flatnonzero(self.base.mask()[i])
        mastery, visits = self.goal_mastery[legal], self.goal_visits[legal]
        complexity = np.array([DIVES[j]['turns'] + .5 * DIVES[j]['twists'] for j in legal])
        learned = complexity[mastery > .5]
        frontier = (float(learned.max()) if len(learned) else float(complexity.min())) + .5
        # 20% uniform coverage ensures no legal target can be permanently
        # avoided. Most practice stays near learned skills and recent progress.
        weights = (1 / np.sqrt(1 + visits) + 4 * np.maximum(self.goal_progress[legal], 0) + .05)
        weights *= np.exp(-np.maximum(0, complexity - frontier))
        weights = .2 / len(legal) + .8 * weights / weights.sum()
        return int(self.base.rng.choice(legal, p=weights))

    def horizons(self):
        horizons = HORIZONS[self.task].copy()
        if self.direction_practice:
            # Gradually connect board takeoff to complete flight/entry. Never
            # inject root motion or change the control policy between phases.
            horizons[self.task == 1] = 1.4 + 3.4 * self.level[1]
        return horizons

    def observe(self):
        obs = self.base.observe()
        # The autoregressive action history must remain the final nine columns.
        goals = np.column_stack([np.eye(5)[self.task], self.shape_goal / 3, self.up_goal])
        goals[self.task == 0, 5:] = 0  # Full dives use only their own declared intent.
        return np.column_stack([obs[:, :-HISTORY], goals, obs[:, -HISTORY:]]).astype(np.float32)

    def initialize_task(self, i):
        task = self.task[i]
        if self.assigned_goal[i]:
            self.base.practice[i] = True
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
        d.qpos[1:4] = [3, 0, (-e.height[i] + rng.uniform(*ENTRY_HEIGHT)) if task == 4 else 4]
        d.qvel[:] = 0
        if task == 4:
            d.qvel[3] = -rng.uniform(.5, 1.5 + 2 * level)
        if task in (3, 4):
            d.qvel[4:7] = rng.uniform(-.1 - 2 * level, .1 + 2 * level, 3)
        target = np.array([rng.uniform(0, .7), rng.uniform(0, .9), 1.2,
                           rng.uniform(0, 3.05), rng.uniform(0, 3.05), -.2, .2,
                           rng.uniform(0, 1.), .04])
        if task == 2:
            target[:2] = rng.uniform(0, [1.7, 2.2])
        if task == 4:
            # The entry task starts near the pose an entry needs and widens with
            # mastery. Started from a random pose it never succeeded once, so its
            # difficulty could never move and the skill that decides every dive
            # was never practised.
            target = ENTRY_POSE + rng.uniform(-1, 1, 9) * (.15 + .6 * level) * np.array([1, 1, .5, 1, 1, .5, .5, 1, .1])
            target = np.clip(target, [-.45, 0, -.55, -.5, -.5, -1.1, -1.1, 0, -.15], [2.2, 2.6, 1.35, 3.14, 3.14, 1.1, 1.1, 2.3, .18])
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
        lean = takeoff_lean(e)
        takeoff = takeoff_cost(rise, e.takeoff_vertical_speed, live_takeoff_legality(e, np.degrees(lean)), self.level[1], lean)
        # Armstand practice rewards a genuine clear release, not an impossible
        # standing-jump target from a hand-supported starting pose.
        takeoff = np.where(e.armstand, (~e.released).astype(float) + 2 * e.board_invalid, takeoff)
        if self.direction_practice:
            momentum = np.where(e.released, e.takeoff_angular_momentum[:, 1], e.sensors[:, 7])
            direction_cost = rotation_direction_cost(momentum, e.goals[:, 0], e.released)
            # A correct contact impulse alone is insufficient if the body then
            # somersaults the wrong way. Small counter-motions have a dead band.
            wrong_way = np.clip(-np.sign(e.goals[:, 0]) * e.phase_theta / np.pi - .04, 0, 2)
            takeoff += .8 * direction_cost + wrong_way
            preparing = entry_weight(e.above_water, e.sensors[:, 5]) * e.released
            desired_position = 1 - live_position_quality(e)
            takeoff += self.level[1] * (desired_position * (1-preparing) + entry * preparing)
        return np.choose(self.task, [np.zeros(self.n), takeoff, shape, aerial, entry]).clip(0, 20)

    def step(self, choices, actions, noise=None):
        self.last_completed = []
        chosen = self.base.choosing.copy()
        tasks = self.task.copy()
        goals = self.assigned_goal.copy()
        horizons = self.horizons()
        groups = self.base.group.copy()
        previous = self.costs()
        choices = np.array(choices, copy=True)
        for i in np.flatnonzero(chosen & goals):
            choices[i] = self.choose_practice_goal(i)
        self.base.recovery_allowed[:] = (tasks == 0) & ~goals
        _, reward, done, rows = self.base.step(choices, actions, noise)
        finished = {row['index']: row for row in rows}
        for row in rows:
            i = row['index']
            if goals[i]:
                target = IDS[row['declaration']]
                outcome = float(row['valid']) * np.clip(row['execution'] / 10, 0, 1)
                if self.base.reward_mode in ('completion-first', 'conjunctive'):
                    # Advance the target frontier on completion, not only once
                    # a newly learned dive also receives near-perfect execution.
                    outcome = float(row.get('completedDeclaration', row['valid'])) * (.85 + .15 * np.clip(row['execution'] / 10, 0, 1))
                delta = .05 * (outcome - self.goal_mastery[target])
                self.goal_mastery[target] += delta
                self.goal_progress[target] += .1 * (delta - self.goal_progress[target])
                self.goal_visits[target] += 1
                row['assignedPracticeGoal'] = True
        for i in np.flatnonzero(chosen):
            self.initialize_task(i)
        cost = self.costs()
        self.age += .02 * ~chosen
        skill = tasks != 0
        # Fixed practice duration removes an incentive to prolong a good pose or
        # terminate early to evade accumulated error.
        timed = skill & ~chosen & (self.age >= horizons - 1e-8)
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
                    cost[i] = (float(takeoff_cost(m['ascent'], m['takeoffVerticalSpeed'], takeoff_legality(m), self.level[1],
                                                  np.radians(m['departureLean'])))
                               if finished[i]['category'] != 6 else 2 * m['boardInvalid'] + float(not m['water']))
                    if self.direction_practice:
                        intent = DIVES[IDS[finished[i]['declaration']]]
                        cost[i] += .8 * rotation_direction_cost(m['takeoffAngularMomentum'][1], intent['sign'] * intent['turns'])
                        cost[i] += np.clip(-intent['sign'] * m['rotation'] * 2 - .04, 0, 2)
                        cost[i] += self.level[1] * (finished[i]['entryAngle'] / 45 + (not m['fullEntryComplete'])
                                                   + sum(np.log1p(v) for v in m['entryFaultLosses'].values()))
                else:
                    cost[i] = 20.  # An early crash cannot improve a motor task.
            # Direct task error plus a progress term. Terminal cancellation keeps
            # the progress term from adding a second final-pose reward.
            reward[i] = -.02 * cost[i] + previous[i] - (.995 * cost[i] if not ended[i] else 0)
            if ended[i]:
                reward[i] -= 2 * cost[i]
                if done[i] and tasks[i] in (2, 3):
                    reward[i] -= 20 * max(0, horizons[i] - self.age[i])
                threshold = [0, .7, .2, .25, .8][tasks[i]]
                success = cost[i] < threshold
                task = tasks[i]
                self.last_completed.append(dict(index=int(i), task=int(task), error=float(cost[i]), success=bool(success)))
                self.visits[task] += 1
                self.successes[task] += success
                self.readiness[task] += .03 * (success - self.readiness[task])
                self.fast[task] += FAST_RATE * (success - self.fast[task])
                self.slow[task] += SLOW_RATE * (success - self.slow[task])
                self.error_sum[task] += cost[i]
                if self.direction_practice and task == 1 and 1 <= groups[i] <= 4:
                    self.direction_visits[groups[i]-1] += 1
                    self.direction_successes[groups[i]-1] += success
                    g = groups[i]-1
                    self.direction_readiness[g] += .03 * (success - self.direction_readiness[g])
                directions = self.practised_directions()
                direction_ready = (not self.direction_practice or task != 1
                                   or (self.direction_visits[directions].min() >= 16 and self.direction_readiness[directions].min() > MASTERED))
                if self.visits[task] >= BLOCK and self.readiness[task] > MASTERED and direction_ready:
                    self.level[task] = min(1., self.level[task] + LEVEL_UP)
                elif self.visits[task] >= BLOCK and self.readiness[task] < LOST:
                    # A task that is being lost narrows again instead of staying unwinnable.
                    self.level[task] = max(0., self.level[task] - LEVEL_DOWN)
        self.task_returns += reward * skill
        # A practice declaration, score or used-dive mask must never become the
        # starting history of a subsequent autonomous competition routine.
        clean_history = ended | (goals & done)
        if clean_history.any():
            self.base.reset(np.flatnonzero(clean_history), new_routine=True)
        done |= ended
        if done.any():
            self.assign(np.flatnonzero(done))
        # Motor practice cannot inflate competition metrics or choose champions.
        rows = [row for row in rows if tasks[row['index']] == 0]
        return self.observe(), reward, done, rows

    def metrics(self):
        result = {TASKS[t]: dict(episodes=int(self.visits[t]), successes=int(self.successes[t]),
                              readiness=float(self.readiness[t]), level=float(self.level[t]),
                              learningProgress=float(self.learning_progress()[t]),
                              meanError=float(self.error_sum[t] / max(1, self.visits[t]))) for t in range(1, 5)}
        if self.direction_practice:
            result['takeoff']['byDirection'] = {
                str(g+1): dict(episodes=int(self.direction_visits[g]), successes=int(self.direction_successes[g]),
                               readiness=float(self.direction_readiness[g]))
                for g in range(4)}
        if self.goal_practice_enabled:
            scope = np.array([in_practice_scope(d) for d in DIVES])
            result['specificDives'] = dict(eligibleTargets=int(scope.sum()), attempts=int(self.goal_visits[scope].sum()),
                                           practicedTargets=int(((self.goal_visits > 0) & scope).sum()),
                                           masteredTargets=int(((self.goal_mastery > .5) & scope).sum()))
        return result

    def state_dict(self):
        arrays = {k: v.copy() for k, v in vars(self).items() if isinstance(v, np.ndarray)}
        return dict(base=self.base.state_dict(), curriculum=arrays, enabled=self.enabled,
                    directionPractice=self.direction_practice, goalPractice=self.goal_practice_enabled)

    def load_state_dict(self, state):
        if (state['enabled'] != self.enabled or state.get('directionPractice', False) != self.direction_practice
                or state.get('goalPractice', False) != self.goal_practice_enabled):
            raise ValueError('Cannot resume with changed motor curriculum')
        self.base.load_state_dict(state['base'])
        for k, v in state['curriculum'].items():
            getattr(self, k)[:] = copy.deepcopy(v)


def live_position_quality(e):
    from positions import position_qualities
    from geometry import tuck_geometry
    q = e.state[:, 1 + e.qadr]
    qualities = position_qualities(q[:, 0], q[:, 1], np.max(tuck_geometry(e.sensors)[0], axis=1), q[:, 3], q[:, 4])
    # The physics goal order is C, B, D, A; qualities use A, B, C, D.
    indices = np.array([2, 1, 3, 0])[e.goals[:, 3].astype(int)]
    return qualities[np.arange(e.n), indices]
