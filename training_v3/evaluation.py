"""Held-out evaluation of a policy on full routines. Nothing here publishes assets."""
import numpy as np
import torch

import math

from engine import DT, ENTRY_WINDOW, TIMEOUT
from geometry import ANGULAR_VELOCITY
from judge import ROTATION_TOLERANCE, TWIST_TOLERANCE
from environment import Arena
from rules import DIVES, IDS

EVALUATION_SEED = 771100
ROUTINE_LENGTH = 6
# One dive lasts at most the no-water timeout plus the entry window, plus a
# declaration tick and a reset tick. A fixed 1200-tick budget was enough only
# while every dive was a short fall; jumping athletes overran it and the
# evaluation raised mid-run.
DIVE_TICKS = math.ceil((TIMEOUT + ENTRY_WINDOW) / DT) + 2
MAX_TICKS = math.ceil(1.1 * ROUTINE_LENGTH * DIVE_TICKS)


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else None


JUMP_SPEED = .5   # m/s of upward departure speed that counts as a jump rather than a fall


def measured(rows, key):
    return float(np.mean([r['measurements'][key] for r in rows])) if rows else None


def group(rows):
    signed = [DIVES[IDS[r['declaration']]]['sign'] * r['measurements']['rotation'] for r in rows]
    momentum = [DIVES[IDS[r['declaration']]]['sign'] * r['measurements']['takeoffAngularMomentum'][1]
                for r in rows if 'takeoffAngularMomentum' in r['measurements']]
    return dict(n=len(rows), points=mean(rows, 'points'), execution=mean(rows, 'execution'),
                signedRotationTurns=float(np.mean(signed)) if signed else None,
                wrongDirection=float(np.mean(np.asarray(signed) < -.02)) if signed else None,
                correctDirection=float(np.mean(np.asarray(signed) > .02)) if signed else None,
                signedTakeoffMomentum=float(np.mean(momentum)) if momentum else None,
                correctTakeoffDirection=float(np.mean(np.asarray(momentum) > 2.)) if momentum else None,
                difficulty=mean(rows, 'difficulty'), clean=mean(rows, 'clean'), valid=mean(rows, 'valid'),
                uniqueDives=len({r['declaration'] for r in rows}), entryAngle=mean(rows, 'entryAngle'),
                trainingReturn=mean(rows, 'return'),
                # Takeoff and position diagnostics: whether it jumps, how high, how upright, how well shaped.
                jumped=float(np.mean([r['measurements']['takeoffVerticalSpeed'] > JUMP_SPEED for r in rows])) if rows else None,
                rise=measured(rows, 'ascent'), takeoffSpeed=measured(rows, 'takeoffVerticalSpeed'),
                departureLean=measured(rows, 'departureLean'), positionQuality=measured(rows, 'positionQuality'),
                motorPositionQuality=measured(rows, 'motorPositionQuality'),
                entryArms=measured(rows, 'entryArmPositionValid'),
                completedRotation=float(np.mean([r['rotationError'] <= ROTATION_TOLERANCE and r['twistError'] < TWIST_TOLERANCE for r in rows])) if rows else None,
                entryAngularSpeed=measured(rows, 'entryAngularSpeed'),
                postContactRotationDegrees=float(np.mean([r.get('postContactRotationDegrees', 0) for r in rows])) if rows else None,
                entryAngleDeterioration=float(np.mean([max(0, r['entryAngle'] - r['measurements']['firstContactAngle']) for r in rows])) if rows else None)


def summary(records):
    """Aggregate judged dives overall and per group, excluding recovery practice."""
    full = [r for r in records if not r['practice']]
    return dict(full=group(full), categories={str(g): group([r for r in full if r['category'] == g]) for g in range(1, 7)},
                practiceEpisodes=len(records) - len(full))


@torch.no_grad()
def evaluate(policy, seed=EVALUATION_SEED, cases=48, perturb=False, reward_mode="v12", deterministic=True, policy_seed=91731):
    """Deterministic full routines on held-out worlds; the recovery bank is always disabled."""
    if cases < 1:
        raise ValueError("Evaluation needs at least one world")
    rng = torch.get_rng_state()
    torch.manual_seed(policy_seed)
    e = None
    result = []
    counts = np.zeros(cases, int)
    rotation = np.zeros(cases)
    try:
        e = Arena(cases, seed, threads=1, training=False, reward_mode=reward_mode)
        if getattr(policy, 'obs', e.observation_size) == e.observation_size + 10:
            from motor_curriculum import MotorCurriculum
            e = MotorCurriculum(e, enabled=False)
        e.perturb = perturb
        if hasattr(e, 'base'):
            e.base.perturb = perturb
        for _ in range(MAX_TICKS):
            wet = np.isfinite(e.physics.entry_time)
            rotation += np.linalg.norm(e.physics.sensors[:, ANGULAR_VELOCITY], axis=1) * DT * wet
            out = policy(torch.tensor(e.observe()), torch.tensor(e.mask()), torch.tensor(e.choosing), deterministic=deterministic)
            _, _, _, rows = e.step(out['choice'].numpy(), out['action'].numpy())
            for row in rows:
                i = row['index']
                row['postContactRotationDegrees'] = float(np.degrees(rotation[i]))
                rotation[i] = 0
                if counts[i] < ROUTINE_LENGTH:
                    result.append(row)
                    counts[i] += 1
            if np.all(counts == ROUTINE_LENGTH):
                break
        if len(result) < cases * ROUTINE_LENGTH:
            raise RuntimeError('Evaluation failed to complete the declared episode count')
        return dict(evaluationVersion=3, seed=seed, cases=cases, deterministic=deterministic, policySeed=policy_seed, perturbed=perturb, rewardMode=reward_mode, summary=summary(result), episodes=result)
    finally:
        if e is not None:
            e.close()
        torch.set_rng_state(rng)


@torch.no_grad()
def evaluate_motor_skills(policy, seed=782100, cases=24, level=0., direction_practice=False):
    """Reproducible motor-only diagnostics, explicitly excluded from dive scores."""
    from motor_curriculum import MotorCurriculum, TASKS, HORIZONS
    if cases < 1 or not 0 <= level <= 1:
        raise ValueError('Invalid motor evaluation cohort')
    torch_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    e = None
    records = []
    try:
        base = Arena(cases * 4, seed, threads=1, training=False, practice=0)
        e = MotorCurriculum(base, enabled=False, direction_practice=direction_practice)
        e.task[:] = np.repeat(np.arange(1, 5), cases)
        e.level[:] = level
        goals = np.random.default_rng(seed + 1)
        e.shape_goal[:] = np.array([(0, 0), (1.5, 0), (1.4, 2)])[goals.integers(3, size=e.n)]
        theta = goals.uniform(-np.pi, np.pi, e.n)
        e.up_goal[:] = np.column_stack([np.sin(theta), np.zeros(e.n), np.cos(theta)])
        e.up_goal[e.task == 4] = [0, 0, -1]
        if direction_practice:
            ids = np.flatnonzero(e.task == 1)
            e.group[ids] = np.arange(len(ids)) % 4 + 1
        for _ in range(int(e.horizons().max() / DT) + 3):
            obs = e.observe()
            if policy.obs == base.observation_size:
                # A baseline without task context still receives the same physical
                # state. Its result measures existing skills, not goal-conditioned IQ.
                obs = np.column_stack([obs[:, :base.observation_size - 9], obs[:, -9:]])
            out = policy(torch.tensor(obs), torch.tensor(e.mask()), torch.tensor(e.choosing), deterministic=True)
            e.step(out['choice'].numpy(), out['action'].numpy())
            records.extend(e.last_completed)
            if len(records) == cases * 4:
                break
        if len(records) != cases * 4:
            raise RuntimeError('Incomplete motor evaluation')
        return dict(seed=seed, casesPerTask=cases, level=level, practiceOnly=True, directionPractice=direction_practice,
                    tasks={TASKS[t]: dict(n=cases, meanError=float(np.mean([r['error'] for r in records if r['task'] == t])),
                                         success=float(np.mean([r['success'] for r in records if r['task'] == t])))
                           for t in range(1, 5)}, episodes=records)
    finally:
        if e is not None:
            e.close()
        torch.set_rng_state(torch_rng)


@torch.no_grad()
def evaluate_targets(policy, seed=794100, cases_per_target=4, reward_mode='phase-dense'):
    """Explicit-target motor benchmark, separate from autonomous routine results.

    Every rollout starts on the board. This reports whether the requested dive
    was performed; its assigned declaration is never a selector success.
    """
    targets = ('101C', '103C', '201C', '203C', '301C', '303C', '401C', '403C', '5132D', '612C')
    assignments = np.repeat([IDS[code] for code in targets], cases_per_target)
    random_state = torch.get_rng_state()
    torch.manual_seed(seed)
    e = None
    records, completed = [], set()
    try:
        e = Arena(len(assignments), seed, threads=1, training=False, practice=0, reward_mode=reward_mode)
        if policy.obs == e.observation_size + 10:
            from motor_curriculum import MotorCurriculum
            e = MotorCurriculum(e, enabled=False)
        base = getattr(e, 'base', e)
        for i, target in enumerate(assignments):
            base.group[i], base.height[i], base.apparatus[i] = DIVES[target]['group'], 10, 1
            base.declare(i, int(target))
            # Paired deterministic disturbances test more than one identical reset.
            if i % cases_per_target:
                base.physics.disturbance[i] = (-1 if i % 2 else 1) * (10 + 5 * (i % cases_per_target))
                base.physics.disturbance_time[i] = .75
        for _ in range(DIVE_TICKS):
            out = policy(torch.tensor(e.observe()), torch.tensor(e.mask()), torch.tensor(e.choosing), deterministic=True)
            _, _, _, rows = e.step(out['choice'].numpy(), out['action'].numpy())
            for row in rows:
                if row['index'] not in completed:
                    row['assignedEvaluationGoal'] = True
                    records.append(row)
                    completed.add(row['index'])
            if len(completed) == len(assignments):
                break
        if len(completed) != len(assignments):
            raise RuntimeError('Incomplete target benchmark')
        return dict(seed=seed, casesPerTarget=cases_per_target, autonomousSelection=False, boardStartsOnly=True,
                    targets={code: group([r for r in records if r['declaration'] == code]) for code in targets}, episodes=records)
    finally:
        if e is not None:
            e.close()
        torch.set_rng_state(random_state)
