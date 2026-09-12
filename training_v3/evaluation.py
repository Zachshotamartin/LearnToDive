"""Held-out evaluation of a policy on full routines. Nothing here publishes assets."""
import numpy as np
import torch

import math

from engine import DT, ENTRY_WINDOW, TIMEOUT
from environment import Arena

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
    return dict(n=len(rows), points=mean(rows, 'points'), execution=mean(rows, 'execution'),
                difficulty=mean(rows, 'difficulty'), clean=mean(rows, 'clean'), valid=mean(rows, 'valid'),
                uniqueDives=len({r['declaration'] for r in rows}), entryAngle=mean(rows, 'entryAngle'),
                trainingReturn=mean(rows, 'return'),
                # Takeoff and position diagnostics: whether it jumps, how high, how upright, how well shaped.
                jumped=float(np.mean([r['measurements']['takeoffVerticalSpeed'] > JUMP_SPEED for r in rows])) if rows else None,
                rise=measured(rows, 'ascent'), takeoffSpeed=measured(rows, 'takeoffVerticalSpeed'),
                departureLean=measured(rows, 'departureLean'), positionQuality=measured(rows, 'positionQuality'))


def summary(records):
    """Aggregate judged dives overall and per group, excluding recovery practice."""
    full = [r for r in records if not r['practice']]
    return dict(full=group(full), categories={str(g): group([r for r in full if r['category'] == g]) for g in range(1, 7)},
                practiceEpisodes=len(records) - len(full))


@torch.no_grad()
def evaluate(policy, seed=EVALUATION_SEED, cases=48, perturb=False):
    """Deterministic full routines on held-out worlds; the recovery bank is always disabled."""
    e = Arena(cases, seed, threads=4, training=False)
    result = []
    counts = np.zeros(cases, int)
    try:
        e.perturb = perturb
        for _ in range(MAX_TICKS):
            out = policy(torch.tensor(e.observe()), torch.tensor(e.mask()), torch.tensor(e.choosing), deterministic=True)
            _, _, _, rows = e.step(out['choice'].numpy(), out['action'].numpy())
            for row in rows:
                i = row['index']
                if counts[i] < ROUTINE_LENGTH:
                    result.append(row)
                    counts[i] += 1
            if np.all(counts == ROUTINE_LENGTH):
                break
        if len(result) < cases * ROUTINE_LENGTH:
            raise RuntimeError('Evaluation failed to complete the declared episode count')
        return dict(seed=seed, perturbed=perturb, summary=summary(result), episodes=result)
    finally:
        e.close()
