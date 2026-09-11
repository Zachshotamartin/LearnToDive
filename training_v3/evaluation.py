"""Held-out evaluation of a policy on full routines. Nothing here publishes assets."""
import numpy as np
import torch

from environment import Arena

EVALUATION_SEED = 771100
ROUTINE_LENGTH = 6
MAX_TICKS = 1200


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else None


def group(rows):
    return dict(n=len(rows), points=mean(rows, 'points'), execution=mean(rows, 'execution'),
                difficulty=mean(rows, 'difficulty'), clean=mean(rows, 'clean'), valid=mean(rows, 'valid'),
                uniqueDives=len({r['declaration'] for r in rows}), entryAngle=mean(rows, 'entryAngle'),
                trainingReturn=mean(rows, 'return'))


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
