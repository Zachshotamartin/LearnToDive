"""Physics ceiling probe: can ANY open-loop forward dive (101A) score a clean entry with this body and judge?

Random search over a five-phase servo-target schedule, judged with the live judge.
This is a test of the environment, never a training target.
"""
import sys, json, collections
import numpy as np
SRC = "/Volumes/Zach's SSD/PortfolioTraining/2026-09-14-bounded-fresh/source/LearnToDive/training_v3"
sys.path.insert(0, SRC)
from engine import Arena as Physics
from geometry import encoded_action
from rules import IDS, DIVES
from judge import judge
N = 64
STRAIGHT_ENTRY = np.array([0, 0, 1.2, 3.14, 3.14, -.3, .3, 0, .06])
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)

def sample(center=None, spread=1.0):
    lo = np.array([.3, -.3, .8, -.55, -.5, 0., 0., -.3, 0., -.5, 1.5, .05, .2, 0., 2.5])
    hi = np.array([.7, 1.2, 2.2, .3, .5, 1., .5, .3, .3, 0., 3.14, .3, 1.2, 1.5, 3.14])
    if center is None:
        return rng.uniform(lo, hi)
    return np.clip(center + spread * .15 * (hi - lo) * rng.normal(size=len(lo)), lo, hi)

def schedule(p, t, released, t_release):
    t1, hip_c, knee_c, ankle_c, sh_c, elbow_c, hip_e, ankle_e, knee_e, sh_back, sh_e, delta, t_open, hip_f, sh_f = p
    if t < .1:
        return None
    if t < t1:
        return np.array([hip_c, knee_c, ankle_c, sh_c, sh_c, 0, 0, elbow_c, 0.])
    if not released:
        ankle = ankle_e if t < t1 + delta else 1.3
        return np.array([hip_e, knee_e, ankle, sh_e, sh_e, -.3, .3, 0, .06])
    if t - t_release < t_open:
        return np.array([hip_f, 0., 1.2, sh_f, sh_f, -.3, .3, 0, .06])
    return STRAIGHT_ENTRY

def run_batch(params, height):
    p = Physics(N, seed=7, threads=4, skills=[0], heights=(1, 10), minimum_heights=[1] * 6, training=False)
    p.reset(list(range(N)), [dict(skill=0, height=height, preload=0, disturbance=0, platform=True, lean=.04, hip=.12, knee=.2, toeOver=.03)] * N)
    p.platform[:] = True; p.headfirst[:] = True; p.goals[:] = [.5, 0, 0, 3]
    finished = [None] * N; t_release = np.full(N, np.nan)
    for _ in range(300):
        actions = np.zeros((N, 9))
        for i in range(N):
            if finished[i] is not None:
                actions[i] = encoded_action(STRAIGHT_ENTRY); continue
            if p.released[i] and np.isnan(t_release[i]):
                t_release[i] = p.state[i, 0]
            target = schedule(params[i], p.state[i, 0], bool(p.released[i]), t_release[i])
            actions[i] = encoded_action(p.targets[i] if target is None else target)
        _, _, done, infos = p.step(actions, auto_reset=False)
        for m in infos:
            if finished[m['index']] is None:
                finished[m['index']] = m
        if all(f is not None for f in finished):
            break
    p.close()
    rows = []
    for i, m in enumerate(finished):
        if m is None:
            rows.append(dict(execution=-1, entryAngle=180, valid=False, failures=['timeout'], rotation=0, ascent=0, clean=False)); continue
        m = dict(m); m['positionQuality'] = m['positionQualities'][0]
        try:
            s = judge(IDS['101A'], 'platform', height, m)
            rows.append(dict(execution=s['execution'], entryAngle=s['entryAngle'], valid=s['valid'], clean=s['clean'], failures=s['failures'], deductions=s['deductions'], rotation=m['rotation'], ascent=m['ascent'], form=m['form'], firstContactAngle=m['firstContactAngle']))
        except Exception as error:
            rows.append(dict(execution=-1, entryAngle=m.get('entryAngle', 180), valid=False, clean=False, failures=[repr(error)[:80]], rotation=m.get('rotation', 0), ascent=m.get('ascent', 0)))
    return rows

def objective(r):
    return r['execution'] - (0 if r['valid'] else 3) - r['entryAngle'] / 90

height = float(sys.argv[1]) if len(sys.argv) > 1 else 10.
best = None; best_row = None; history = []
for batch in range(int(sys.argv[3]) if len(sys.argv) > 3 else 24):
    params = [sample() if best is None or batch < 8 or k < 16 else sample(best, spread=1.0 if k < 40 else .4) for k in range(N)]
    rows = run_batch(params, height)
    for pr, r in zip(params, rows):
        if best_row is None or objective(r) > objective(best_row):
            best, best_row = pr, r
    valid = sum(r['valid'] for r in rows); angles = sorted(r['entryAngle'] for r in rows)
    history.append(dict(batch=batch, valid=valid, bestExecution=best_row['execution'], bestAngle=best_row['entryAngle'], medianAngle=angles[len(angles)//2], minAngle=angles[0]))
    print(json.dumps(history[-1]), flush=True)
print('BEST', json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_row.items()}, default=float))
print('PARAMS', np.round(best, 3).tolist())
print('failure mix over last batch', collections.Counter(f for r in rows for f in r['failures']).most_common(6))
