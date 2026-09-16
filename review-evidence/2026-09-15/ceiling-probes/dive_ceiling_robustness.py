"""How robust is the best open-loop 10 m dive to the training stance randomization?
If it survives, timing is forgiving and feedback is a refinement; if it collapses, closed-loop control is essential."""
import sys, json, numpy as np
sys.path.insert(0, "/Volumes/Zach's SSD/PortfolioTraining/2026-09-14-bounded-fresh/source/LearnToDive/training_v3")
from engine import Arena as Physics
from geometry import encoded_action
from rules import IDS
from judge import judge
N = 64
STRAIGHT_ENTRY = np.array([0, 0, 1.2, 3.14, 3.14, -.3, .3, 0, .06])


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
params = np.array(json.loads(sys.argv[1])); height = float(sys.argv[2]); spread = float(sys.argv[3])
rng = np.random.default_rng(11)
contexts = [dict(skill=0, height=height, preload=0, disturbance=0, platform=True,
                 lean=.04 + spread * rng.uniform(-.02, .04), hip=.12 + spread * rng.uniform(-.04, .08),
                 knee=.2 + spread * rng.uniform(-.05, .1), toeOver=.03 + spread * rng.uniform(-.03, .03)) for _ in range(N)]
p = Physics(N, seed=5, threads=4, skills=[0], heights=(1, 10), minimum_heights=[1] * 6, training=False)
p.reset(list(range(N)), contexts); p.platform[:] = True; p.headfirst[:] = True; p.goals[:] = [.5, 0, 0, 3]
finished = [None] * N; t_release = np.full(N, np.nan)
for _ in range(300):
    actions = np.zeros((N, 9))
    for i in range(N):
        if finished[i] is not None:
            actions[i] = encoded_action(STRAIGHT_ENTRY); continue
        if p.released[i] and np.isnan(t_release[i]): t_release[i] = p.state[i, 0]
        target = schedule(params, p.state[i, 0], bool(p.released[i]), t_release[i])
        actions[i] = encoded_action(p.targets[i] if target is None else target)
    _, _, done, infos = p.step(actions, auto_reset=False)
    for m in infos:
        if finished[m['index']] is None: finished[m['index']] = m
    if all(f is not None for f in finished): break
p.close()
rows = []
for m in finished:
    m = dict(m); m['positionQuality'] = m['positionQualities'][0]
    s = judge(IDS['101A'], 'platform', height, m)
    rows.append(dict(valid=s['valid'], execution=s['execution'], angle=s['entryAngle'], clean=s['clean'], failures=s['failures'], rotation=m['rotation']))
angles = np.array([r['angle'] for r in rows]); ex = np.array([r['execution'] for r in rows])
import collections
print(json.dumps(dict(height=height, spread=spread, valid=int(sum(r['valid'] for r in rows)), clean=int(sum(r['clean'] for r in rows)), n=N,
                      executionMean=float(ex.mean()), executionMax=float(ex.max()), angleP10=float(np.percentile(angles, 10)), angleMedian=float(np.median(angles)), angleP90=float(np.percentile(angles, 90)),
                      failures=collections.Counter(f for r in rows for f in r['failures']).most_common(4))))
