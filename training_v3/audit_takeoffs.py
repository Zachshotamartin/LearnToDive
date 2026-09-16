"""Bounded mechanical search over every legal apparatus/height/category start.

Probe commands are diagnostic witnesses only, never imported by a trainer.
Failure to find a witness is explicitly inconclusive, not proof of impossibility.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from environment import Arena
from geometry import encoded_action, ACTION_LOW, ACTION_HIGH
from rules import legal_mask


def probe(apparatus, height, group, trials=64):
    env = Arena(trials, seed=2231, threads=1, training=False)
    rng = np.random.default_rng(211)
    try:
        env.apparatus[:] = apparatus == 'platform'
        env.height[:] = height
        env.group[:] = group
        env.used[:] = False
        declaration = np.flatnonzero(legal_mask(group, apparatus, height))[0]
        for i in range(trials):
            env.declare(i, int(declaration))
        e = env.physics
        hold = e.targets.copy()
        armstand = bool(e.armstand[0])
        crouch = np.tile([.67, 1.73, -.55, .01, .01, 0, 0, 0, 0.], (trials, 1))
        extend = np.tile([.32, 0., -.13, 3.1, 3.1, -.3, .3, 0, .06], (trials, 1))
        if armstand:
            crouch[:] = [0, 0, 1.2, 3.14, 3.14, 0, 0, .7, .06]
            extend[:] = [0, 0, 1.2, 3.14, 3.14, 0, 0, 0, .06]
        crouch += rng.normal(0, .25, crouch.shape)
        extend += rng.normal(0, .25, extend.shape)
        # Include the established forward witness without random perturbation.
        if not armstand:
            crouch[0] = [.67, 1.73, -.55, .01, .01, 0, 0, 0, 0.]
            extend[0] = [.32, 0., -.13, 3.1, 3.1, -.3, .3, 0, .06]
        duration = rng.uniform(.2, .6, trials)
        duration[0] = .42
        qualified = np.zeros(trials, bool)
        best = dict(rise=0., departureSpeed=0.)
        for tick in range(100):
            time = tick * .02
            target = np.where((time < .1 + duration)[:, None], crouch, extend).copy()
            if time < .1:
                target = hold
            elif not armstand:
                target[time >= .35 + duration, 2] = 1.3
            target = np.clip(target, ACTION_LOW, ACTION_HIGH)
            e.step(encoded_action(target), auto_reset=False)
            rise = np.maximum(0, e.apex_com - e.departure_com)
            good = e.released & ~e.board_invalid & (e.takeoff_vertical_speed > .5) & (rise > .1)
            qualified |= good
            best['rise'] = max(best['rise'], float(rise.max()))
            best['departureSpeed'] = max(best['departureSpeed'], float(e.takeoff_vertical_speed.max()))
        return dict(apparatus=apparatus, height=height, group=group, candidates=trials,
                    demonstrated=bool(qualified.any()), witnessIndices=np.flatnonzero(qualified).tolist(),
                    observedMaxima=best, interpretation='Takeoff only; does not establish a valid completed dive')
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--trials', type=int, default=64)
    args = p.parse_args()
    rows = []
    for apparatus, heights, groups in [('platform', [5, 7.5, 10], range(1, 7)),
                                     ('springboard', [1, 3], range(1, 6))]:
        for height in heights:
            for group in groups:
                row = probe(apparatus, height, group, args.trials)
                rows.append(row)
                print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps(dict(trainingUsesProbes=False, cases=rows), indent=2))
