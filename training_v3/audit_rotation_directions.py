"""Search bounded joint controls to audit direction feasibility, never train an actor.

Every replay starts on the real board. No root velocity/force is injected.
These diagnostic controls are not demonstrations or training data.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from environment import Arena
from geometry import ACTION_LOW, ACTION_HIGH, encoded_action
from rules import IDS, DIVES


def search(code, population=48, generations=8, seed=1921):
    rng = np.random.default_rng(seed)
    e = Arena(population, seed, threads=2, training=False, practice=0)
    sign = DIVES[IDS[code]]['sign']
    knots = np.array([.08, .32, .64, 1.05])
    mean = np.tile([.5, .8, .65, 1.7, 1.7, 0, 0, .15, .04], (4, 1))
    std = np.tile((ACTION_HIGH - ACTION_LOW) * .32, (4, 1))
    best = None
    try:
        for generation in range(generations):
            controls = np.clip(rng.normal(mean, std, (population, 4, 9)), ACTION_LOW, ACTION_HIGH)
            controls[:, :, 4] = controls[:, :, 3]
            controls[:, :, 5:7] = 0
            controls[:, :, 8] = .04
            if best is not None:
                controls[0] = best['controls']
            e.apparatus[:] = 1
            e.height[:] = 10
            e.group[:] = int(code[0])
            e.used[:] = False
            for i in range(population):
                e.declare(i, IDS[code])
            hold = e.physics.targets.copy()
            release_momentum = np.zeros(population)
            seen = np.zeros(population, bool)
            for tick in range(80):
                t = tick * .02
                k = min(2, max(0, np.searchsorted(knots, t) - 1))
                f = np.clip((t - knots[k]) / (knots[k+1] - knots[k]), 0, 1)
                target = hold if t < knots[0] else controls[:, k] * (1-f) + controls[:, k+1] * f
                e.physics.step(encoded_action(target), auto_reset=False)
                newly = e.physics.released & ~seen
                release_momentum[newly] = e.physics.sensors[newly, 7]  # world Y angular momentum
                seen |= newly
            p = e.physics
            # Judge the final departure, matching rotation bookkeeping. Earlier
            # tiny contact losses may be followed by another legal board contact.
            release_momentum = p.takeoff_angular_momentum[:, 1].copy()
            rotation = p.phase_theta / (2 * np.pi)
            rise = np.maximum(0, p.apex_com - p.departure_com)
            score = (np.clip(sign * release_momentum / 20, -3, 2) + np.clip(sign * rotation * 4, -4, 2)
                     + np.clip(p.takeoff_vertical_speed / 2, -1, 1) + np.clip(rise / .3, 0, 1)
                     - 8 * p.board_invalid - 4 * ~seen - 3 * p.max_lateral)
            elite = np.argsort(score)[-max(4, population // 6):]
            winner = int(elite[-1])
            if best is None or float(score[winner]) > best['searchScore']:
                best = dict(code=code, sign=sign, searchScore=float(score[winner]), controls=controls[winner].tolist(),
                            released=bool(seen[winner]), boardInvalid=bool(p.board_invalid[winner]),
                            takeoffMomentumY=float(release_momentum[winner]), rotationTurns=float(rotation[winner]),
                            takeoffSpeed=float(p.takeoff_vertical_speed[winner]), rise=float(rise[winner]),
                            lateralTilt=float(p.max_lateral[winner]))
            mean = .25 * mean + .75 * controls[elite].mean(0)
            std = np.maximum(.06, .25 * std + .75 * controls[elite].std(0))
            print(json.dumps(dict(code=code, generation=generation, best={k:v for k,v in best.items() if k != 'controls'})), flush=True)
        best['directionFeasible'] = bool(best['released'] and not best['boardInvalid']
                                         and sign * best['takeoffMomentumY'] > 2
                                         and sign * best['rotationTurns'] > .03)
        return best
    finally:
        e.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--population', type=int, default=48)
    p.add_argument('--generations', type=int, default=8)
    p.add_argument('--codes', nargs='+', default=['101C', '201C', '301C', '401C'])
    args = p.parse_args()
    results = []
    for code in args.codes:
        results.append(search(code, args.population, args.generations))
        Path(args.output).write_text(json.dumps(dict(diagnosticOnly=True, policyTrainingData=False, results=results), indent=2))
