"""Bounded CEM search for actual, torque-limited twisting from a real board.

An actuator feasibility experiment only. No searched control sequence is
imported by a trainer or substituted for neural inference in the browser.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from environment import Arena
from geometry import ACTION_LOW, ACTION_HIGH, encoded_action
from rules import IDS


def search(population=64, generations=10, seed=418):
    rng = np.random.default_rng(seed)
    e = Arena(population, seed=seed, threads=2, training=False)
    # Generic press, extension and two independently varied aerial control knots.
    mean = np.array([[.67, 1.73, -.55, .01, .01, 0, 0, 0, 0],
                     [.32, 0, -.13, 3.05, 3.05, -.3, .3, 0, .06],
                     [.5, .3, 1.3, 0., 3., -.8, .8, .2, .05],
                     [0, 0, 1.3, 3., .2, .8, -.8, 0, .05]])
    spread = np.tile((ACTION_HIGH - ACTION_LOW) * .3, (4, 1))
    best = None
    history = []
    try:
        for generation in range(generations):
            samples = np.clip(mean + rng.normal(size=(population, 4, 9)) * spread, ACTION_LOW, ACTION_HIGH)
            samples[0] = mean
            if best is not None:
                samples[1] = best['controls']
            e.apparatus[:] = 1; e.height[:] = 10; e.group[:] = 5; e.used[:] = False
            for i in range(population):
                e.declare(i, IDS['5132D'])
            hold = e.physics.targets.copy()
            maximum_twist = np.zeros(population)
            rotation = np.zeros(population)
            contact = np.zeros(population, bool)
            for tick in range(150):
                time = tick * .02
                if time < .1:
                    target = hold
                else:
                    # Linear actuator targets, no root forces or pose changes.
                    knots = [.1, .46, .76, 1.3]
                    k = min(2, max(0, np.searchsorted(knots, time) - 1))
                    f = np.clip((time - knots[k]) / (knots[k + 1] - knots[k]), 0, 1)
                    target = samples[:, k] * (1 - f) + samples[:, k + 1] * f
                e.physics.step(encoded_action(target), auto_reset=False)
                dry = ~np.isfinite(e.physics.entry_time)
                maximum_twist = np.maximum(maximum_twist, np.abs(e.physics.air_twist) / (2 * np.pi) * dry)
                rotation = np.maximum(rotation, np.abs(e.physics.phase_theta) / (2 * np.pi) * dry)
                contact |= e.physics.board_invalid
            rise = np.maximum(0, e.physics.apex_com - e.physics.departure_com)
            qualified = e.physics.released & ~contact & (e.physics.takeoff_vertical_speed > .5)
            fitness = np.minimum(maximum_twist, 2) + .25 * np.minimum(rotation, 2) + .1 * np.minimum(rise, .5)
            fitness -= 3 * ~qualified
            winner = int(fitness.argmax())
            row = dict(generation=generation, fitness=float(fitness[winner]), twistTurns=float(maximum_twist[winner]),
                       somersaultTurns=float(rotation[winner]), rise=float(rise[winner]),
                       clearTakeoff=bool(qualified[winner]), controls=samples[winner].tolist())
            if best is None or row['fitness'] > best['fitness']:
                best = row
            elite = samples[np.argsort(fitness)[-max(4, population // 8):]]
            mean = .25 * mean + .75 * elite.mean(axis=0)
            spread = np.maximum(.035 * (ACTION_HIGH - ACTION_LOW), .25 * spread + .75 * elite.std(axis=0))
            history.append(row)
            print(json.dumps(row), flush=True)
        return dict(seed=seed, population=population, generations=generations, best=best, history=history,
                    demonstratedHalfTwist=bool(best['clearTakeoff'] and best['twistTurns'] >= .5),
                    rootActuation=False, trainingUsesControls=False,
                    scope='Twisting capability; does not establish a valid declared dive or clean entry')
    finally:
        e.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--population', type=int, default=64)
    p.add_argument('--generations', type=int, default=10)
    a = p.parse_args()
    result = search(a.population, a.generations)
    Path(a.output).write_text(json.dumps(result, indent=2))
