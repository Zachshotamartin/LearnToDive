"""Exercise the actual motor PPO/GAE/PopArt path on a simple inertial control task.

No benchmark states or trajectories ever enter diving training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from policy import Policy
from rules import DIVES
from train import Trainer, estimate_advantages, parser


class PointMass:
    observation_size = 11

    def __init__(self, n=32, seed=1):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.choosing = np.zeros(n, bool)
        self.practice = np.zeros(n, bool)
        self.age = np.zeros(n, int)
        self.previous = np.zeros((n, 9))
        self.x = self.rng.uniform(-2, 2, n)
        self.v = np.zeros(n)

    def observe(self):
        return np.column_stack([self.x, self.v, self.previous]).astype(np.float32)

    def mask(self):
        mask = np.zeros((self.n, len(DIVES)), bool)
        mask[:, 0] = True
        return mask

    def step(self, choices, actions):
        self.v = .97 * self.v + .15 * actions[:, 0]
        self.x += .1 * self.v
        self.previous[:] = actions
        reward = -self.x ** 2 - .05 * self.v ** 2 - .001 * np.sum(actions ** 2, axis=1)
        self.age += 1
        done = self.age >= 80
        self.x[done] = self.rng.uniform(-2, 2, done.sum())
        self.v[done] = 0
        self.age[done] = 0
        self.previous[done] = 0
        return self.observe(), reward, done, []


def evaluate(policy):
    env = PointMass(64, 91919)
    errors = []
    with torch.no_grad():
        for _ in range(75):
            result = policy(torch.tensor(env.observe()), torch.tensor(env.mask()), torch.tensor(env.choosing), deterministic=True)
            env.step(None, result['action'].numpy())
            errors.append(np.mean(env.x ** 2))
    return float(np.mean(errors[25:]))


def run(seed, updates, exploration='diagonal', gae_lambda=.95):
    torch.manual_seed(seed)
    trainer = Trainer.__new__(Trainer)
    trainer.args = parser().parse_args(['--output', '/unused', '--envs', '32', '--horizon', '64', '--batch', '256'])
    trainer.env = PointMass(32, seed)
    trainer.policy = Policy(11, (32, 32), rho=0., exploration=exploration)
    trainer.optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=.0003, eps=1e-5)
    trainer.state = dict(evaluations=[])
    before = evaluate(trainer.policy)
    for _ in range(updates):
        batch = trainer.collect()
        advantages, returns = estimate_advantages(batch, gae_lambda)
        trainer.optimize(batch, advantages, returns)
    after = evaluate(trainer.policy)
    return dict(seed=seed, exploration=exploration, gaeLambda=gae_lambda, beforeMSE=before, afterMSE=after, passed=after < .5 * before)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--updates', type=int, default=160)
    p.add_argument('--exploration', choices=['diagonal', 'state-covariance'], default='diagonal')
    p.add_argument('--gae-lambda', type=float, default=.95)
    args = p.parse_args()
    rows = [run(seed, args.updates, args.exploration, args.gae_lambda) for seed in [811, 812, 813]]
    Path(args.output).write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows), flush=True)
    if not all(r['passed'] for r in rows):
        raise SystemExit('Control benchmark failed: investigate the learner before drawing environment conclusions')
