"""Paired, bounded reward experiments; never promotes weights or stops other runs."""
import argparse
import fcntl
import hashlib
import json
import signal
import subprocess
import sys
from pathlib import Path

import torch
from checkpointing import atomic_json, hashes

HERE = Path(__file__).resolve().parent


def arms_for(suite):
    if suite == 'reward':
        return [('continuous-entry', ['--reward-mode', 'continuous-entry']), ('v12', ['--reward-mode', 'v12'])]
    base = ['--reward-mode', 'continuous-entry']
    if suite == 'initialization':
        settings = base + ['--gae-lambda', '.98', '--recovery-mode', 'progress', '--exploration', 'state-covariance']
        return [('continued', settings), ('fresh', settings)]
    return [
        ('combined', base + ['--gae-lambda', '.98', '--recovery-mode', 'progress', '--exploration', 'state-covariance']),
        ('baseline', base),
        ('lambda-98', base + ['--gae-lambda', '.98']),
        ('lambda-99', base + ['--gae-lambda', '.99']),
        ('recovery-progress', base + ['--recovery-mode', 'progress']),
        ('state-covariance', base + ['--exploration', 'state-covariance']),
    ]


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    arms = arms_for(args.suite)
    identity = dict(arguments=vars(args), arms=arms, sourceHashes=hashes(),
                    controllerHashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('compare_rewards.py','comparison_summary.py')},
                    parentSHA256=hashlib.sha256(Path(args.initialize_from).read_bytes()).hexdigest())
    plan = out / 'PLAN.json'
    identity = json.loads(json.dumps(identity))
    if plan.exists() and json.loads(plan.read_text()) != identity:
        raise ValueError('Comparison inputs changed; choose a new folder')
    atomic_json(plan, identity)
    parent = torch.load(args.initialize_from, map_location='cpu', weights_only=False)
    widths = [str(x) for x in parent['config']['widths']]
    rho = str(parent['config']['rho'])
    stopped = False
    child = None

    def stop(*_):
        nonlocal stopped
        stopped = True
        if child and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for seed in args.seeds:
        for mode, settings in arms:
            dest = out / f'{mode}-{seed}'
            saved = dest / 'latest.pt'
            steps = torch.load(saved, map_location='cpu', weights_only=False)['training']['steps'] if saved.exists() else 0
            report = dest / 'evaluations' / f'{args.steps}.json'
            if steps >= args.steps and report.exists():
                continue
            command = [sys.executable, str(HERE / 'train.py'), '--output', str(dest), '--widths', *widths, '--rho', rho,
                       '--envs', '32', '--threads', '1', '--horizon', '160', '--seed', str(seed),
                       '--steps', str(max(0, args.steps - steps)), '--minimum-steps', str(args.steps * 2),
                       '--evaluate-every', '200', '--eval-cases', '24', *settings]
            command += ['--resume', str(saved)] if saved.exists() else ([] if mode == 'fresh' else ['--initialize-from', args.initialize_from])
            with (out / f'{mode}-{seed}.log').open('a') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(out / 'STATUS.json', dict(phase='training', seed=seed, mode=mode, childPID=child.pid))
                code = child.wait()
            if stopped:
                atomic_json(out / 'STATUS.json', dict(phase='paused', childPID=None))
                return
            if code:
                atomic_json(out / 'STATUS.json', dict(phase='failed', mode=mode, seed=seed, exitCode=code))
                raise RuntimeError('Comparison child failed; inspect its log')
    results = []
    for seed in args.seeds:
        pair = {}
        for mode, settings in arms:
            pair[mode] = json.loads((out / f'{mode}-{seed}' / 'evaluations' / f'{args.steps}.json').read_text())['summary']['full']
        results.append(dict(seed=seed, metrics=pair))
    from comparison_summary import summarize_comparison
    statistics = summarize_comparison(results, 'continued' if args.suite == 'initialization' else 'v12' if args.suite == 'reward' else 'baseline')
    atomic_json(out / 'RESULTS.json', dict(pairedSeeds=results, statistics=statistics, publication='Manual review required; no automatic selection'))
    atomic_json(out / 'STATUS.json', dict(phase='complete-awaiting-review', childPID=None))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=['reward', 'research', 'initialization'], default='reward')
    p.add_argument('--output', required=True)
    p.add_argument('--initialize-from', required=True)
    p.add_argument('--steps', type=int, default=2048000)
    p.add_argument('--seeds', nargs='+', type=int, default=[91201, 91202, 91203, 91204, 91205])
    args = p.parse_args()
    if args.steps <= 0 or args.steps % 5120:
        p.error('Use a positive whole number of 5120-step updates')
    run(args)
