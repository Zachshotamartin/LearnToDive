"""Bounded matched trials followed by a resumable plateau-controlled run.

This is a training process, not an agent monitoring loop, and it never
promotes anything to the browser.
"""
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from checkpointing import atomic_json
from model_selection import RANKING

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIGURATIONS = {'96x96': [96, 96], '256x256': [256, 256], '256x256x128': [256, 256, 128]}
# The final configuration: conjunctive credit, coherent exploration, normalised inputs,
# mastery stages, adaptive motor practice with the entry sub-task, split critic.
TRIAL_ARGUMENTS = ['--envs', '128', '--threads', '4', '--horizon', '160', '--batch', '2048', '--epochs', '3',
                   '--evaluate-every', '100', '--archive-every', '100', '--reward-mode', 'conjunctive',
                   '--motor-curriculum', 'adaptive', '--direction-practice', '--goal-practice', '--rotation-progress',
                   '--architecture', 'split', '--recovery-mode', 'progress', '--practice', '.15', '--gae-lambda', '.99',
                   '--noise-rho', '.9', '--input-normalization', '1', '--stage-curriculum', '1', '--gates', '1',
                   '--motor-init', 'entry-pose', '--motor-logstd', '-2', '--motor-logstd-max', '-1.9', '--entropy', '.002',
                   '--final-seed', '883117']
# The continuation may only stop on a plateau after a quarter of a billion steps and
# twenty-five evaluation windows (51M steps) without any category improving.
CONTINUATION_ARGUMENTS = ['--minimum-steps', '256000000', '--patience', '25']
FINISHED_PHASES = ('budget-complete-awaiting-review', 'plateau-awaiting-review')
ATTEMPTS = 3
RETRY_DELAY = 30
BATCH = 10240
SELECTION_EVIDENCE = ('Median fixed-development score across seeds; alignment is a tie breaker, not qualification. '
                      'Full held-out qualification remains required.')
UNINFORMATIVE = ('No pilot completed a single declared dive; the environment needs review before spending the '
                 'continuation budget.')


class Suite:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / '.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.stop = False
        self.child = None
        self.state_path = self.out / 'SUITE.json'
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = dict(config=vars(args), trials=[], publication='No automatic publication')
        if self.identity(self.state['config']) != self.identity(vars(args)):
            raise ValueError('Resume with the same suite settings')
        self.configurations = {name: CONFIGURATIONS[name] for name in args.architectures}
        if args.warm_start:
            self.configurations['256x256-warm'] = [256, 256]

    @staticmethod
    def identity(config):
        """The settings that define a suite; an explicit source amendment does not."""
        return {k: v for k, v in config.items() if k != 'accept_source_change'}

    def save(self, **fields):
        self.state.update(fields, pid=os.getpid(), updated=time.time())
        atomic_json(self.state_path, self.state)

    def halt(self, sig, frame):
        self.stop = True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(sig)

    # -------------------------------------------------------------- one trial
    def command(self, name, widths, steps, seed, warm):
        """The train.py command for this trial, or None when its budget is complete."""
        folder = self.out / name
        status = folder / 'STATUS.json'
        checkpoint = folder / 'latest.pt'
        command = [sys.executable, str(HERE / 'train.py'), '--output', str(folder), '--widths', *map(str, widths),
                   '--steps', str(steps), '--seed', str(seed), *TRIAL_ARGUMENTS]
        if name == 'continued':
            command += CONTINUATION_ARGUMENTS
        if status.exists() and json.loads(status.read_text())['phase'] in FINISHED_PHASES:
            return None
        if checkpoint.exists():
            remaining = steps - json.loads(status.read_text())['steps']
            if remaining <= 0:
                return None
            command[command.index('--steps') + 1] = str(remaining)
            command += ['--resume', str(checkpoint)]
            if self.args.accept_source_change:
                command += ['--accept-source-change', *self.args.accept_source_change]
        elif warm:
            command += ['--warm-start', warm]
        return command

    def launch(self, name, command, attempt):
        environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        with (self.out / (name + '.log')).open('a') as log:
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment)
            self.save(childPID=self.child.pid, attempt=attempt)
            code = self.child.wait()
            self.child = None
        return code

    def run(self, name, widths, steps, seed, warm=None):
        """Train one trial to its step budget, resuming from its own checkpoint after a crash."""
        if self.command(name, widths, steps, seed, warm) is None:
            return
        self.save(phase='training', current=name)
        for attempt in range(ATTEMPTS):
            command = self.command(name, widths, steps, seed, warm)
            if command is None:
                return
            code = self.launch(name, command, attempt)
            if not code or self.stop:
                return
            # A transient failure resumes from the last saved update; a repeatable one stops the suite.
            self.save(phase='retrying', lastExitCode=code)
            time.sleep(RETRY_DELAY)
        raise RuntimeError(f'{name} failed {ATTEMPTS} times; see its log')

    # ---------------------------------------------------------------- pilots
    def pilots(self):
        """Every architecture on every seed for the pilot budget; False when interrupted."""
        for name, widths in self.configurations.items():
            warm = self.args.warm_start if name.endswith('-warm') else None
            for seed in self.args.seeds:
                trial = f'{name}-seed-{seed}'
                self.run(trial, widths, self.args.pilot_steps, seed, warm)
                if self.stop:
                    return False
                # Milestones are recorded per trial and surfaced here for review; a
                # missed one is evidence, not a reason to stop spending budget.
                status = json.loads((self.out / trial / 'STATUS.json').read_text())
                self.save(milestones={**self.state.get('milestones', {}), trial: status.get('gates', [])})
                self.record_pilot(trial, name, seed)
        return True

    def record_pilot(self, trial, name, seed):
        """Rank a pilot by its champion evaluation, the checkpoint a continuation starts from."""
        final_path = self.out / trial / 'evaluations' / f'{self.args.pilot_steps}.json'
        if not final_path.exists():
            raise RuntimeError('Pilot missing complete fixed-case evaluation')
        final = json.loads(final_path.read_text())
        champion = json.loads((self.out / trial / 'selection.json').read_text())['champion']
        report = json.loads((self.out / trial / 'evaluations' / f"{champion['steps']}.json").read_text())
        record = dict(name=trial, architecture=name, seed=seed, metrics=report['summary']['full'],
                      championSteps=champion['steps'], finalMetrics=final['summary']['full'])
        self.state['trials'] = [x for x in self.state['trials'] if x['name'] != trial] + [record]
        self.save(phase='pilot-evaluated')

    def select(self):
        """Rank architecture medians across seeds, never one lucky maximum seed.

        Points come first; the dense development return (the actual training
        objective on the fixed cases) breaks ties while points are still zero.
        """
        trials = self.state['trials']
        scores = {}
        for name in self.configurations:
            rows = [r['metrics'] for r in trials if r['architecture'] == name]
            scores[name] = tuple(float(np.median([r[key] for r in rows])) for key in RANKING)
        if all(r['metrics']['valid'] == 0 for r in trials):
            self.save(phase='pilot-comparison-uninformative', architectureScores=scores, reason=UNINFORMATIVE)
            raise RuntimeError('Pilot comparison uninformative: no completed declared dive in any pilot')
        winner = max(scores, key=scores.get)
        candidates = [r for r in trials if r['architecture'] == winner]
        order = lambda r: (r['metrics']['clean'], r['metrics']['execution'], r['metrics']['points'], r['metrics']['trainingReturn'])
        chosen = sorted(candidates, key=order)[len(candidates) // 2]
        self.save(phase='pilot-comparison-complete', architectureScores=scores, selected=chosen, selectionEvidence=SELECTION_EVIDENCE)
        return winner, chosen

    def continuation(self, winner, chosen):
        """Continue the median seed from its champion's exact optimizer and environment checkpoint.

        A pilot's final weights can already be past its peak (v13.5 continued
        from 7.8 points when the same seed had scored 11.3 two evaluations earlier).
        """
        long = self.out / 'continued'
        long.mkdir(exist_ok=True)
        source = self.out / chosen['name']
        if not (long / 'latest.pt').exists():
            shutil.copy2(source / 'best.pt', long / 'latest.pt')
            shutil.copy2(source / 'best.pt', long / 'best.pt')
            metadata = json.loads((source / 'STATUS.json').read_text())
            metadata['phase'] = 'paused'
            metadata['steps'] = chosen['championSteps']
            metadata['continuedFrom'] = dict(trial=chosen['name'], checkpoint='best.pt', steps=chosen['championSteps'])
            atomic_json(long / 'STATUS.json', metadata)
        self.run('continued', self.configurations[winner], self.args.total_steps, chosen['seed'])

    def main(self):
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self.halt)
        try:
            if not self.pilots():
                self.save(phase='paused', childPID=None)
                return
            winner, chosen = self.select()
            self.continuation(winner, chosen)
            self.save(phase='paused' if self.stop else 'finished-awaiting-review', childPID=None)
        except BaseException as error:
            self.save(phase='failed', error=repr(error), childPID=None)
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--warm-start')
    p.add_argument('--accept-source-change', nargs='*', default=[])
    p.add_argument('--architectures', nargs='+', choices=list(CONFIGURATIONS), default=list(CONFIGURATIONS))
    p.add_argument('--seeds', type=int, nargs='+', default=[109310, 109311, 109312])
    p.add_argument('--pilot-steps', type=int, default=10240000)
    p.add_argument('--total-steps', type=int, default=512000000)
    return p


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.pilot_steps % BATCH or arguments.total_steps <= arguments.pilot_steps:
        parser().error('Use complete 10,240-interaction batches and a larger continuation budget')
    Suite(arguments).main()
