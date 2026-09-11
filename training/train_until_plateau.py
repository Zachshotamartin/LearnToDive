"""Resume original PPO in measured stages until development progress plateaus.

No changes to physics, judge, PPO, or reward. Evaluation is not release
qualification. Completed stages and their optimizer checkpoints are immutable.
"""
import argparse
import fcntl
import math
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from checkpoints import atomic_json, register_evaluation, utc_now

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = {'clean': .01, 'execution': .1, 'alignment': .5, 'geometry': .02}


def metrics(report):
    groups = {'all': report['policies']['pretrained'], **report['perSkill']['pretrained']}
    result = {name: {'clean': s['cleanCompletionRate'], 'execution': s['meanExecution'],
                   'alignment': -s['meanEntryAngle'], 'geometry': s['geometryPassRate']}
            for name, s in groups.items()}
    if len(groups) != 7 or any(not math.isfinite(v) for row in result.values() for v in row.values()):
        raise ValueError('Expected six complete, finite skill evaluations')
    return result


def plateau(history, minimum_steps, patience):
    if len(history) <= patience or history[-1]['additionalSteps'] < minimum_steps:
        return False
    old, recent = history[:-patience], history[-patience:]
    # A meaningful improvement in ANY individual dive keeps training alive.
    # Use fixed cases and running bests, not a noisy single training-return dip.
    for group in history[0]['metrics']:
        for key, threshold in THRESHOLDS.items():
            prior = max(row['metrics'][group][key] for row in old)
            newest = max(row['metrics'][group][key] for row in recent)
            if newest - prior >= threshold:
                return False
    return True


def main(args):
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = out / 'status.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'createdUTC': utc_now(), 'history': [], 'sourceCheckpoint': str(Path(args.checkpoint).resolve()),
        'configuration': vars(args), 'thresholds': THRESHOLDS,
        'definition': 'No meaningful fixed-development-case improvement in any tracked skill over the patience window, after minimum training. Plateau does not mean success.',
        'publication': 'Never automatic; final held-out qualification remains separate.'}
    if state['configuration'] != vars(args):
        raise ValueError('Resume this controller with identical arguments, or use a new output directory')
    stopped = False; child = None
    def stop(signum, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None: child.send_signal(signum)
    for sig in [signal.SIGINT, signal.SIGTERM]: signal.signal(sig, stop)
    def save(**changes):
        state.update(changes, updatedUTC=utc_now(), pid=os.getpid()); atomic_json(path, state)
    def run(command, label):
        nonlocal child
        with (out / (label + '.log')).open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'})
            save(phase=label, childPID=child.pid)
            code = child.wait(); child = None
        if code and not stopped: raise RuntimeError(f'{label} exited {code}; inspect its log')
    def assess(checkpoint, directory, label):
        if not directory.exists():
            temporary = directory.with_name(directory.name + '.freezing-' + str(time.time_ns()))
            run([sys.executable, 'training/freeze_policy.py', str(checkpoint), str(temporary)], label+'-freeze')
            if stopped: return
            os.rename(temporary, directory)
        if stopped: return
        if not all((directory / name).exists() for name in ['evaluation.json', 'cases.json']):
            run([sys.executable, 'training/evaluate.py', '--policy', str(directory/'policy.json'),
                 '--initial', 'training/initial-v8.json', '--per-skill','64','--quick',
                 '--minimum-heights','3','5','7.5','3','7.5','9','--output',str(directory)], label+'-evaluate')
        if stopped: return
        report = json.loads((directory/'evaluation.json').read_text())
        register_evaluation(out, directory)
        return report
    try:
        if not state['history']:
            baseline = assess(Path(state['sourceCheckpoint']), out/'baseline', 'baseline')
            if stopped: save(phase='paused', childPID=None); return
            state['initialSteps'] = baseline['modelSteps']
            state['history'].append({'additionalSteps':0,'metrics':metrics(baseline),
                'checkpoint':state['sourceCheckpoint'], 'evaluation':str(out/'baseline/evaluation.json')})
            save(phase='baseline-evaluated')
        while not stopped:
            if plateau(state['history'], args.minimum_steps, args.patience):
                save(phase='plateau-awaiting-review', childPID=None); return
            index = len(state['history']); name = f'{args.prefix}-{index:04d}'
            directory = ROOT/'training'/name
            checkpoint = Path(state['history'][-1]['checkpoint'])
            pending = state.get('pending')
            # An interrupted stage continues from its last complete saved update.
            already = 0
            if pending and pending['index'] == index and (Path(pending['directory'])/'latest.json').exists():
                prior_dir = Path(pending['directory'])
                latest = json.loads((prior_dir/'latest.json').read_text())
                already = latest['steps'] - (state['initialSteps'] + state['history'][-1]['additionalSteps'])
                checkpoint = prior_dir/'latest.pt'
                name += '-resume-' + str(time.time_ns()); directory = ROOT/'training'/name
            if directory.exists():
                name += '-retry-' + str(time.time_ns()); directory = ROOT/'training'/name
            remaining = args.chunk_steps - already
            if remaining > 0:
                save(pending={'index':index,'directory':str(directory)})
                run([sys.executable,'training/train.py','--name',name,'--resume',str(checkpoint),
                     '--resume-mode','exact','--additional-steps',str(remaining)], f'stage-{index}-train')
                if stopped: break
                checkpoint = directory/'latest.pt'
            evaluated = out/f'evaluation-{index:04d}'
            report = assess(checkpoint, evaluated, f'stage-{index}')
            if stopped: break
            state['history'].append({'additionalSteps':report['modelSteps']-state['initialSteps'],
                'metrics':metrics(report),'checkpoint':str(evaluated/'checkpoint.pt'),
                'evaluation':str(evaluated/'evaluation.json')})
            save(phase='stage-evaluated', pending=None, childPID=None)
        save(phase='paused',childPID=None)
    except Exception as error:
        save(phase='failed', error=str(error),childPID=None); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--prefix', default='ppo-entry-sustained')
    parser.add_argument('--chunk-steps',type=int,default=2048000)
    parser.add_argument('--minimum-steps',type=int,default=20480000)
    parser.add_argument('--patience',type=int,default=5)
    args=parser.parse_args()
    if min(args.chunk_steps,args.minimum_steps,args.patience)<1: parser.error('Positive budgets and patience required')
    main(args)
