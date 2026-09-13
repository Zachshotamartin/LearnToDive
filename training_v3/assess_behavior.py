"""Read-only paired mean/sampled playback, judged outcomes and dive phase diagnostics."""
import argparse
import hashlib
import io
from pathlib import Path
import torch
from assessment_stats import paired_interval
from checkpointing import atomic_json, hashes
from evaluation import evaluate
from policy import Policy, FORMAT, MOTOR_FORMAT


def load_policy(checkpoint):
    saved = checkpoint if isinstance(checkpoint,dict) else torch.load(checkpoint, map_location='cpu', weights_only=False)
    contract, config = saved['contract'], saved['config']
    if contract['format'] not in (FORMAT, MOTOR_FORMAT):
        raise ValueError('Unsupported observation/policy format')
    for name in ('diver.xml','geometry.py','stance.py','water.py'):
        if contract['sourceHashes'][name] != hashes()[name]:
            raise ValueError('Physical model changed: ' + name)
    policy = Policy(contract['observationSize'], config['widths'], config['rho'],
                    exploration=config.get('exploration', 'diagonal'), architecture=config.get('architecture', 'shared'))
    policy.load_state_dict(saved['model']); policy.eval().requires_grad_(False)
    return policy, saved


def assess(checkpoint, seeds, cases=24):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Use distinct evaluation seeds')
    raw = Path(checkpoint).read_bytes()
    policy, saved = load_policy(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False))
    rows = []
    for seed in seeds:
        for mode in ('mean','sample'):
            report = evaluate(policy, seed=seed, cases=cases, deterministic=mode=='mean',
                              policy_seed=seed+17, reward_mode=saved['config'].get('reward_mode','v12'))
            rows.append(dict(mode=mode, **report))
    metrics=('points','execution','clean','jumped','completedRotation','entryAngle','entryAngularSpeed',
             'postContactRotationDegrees','entryAngleDeterioration')
    contrasts={}
    for key in metrics:
        a,b=[{str(r['seed']):r['summary']['full'][key] for r in rows if r['mode']==mode} for mode in ('mean','sample')]
        contrasts[key]=paired_interval(a,b)
    return dict(format='diver-behavior-assessment-v1', checkpointSHA256=hashlib.sha256(raw).hexdigest(),
                checkpointSteps=saved['training']['steps'], sourceHashes=hashes(), evaluations=rows, contrasts=contrasts,
                phaseDefinitions=dict(jumped='upward departure speed > 0.5 m/s', completedRotation='declared somersault and twist tolerances met',
                    postContactRotationDegrees='20 ms sampled integral of angular speed after first water contact; total rotation, not signed somersault count',
                    entryAngleDeterioration='worst entry angle minus first-contact angle, in degrees'),
                scope='Paired evaluation-seed clusters for a single fixed checkpoint; not independent training runs or proof of generalization.',
                automaticPublication=False, optimizerSteps=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(881100,881108)))
    p.add_argument('--cases',type=int,default=24)
    a=p.parse_args();out=Path(a.output)
    if out.exists(): raise ValueError('Preserve completed assessments')
    torch.set_num_threads(1)
    atomic_json(out,assess(a.checkpoint,a.seeds,a.cases))
