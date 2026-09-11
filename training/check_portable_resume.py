"""Prove that a native checkpoint resumes from only the selected delivery files.

Uses installed pinned dependencies, but no ignored phase directory or local model
path. The temporary training update is a smoke test, never a release candidate.
"""
import argparse,hashlib,io,json,subprocess,sys,tempfile
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
# Copy source modules, never ignored training outputs or another local checkout.
# This includes the water model and checkpoint helpers as the trainer evolves.
SOURCE_FILES=sorted({
 *[str(path.relative_to(ROOT)) for path in (ROOT/'training').glob('*.py')],
 *[str(path.relative_to(ROOT)) for path in (ROOT/'src/core').glob('*.js')],
 'training/requirements.txt','public/physics/diver.xml',
})

def optimizer_step(checkpoint):
 return max((float(state.get('step',0)) for state in checkpoint['optimizer']['state'].values()),default=0)

def check(checkpoint_path):
 payload=Path(checkpoint_path).read_bytes();before=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
 config=before['config'];sha=hashlib.sha256(payload).hexdigest()
 with tempfile.TemporaryDirectory(prefix='dive-portable-resume-') as temporary:
  clone=Path(temporary)
  for relative in SOURCE_FILES:
   target=clone/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
  target=clone/'training/release/checkpoint.pt';target.parent.mkdir(parents=True);target.write_bytes(payload)
  exact=before.get('checkpointFormat')=='dive-training-state-v2'
  additional=config.get('envs',128)*config.get('horizon',160)
  mode='exact' if exact else 'restart'
  args=[sys.executable,'training/train.py','--resume','training/release/checkpoint.pt','--name','portable-smoke','--additional-steps',str(additional),'--resume-mode',mode]
  if not exact:
   # Legacy files cannot restore physical worlds they never saved. Preserve
   # weights/optimizer but explicitly restart worlds; do not call this exact.
   for key in ['envs','horizon','epochs','batch','seed','lr','entropy','mean_penalty','execution_weight','ascent_weight','launch_speed_weight','entry_geometry_weight','maximum_std','min_height','max_height','boundary_rate','board_leg_initial_std','board_leg_minimum_std','board_leg_maximum_std']:
    if key in config:args.extend(['--'+key.replace('_','-'),str(config[key])])
   for key in ['skills','skill_min_heights']:
    if config.get(key) is not None:args.extend(['--'+key.replace('_','-'),*[str(v)for v in config[key]]])
   if config.get('phase_exploration'):args.append('--phase-exploration')
  completed=subprocess.run(args,cwd=clone,capture_output=True,text=True,timeout=180)
  if completed.returncode:raise RuntimeError(completed.stdout+'\n'+completed.stderr)
  after=torch.load(clone/'training/portable-smoke/latest.pt',map_location='cpu',weights_only=False)
  assert after['steps']==before['steps']+additional
  assert after['contract']['stepsOnContract']==before['contract']['stepsOnContract']+additional
  assert optimizer_step(after)>optimizer_step(before)
  assert after['provenance']['parentCheckpointSHA256']==sha
  assert after['contract']['xmlSHA256']==before['contract']['xmlSHA256']
  assert tuple(after['model']['logstd'].shape)==tuple(before['model']['logstd'].shape)
  # Continue the newly completed run as well. This catches accidental target
  # ceilings and verifies the complete saved state survives another process.
  extend=[sys.executable,'training/train.py','--resume','training/portable-smoke/latest.pt','--name','portable-exact-extension','--additional-steps',str(additional),'--resume-mode','exact']
  extension=subprocess.run(extend,cwd=clone,capture_output=True,text=True,timeout=180)
  if extension.returncode:raise RuntimeError(extension.stdout+'\n'+extension.stderr)
  extended=torch.load(clone/'training/portable-exact-extension/latest.pt',map_location='cpu',weights_only=False)
  assert extended['steps']==after['steps']+additional
  assert optimizer_step(extended)>optimizer_step(after)
  assert extended['trainingState']['updatesCompleted']==after['trainingState']['updatesCompleted']+1
  assert extended['trainingState']['cumulativeElapsedSeconds']>=after['trainingState']['cumulativeElapsedSeconds']
  assert sha==hashlib.sha256(target.read_bytes()).hexdigest()
  return dict(status='passed',resumeMode=mode,scope='Two temporary PPO updates in a source-only directory, including exact continuation beyond a completed run; not release training.',checkpointSHA256=sha,sourceFiles=SOURCE_FILES,stepsBefore=before['steps'],stepsAfter=after['steps'],exactExtensionSteps=extended['steps'],contractStepsBefore=before['contract']['stepsOnContract'],contractStepsAfter=after['contract']['stepsOnContract'],explorationShapeBefore=list(before['model']['logstd'].shape),explorationShapeAfter=list(after['model']['logstd'].shape),optimizerStepBefore=optimizer_step(before),optimizerStepAfter=optimizer_step(after),exactExtensionOptimizerStep=optimizer_step(extended),command=['python',*args[1:]],extensionCommand=['python',*extend[1:]],output=completed.stdout+'\n'+extension.stdout)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--output');args=parser.parse_args();report=check(args.checkpoint)
 if args.output:
  output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items()if k!='output'},indent=2))
