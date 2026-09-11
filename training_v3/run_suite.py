"""Bounded matched trials followed by a resumable plateau-controlled run.
This is a training process, not an agent monitoring loop. No browser promotion.
"""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from train import atomic_json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def main(a):
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 lock=(out/'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 stop=False;child=None
 state_path=out/'SUITE.json';state=json.loads(state_path.read_text()) if state_path.exists() else dict(config=vars(a),trials=[],publication='No automatic publication')
 if state['config']!=vars(a):raise ValueError('Resume with the same suite settings')
 def save(**kw):state.update(kw,pid=os.getpid(),updated=time.time());atomic_json(state_path,state)
 def halt(sig,frame):
  nonlocal stop
  stop=True
  if child is not None and child.poll() is None:child.send_signal(sig)
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,halt)
 def run(name,widths,steps,seed,warm=None):
  nonlocal child
  folder=out/name;status=folder/'STATUS.json'
  if status.exists() and json.loads(status.read_text())['phase'] in ('budget-complete-awaiting-review','plateau-awaiting-review'):return
  command=[sys.executable,str(HERE/'train.py'),'--output',str(folder),'--widths',*map(str,widths),'--steps',str(steps),'--seed',str(seed),'--envs','64','--threads','4','--evaluate-every','200','--archive-every','100']
  checkpoint=folder/'latest.pt'
  if checkpoint.exists():
   old=json.loads(status.read_text());remaining=steps-old['steps']
   if remaining<=0:return
   command[command.index('--steps')+1]=str(remaining);command+=['--resume',str(checkpoint)]
  elif warm:command+=['--warm-start',warm]
  save(phase='training',current=name)
  with (out/(name+'.log')).open('a') as log:
   child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});save(childPID=child.pid)
   code=child.wait();child=None
  if code and not stop:raise RuntimeError(f'{name} failed; see its log')
 configurations={'96x96':[96,96],'256x256':[256,256],'256x256x128':[256,256,128]}
 if a.warm_start:configurations['256x256-warm']=[256,256]
 try:
  for name,widths in configurations.items():
   for seed in a.seeds:
    trial=f'{name}-seed-{seed}';run(trial,widths,a.pilot_steps,seed,a.warm_start if name.endswith('-warm') else None)
    if stop:save(phase='paused',childPID=None);return
    report_path=out/trial/'evaluations'/f'{a.pilot_steps}.json'
    if not report_path.exists():raise RuntimeError('Pilot missing complete fixed-case evaluation')
    report=json.loads(report_path.read_text());record=dict(name=trial,architecture=name,seed=seed,metrics=report['summary']['full'])
    state['trials']=[x for x in state['trials'] if x['name']!=trial]+[record];save(phase='pilot-evaluated')
  # Rank architecture medians across seeds, never one lucky maximum seed.
  scores={}
  for name in configurations:
   rows=[r['metrics'] for r in state['trials'] if r['architecture']==name]
   scores[name]=tuple(float(np.median([r[k] for r in rows]))*sign for k,sign in [('points',1),('execution',1),('clean',1),('entryAngle',-1)])
  winner=max(scores,key=scores.get);candidates=[r for r in state['trials'] if r['architecture']==winner]
  chosen=sorted(candidates,key=lambda r:(r['metrics']['points'],r['metrics']['execution'],-r['metrics']['entryAngle']))[len(candidates)//2]
  save(phase='pilot-comparison-complete',architectureScores=scores,selected=chosen,selectionEvidence='Median fixed-development score across seeds; alignment is a tie breaker, not qualification. Full held-out qualification remains required.')
  # Continue the median seed from its exact optimizer/environment checkpoint.
  long=out/'continued';long.mkdir(exist_ok=True)
  if not (long/'latest.pt').exists():
   import shutil
   shutil.copy2(out/chosen['name']/'latest.pt',long/'latest.pt')
   metadata=json.loads((out/chosen['name']/'STATUS.json').read_text());metadata['phase']='paused';atomic_json(long/'STATUS.json',metadata)
  run('continued',configurations[winner],a.total_steps,chosen['seed'])
  save(phase='paused' if stop else 'finished-awaiting-review',childPID=None)
 except BaseException as error:save(phase='failed',error=repr(error),childPID=None);raise
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--warm-start');p.add_argument('--seeds',type=int,nargs='+',default=[109310,109311,109312]);p.add_argument('--pilot-steps',type=int,default=10240000);p.add_argument('--total-steps',type=int,default=512000000)
 args=p.parse_args()
 if args.pilot_steps%10240 or args.total_steps<=args.pilot_steps:p.error('Use complete 10,240-interaction batches and a larger continuation budget')
 main(args)
