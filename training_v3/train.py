"""Original hybrid PPO; category-conditioned self-declarations and physical motors.
SIGTERM checkpoints at an update boundary. Resume preserves simulator, optimizer,
RNG, curriculum and routine state. Evaluation never publishes browser assets.
"""
import argparse,copy,hashlib,json,math,os,random,signal,time
from pathlib import Path
import numpy as np
import torch
from environment import Arena
from engine import TRAINING_SOURCES
from policy import Policy,FORMAT
from rules import DIVES,DATA
from judge import VERSION
from losses import ppo_terms
from water import WATER_VERSION

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent

def hashes():
 # Only files that change the learning problem are part of the exact-resume contract;
 # editing the suite controller or audit scripts must not strand a paused run.
 return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in TRAINING_SOURCES}
def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)
def atomic_checkpoint(path,value):
 path=Path(path);temp=path.with_suffix('.tmp');torch.save(value,temp);os.replace(temp,path)
def summary(records):
 full=[r for r in records if not r['practice']]
 def group(rows):
  return dict(n=len(rows),points=float(np.mean([r['points'] for r in rows])) if rows else None,
   execution=float(np.mean([r['execution'] for r in rows])) if rows else None,
   difficulty=float(np.mean([r['difficulty'] for r in rows])) if rows else None,
   clean=float(np.mean([r['clean'] for r in rows])) if rows else None,
   valid=float(np.mean([r['valid'] for r in rows])) if rows else None,
   uniqueDives=len({r['declaration'] for r in rows}),
   entryAngle=float(np.mean([r['entryAngle'] for r in rows])) if rows else None,
   trainingReturn=float(np.mean([r['return'] for r in rows])) if rows else None)
 return dict(full=group(full),categories={str(g):group([r for r in full if r['category']==g]) for g in range(1,7)},practiceEpisodes=len(records)-len(full))

@torch.no_grad()
def evaluate(policy,seed=771100,cases=48,perturb=False):
 # Full routines on held-out worlds; recovery curriculum is always disabled.
 e=Arena(cases,seed,threads=4,training=False);result=[];counts=np.zeros(cases,int)
 try:
  e.perturb=perturb
  for _ in range(1200):
   o=torch.tensor(e.observe());out=policy(o,torch.tensor(e.mask()),torch.tensor(e.choosing),deterministic=True)
   _,_,_,rows=e.step(out['choice'].numpy(),out['action'].numpy())
   for row in rows:
    i=row['index']
    if counts[i]<6:result.append(row);counts[i]+=1
   if np.all(counts==6):break
  if len(result)<cases*6:raise RuntimeError('Evaluation failed to complete the declared episode count')
  return dict(seed=seed,perturbed=perturb,summary=summary(result),episodes=result)
 finally:e.close()

def train(args):
 out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'latest.pt').exists() and not args.resume:raise ValueError('Use --resume; never overwrite a saved run')
 seed=args.seed;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 env=Arena(args.envs,seed,args.threads,practice=args.practice);policy=Policy(env.observation_size,tuple(args.widths),args.rho,initial_action=env.initial_action)
 optimizer=torch.optim.Adam(policy.parameters(),lr=args.lr,eps=1e-5)
 state=dict(steps=0,updates=0,elapsedSeconds=0,history=[],evaluations=[],bestValue=-1e30,stopReason=None)
 signature=hashes();contract=dict(format=FORMAT,judge=VERSION,water=WATER_VERSION,observationSize=env.observation_size,actions=9,declarations=[d['id'] for d in DIVES],sourceHashes=signature)
 if args.resume:
  saved=torch.load(args.resume,map_location='cpu',weights_only=False)
  if saved['contract']!=contract:raise ValueError('Changed source or contract; exact resume refused')
  for key in ['envs','widths','rho','horizon','seed','lr','epochs','batch']:
   if saved['config'][key]!=getattr(args,key):raise ValueError('Resume changed '+key)
  policy.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);env.load_state_dict(saved['environment'])
  state=copy.deepcopy(saved['training']);torch.set_rng_state(saved['torchRNG']);np.random.set_state(saved['numpyRNG']);random.setstate(saved['pythonRNG'])
 elif args.warm_start:
  # Only explicitly mapped physical features and existing motor outputs transfer.
  # Old prescribed-target columns and critic are NOT transferred as new experience.
  old=torch.load(args.warm_start,map_location='cpu',weights_only=False)['model']
  width=old['actor.0.weight'].shape[0]
  if width>min(args.widths) or len(args.widths)!=2:raise ValueError('Warm start requires two layers at least as wide as the source')
  mapping={**{i:i for i in range(22)},**{22+i:24+i for i in range(12)},**{34+i:38+i for i in range(8)},**{42+i:47+i for i in range(6)}}
  with torch.no_grad():
   first=policy.trunk[0];second=policy.trunk[2]
   first.weight[:width].zero_();first.bias[:width]=old['actor.0.bias']
   for previous,current in mapping.items():first.weight[:width,current]=old['actor.0.weight'][:,previous]
   second.weight[:width].zero_();second.weight[:width,:width]=old['actor.2.weight'];second.bias[:width]=old['actor.2.bias']
   policy.motor.weight.zero_();policy.motor.weight[:,:width]=old['actor.4.weight'];policy.motor.bias.copy_(old['actor.4.bias'])
  atomic_json(out/'warm-start.json',dict(source=str(Path(args.warm_start).resolve()),sourceSHA256=hashlib.sha256(Path(args.warm_start).read_bytes()).hexdigest(),mapping=mapping,newObjectiveSteps=0,optimizer='fresh',declaration='newly initialized',critic='fresh',notEquivalentToOldPolicy=True))
 stop=False
 def run_evaluation():
  # Evaluation uses a separate RNG and environment; preserve learner randomness.
  rng=torch.get_rng_state();report=evaluate(policy,cases=args.eval_cases);torch.set_rng_state(rng)
  report['steps']=state['steps'];atomic_json(out/'evaluations'/f"{state['steps']}.json",report)
  state['evaluations'].append(dict(steps=state['steps'],summary=report['summary']))
  score=report['summary']['full']['points']
  metrics=report['summary']['full'];selection=(score,metrics['execution'],metrics['clean'],-metrics['entryAngle'])
  if score is not None and selection>tuple(state.get('bestKey',[-1e30]*4)):
   state['bestKey']=selection
   state['bestValue']=score;atomic_checkpoint(out/'best.pt',persist('evaluated'))
   atomic_json(out/'best-policy.json',dict(**policy.export(),contract=contract,steps=state['steps'],qualified=False))
  if state['steps']>=args.minimum_steps and len(state['evaluations'])>args.patience:
   old=state['evaluations'][:-args.patience];new=state['evaluations'][-args.patience:]
   def maxmetric(rows,g,key):return max((r['summary']['categories'][str(g)].get(key) or 0) for r in rows)
   improving=any(maxmetric(new,g,k)>maxmetric(old,g,k)+threshold for g in range(1,7) for k,threshold in [('points',.5),('execution',.1),('clean',.02),('valid',.05)])
   return not improving
  return False
 def request_stop(sig,frame):
  nonlocal stop
  stop=True
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
 started=time.monotonic();prior_seconds=state['elapsedSeconds'];start_steps=state['steps'];block=args.envs*args.horizon
 target=state['steps']+math.ceil(args.steps/block)*block
 def persist(reason):
  state['elapsedSeconds']=prior_seconds+time.monotonic()-started;state['stopReason']=reason
  saved=dict(contract=contract,config=vars(args),model=policy.state_dict(),optimizer=optimizer.state_dict(),environment=env.state_dict(),training=copy.deepcopy(state),torchRNG=torch.get_rng_state(),numpyRNG=np.random.get_state(),pythonRNG=random.getstate())
  atomic_checkpoint(out/'latest.pt',saved)
  atomic_json(out/'STATUS.json',dict(pid=os.getpid(),phase=reason,steps=state['steps'],elapsedSeconds=state['elapsedSeconds'],updates=state['updates'],bestValue=state['bestValue'],latestMetrics=state['history'][-1] if state['history'] else None,publication='Not qualified for browser publication'))
  return saved
 (out/'source').mkdir(exist_ok=True)
 for name in signature:
  import shutil
  shutil.copy2(HERE/name,out/'source'/name)
 atomic_json(out/'contract.json',contract);persist('starting')
 try:
  while state['steps']<target and not stop:
   T,N=args.horizon,args.envs
   observations=torch.zeros(T,N,env.observation_size);masks=torch.zeros(T,N,len(DIVES),dtype=torch.bool)
   choosing=torch.zeros(T,N,dtype=torch.bool);raw=torch.zeros(T,N,9);choices=torch.zeros(T,N,dtype=torch.long)
   oldlog=torch.zeros(T,N);values=torch.zeros(T,N);rewards=torch.zeros(T,N);dones=torch.zeros(T,N);learn=torch.ones(T,N,dtype=torch.bool)
   episodes=[]
   for t in range(T):
    obs=torch.tensor(env.observe());mask=torch.tensor(env.mask());select=torch.tensor(env.choosing)
    with torch.no_grad():pred=policy(obs,mask,select)
    observations[t]=obs;masks[t]=mask;choosing[t]=select;raw[t]=pred['raw'];choices[t]=pred['choice'];oldlog[t]=pred['logp'];values[t]=pred['value']
    _,r,d,finished=env.step(pred['choice'].numpy(),pred['action'].numpy())
    # Recovery snapshots assist motor learning, not declaration selection.
    learn[t]=~(select&torch.tensor(env.practice))
    rewards[t]=torch.tensor(r);dones[t]=torch.tensor(d.astype(np.float32));episodes.extend(finished)
   with torch.no_grad():bootstrap=policy(torch.tensor(env.observe()),torch.tensor(env.mask()),torch.tensor(env.choosing))['value']
   advantages=torch.zeros_like(rewards);gae=torch.zeros(N)
   for t in reversed(range(T)):
    future=bootstrap if t==T-1 else values[t+1];live=1-dones[t]
    delta=rewards[t]+.995*future*live-values[t];gae=delta+.995*.95*live*gae;advantages[t]=gae
   returns=advantages+values
   # Declaration is a sparse high-level decision. Use a full n-step return to
   # the episode boundary, not GAE's exponentially attenuated motor credit.
   decision_return=bootstrap.clone()
   for t in reversed(range(T)):
    decision_return=rewards[t]+.995*decision_return*(1-dones[t])
    advantages[t]=torch.where(choosing[t],decision_return-values[t],advantages[t])
   # Change scale before constructing gradient-bearing forward passes.
   rescale=policy.update_value_scale(returns.flatten())
   for parameter in policy.value.parameters():
    moments=optimizer.state.get(parameter,{})
    if 'exp_avg' in moments:moments['exp_avg'].mul_(rescale);moments['exp_avg_sq'].mul_(rescale**2)
   flat=lambda x:x.flatten(0,1)
   adv=flat(advantages);active=flat(learn)
   for kind in (False,True):
    subset=active&(flat(choosing)==kind)
    if subset.any():adv[subset]=(adv[subset]-adv[subset].mean())/(adv[subset].std(unbiased=False)+1e-8)
   ret=flat(returns);losses=[];kls=[]
   # Refinement is explicit and driven by held-out clean rate, not training reward.
   ent_weight=args.entropy;refinement=1.
   if state['evaluations'] and state['evaluations'][-1]['summary']['full']['clean']>.8:ent_weight*=.25;refinement=.25
   for epoch in range(args.epochs):
    for ids in torch.randperm(T*N).split(args.batch):
     pred=policy(flat(observations)[ids],flat(masks)[ids],flat(choosing)[ids],flat(raw)[ids],flat(choices)[ids])
     selected=active[ids]
     pg,kl=ppo_terms(pred['logp'],flat(oldlog)[ids],adv[ids],flat(choosing)[ids],selected,pred['entropy'],motor_entropy=ent_weight,declaration_entropy=args.declaration_entropy*refinement)
     # Value targets from recovery-practice declaration steps mix practice-assisted
     # returns into V(choosing); they are excluded like their policy gradient.
     value_rows=selected if selected.any() else torch.ones_like(selected)
     vf=.5*(pred['normalizedValue']-(ret[ids]-policy.value_mean)/policy.value_std).square()[value_rows].mean()
     loss=pg+vf
     if not torch.isfinite(loss):raise FloatingPointError('Non-finite PPO loss')
     optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
     with torch.no_grad():policy.logstd.clamp_(-2.8,0)
     losses.append(float(loss.detach()));kls.append(float(kl))
    if np.mean(kls[-math.ceil(T*N/args.batch):])>.02:break
   state['steps']+=block;state['updates']+=1
   row=dict(steps=state['steps'],**summary(episodes),loss=float(np.mean(losses)),kl=float(np.mean(kls)),valueScale=float(policy.value_std))
   state['history'].append(row);print(json.dumps(row,allow_nan=False),flush=True)
   saved=persist('training')
   if state['updates']%args.archive_every==0 or stop or state['steps']>=target:
    folder=out/'checkpoints';folder.mkdir(exist_ok=True);atomic_checkpoint(folder/f"{state['steps']}.pt",saved)
   if not stop and args.evaluate_every and (state['updates']%args.evaluate_every==0 or state['steps']>=target):
    if run_evaluation():persist('plateau-awaiting-review');return state
  # A run killed between its final update and its final evaluation resumes here
  # with the budget complete; produce the missing report instead of skipping it.
  if not stop and args.evaluate_every and state['steps']>=target and not (out/'evaluations'/f"{state['steps']}.json").exists():
   run_evaluation()
  persist('paused' if stop else 'budget-complete-awaiting-review');return state
 except BaseException:
  persist('failed');raise
 finally:env.close()

def parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--warm-start')
 p.add_argument('--widths',type=int,nargs='+',default=[256,256]);p.add_argument('--rho',type=float,default=.6)
 for name,default in [('envs',64),('threads',4),('horizon',160),('epochs',4),('batch',1024),('steps',2048000),('seed',109310),('archive-every',100),('evaluate-every',200),('eval-cases',24),('minimum-steps',102400000),('patience',10)]:p.add_argument('--'+name,type=int,default=default)
 p.add_argument('--lr',type=float,default=.0003);p.add_argument('--entropy',type=float,default=.006);p.add_argument('--declaration-entropy',type=float,default=.01);p.add_argument('--practice',type=float,default=.35)
 return p
if __name__=='__main__':train(parser().parse_args())
