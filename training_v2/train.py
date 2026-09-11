"""Original PPO training for the MuJoCo joint-feedback actor. No demonstration data.
Run with the pinned environment in requirements.txt. Checkpoints include optimizer,
RNG, measured episode summaries, and all configuration needed to resume honestly.
"""
import argparse,json,time,os,hashlib,shutil,math,copy,signal,sys,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
from sim import Arena,ROOT,SKILLS
from contract import runtime_contract, validate_contract
from critic_calibration import calibrate
from rewards import REWARD_VERSION
from checkpoints import save_checkpoint,restore_environment,restore_rng,validate_exact,requested_interactions,utc_now,atomic_json

torch.set_num_threads(1)
def cap_exploration(agent,maximum_std,board_minimum_std=.35,board_maximum_std=.5):
 if not math.exp(-2.8)<=maximum_std<=1:raise ValueError('Maximum standard deviation must be between exp(-2.8) and 1')
 with torch.no_grad():
  if agent.logstd.ndim==2:agent.logstd.clamp_(-2.8,math.log(maximum_std))
  else:
   if not math.exp(-2.8)<=board_minimum_std<=board_maximum_std<=1:raise ValueError('Invalid grounded-leg exploration bounds')
   agent.logstd[:,1].clamp_(-2.8,math.log(maximum_std))
   agent.logstd[:,0,3:].clamp_(-2.8,math.log(maximum_std))
   agent.logstd[:,0,:3].clamp_(math.log(board_minimum_std),math.log(board_maximum_std))

def restore_optimizer(opt,checkpoint,agent,reset_optimizer):
 """Optimizer tensors belong to their original parameter shapes. A distribution
 extension is a named weights-only phase, never an implicit optimizer migration.
 """
 old_shape=tuple(checkpoint['model']['logstd'].shape)
 if old_shape!=tuple(agent.logstd.shape) and not reset_optimizer:
  raise ValueError('Exploration distribution shape changed; --reset-optimizer is required')
 if not reset_optimizer:opt.load_state_dict(copy.deepcopy(checkpoint['optimizer']))

class ActorCritic(nn.Module):
 def __init__(self,obs=76,hidden=96,actions=9,phase_exploration=False):
  super().__init__();self.actor=nn.Sequential(nn.Linear(obs,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,actions))
  self.critic=nn.Sequential(nn.Linear(obs,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1));self.logstd=nn.Parameter(torch.ones((6,2,actions) if phase_exploration else (6,actions))*-.5)
  for module in self.modules():
   if isinstance(module,nn.Linear):nn.init.orthogonal_(module.weight,np.sqrt(2));nn.init.zeros_(module.bias)
  nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
 def load_state_dict(self,state_dict,strict=True):
  if state_dict['logstd'].ndim==1:state_dict=dict(state_dict);state_dict['logstd']=state_dict['logstd'][None].repeat(6,1)
  if state_dict['logstd'].ndim==2 and self.logstd.ndim==3:
   state_dict=dict(state_dict);state_dict['logstd']=state_dict['logstd'][:,None].repeat(1,2,1)
  if state_dict['logstd'].ndim!=self.logstd.ndim:raise ValueError('This checkpoint requires --phase-exploration')
  return super().load_state_dict(state_dict,strict=strict)
 def migrate_from_v6(self,state):
  """Explicit architectural extension: retain all existing motor computations.
  New sensory columns start at exactly zero; the independent new output row
  retains its seeded small random initialization and gets a neutral-target bias.
  """
  if tuple(state['actor.0.weight'].shape)!=(96,67) or tuple(state['actor.4.weight'].shape)!=(8,96):raise ValueError('Only the audited 67-input/8-action v6 architecture can be extended here')
  updated=self.state_dict()
  for key,value in state.items():
   if key in ['actor.0.weight','critic.0.weight']:
    updated[key].zero_();updated[key][:,:67]=value
   elif key in ['actor.4.weight','actor.4.bias']:
    updated[key][:8]=value
   elif key=='logstd':
    if updated[key].ndim==3:updated[key][:,:,:8]=value[:,None]
    else:updated[key][:,:8]=value
   else:updated[key]=value
  neutral_action=2*(0-(-.15))/(.18-(-.15))-1
  updated['actor.4.bias'][8]=2*math.atanh(math.atanh(neutral_action)/2)
  updated['logstd'][...,8]=-.5
  return super().load_state_dict(updated)
 def migrate_from_v7(self,state):
  """V8 preserves every existing computation and appends zero sensory columns.
  The changed full-entry physics/objective requires a fresh optimizer and new
  evidence. No old training decision is counted on this contract.
  """
  if tuple(state['actor.0.weight'].shape)!=(96,72) or tuple(state['actor.4.weight'].shape)!=(9,96):raise ValueError('Expected the audited72/9 v7 architecture')
  updated={k:v.clone() for k,v in state.items()}
  for key in ['actor.0.weight','critic.0.weight']:
   updated[key]=torch.zeros_like(self.state_dict()[key]);updated[key][:,:72]=state[key]
  return self.load_state_dict(updated)
 def mean_logits(self,obs):
  # Keep useful exploration available; the Gaussian's latent mean remains
  # smoothly within ±2 before the separate action tanh. No action is clipped.
  return 2*torch.tanh(self.actor(obs)/2)
 def mean_action(self,obs):return torch.tanh(self.mean_logits(obs))
 @staticmethod
 def log_jacobian(z):
  return 2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
 @classmethod
 def statistics(cls,dist,z,include_entropy=True):
  logp=(dist.log_prob(z)-cls.log_jacobian(z)).sum(-1)
  entropy=None
  if include_entropy:
   # Reparameterized expectation of the actual bounded distribution entropy.
   entropy=(dist.entropy()+cls.log_jacobian(dist.rsample())).sum(-1)
  return logp,entropy
 def get(self,obs,latent=None):
  skill=torch.where(obs[:,64]>.1,5,torch.where(obs[:,65]>.5,torch.where(obs[:,63]>.4,4,3),torch.where(obs[:,66]>.1,2,torch.where(obs[:,63]>.4,1,0))))
  variance=self.logstd[skill,self.exploration_phase(obs)] if self.logstd.ndim==3 else self.logstd[skill]
  dist=Normal(self.mean_logits(obs),variance.exp())
  z=dist.sample() if latent is None else latent
  logp,entropy=self.statistics(dist,z,include_entropy=latent is not None)
  return torch.tanh(z),logp,entropy,self.critic(obs).squeeze(-1),z
 @staticmethod
 def exploration_phase(obs):
  # Larger noise is limited to observed loaded-foot contact before release.
  # Brief contact gaps and all confirmed flight retain the narrow variance.
  grounded=(obs[:,58]<=.5)&((obs[:,61]>.5)|(obs[:,62]>.5))
  return (~grounded).long()
 def save_json(self,path,steps,config,contract=None):
  data={'contract':contract or runtime_contract(),'format':'mujoco-joint-ppo-v1','physics':'MuJoCo 3.13.0','observationSize':self.actor[0].in_features,'actionSize':self.actor[-1].out_features,'hidden':96,'activation':'tanh','outputActivation':'squashed-mean-v1','latentMeanLimit':2,'actionDistribution':'tanh-squashed-gaussian-v1','steps':steps,'config':config,'layers':[]}
  for l in self.actor:
   if isinstance(l,nn.Linear):data['layers'].append({'weight':l.weight.detach().numpy().round(8).tolist(),'bias':l.bias.detach().numpy().round(8).tolist()})
  Path(path).write_text(json.dumps(data,separators=(',',':')))

def episode_statistics(episode,skills):
 def mean(rows,key,digits=4):return round(float(np.mean([r[key]for r in rows])),digits)if rows else None
 groups={}
 fields={'entryPractice':'practice','rotation':'rotation','valid':'valid','postureCoverage':'shapeFraction','meanAscent':'ascent','meanTakeoffVerticalSpeed':'takeoffVerticalSpeed','meanEntryAngle':'entryAngle','meanPreparationBounces':'preparationBounces','boardPeak':'boardNonfootPeak'}
 for skill in skills:
  rows=[r for r in episode if r['skill']==skill and not r.get('practice',False)]
  groups[str(skill)]={'n':len(rows),**{label:mean(rows,key)for label,key in fields.items()},'launchClear':mean([{'clear':not r['boardInvalid']}for r in rows],'clear')}
 full=[r for r in episode if not r.get('practice',False)];practice=[r for r in episode if r.get('practice',False)]
 return {'return':mean(full,'return_'),'valid':mean(full,'valid'),'skills':groups,'fullDiveEpisodes':len(full),'entryPractice':{'episodes':len(practice),'return':mean(practice,'return_'),'meanEntryAngle':mean(practice,'entryAngle'),'valid':mean(practice,'valid')}}

def train(args):
 segment_start=time.monotonic();segment_started_utc=utc_now();stop_requested={'signal':None}
 def stop_after_update(signum,frame):stop_requested['signal']=signal.Signals(signum).name
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop_after_update)
 exact=bool(args.resume and args.resume_mode=='exact')
 if exact and args.reset_optimizer:raise ValueError('Exact resume cannot reset the optimizer; choose --resume-mode restart')
 random.seed(args.seed)
 np.random.seed(args.seed);torch.manual_seed(args.seed);env=Arena(args.envs,seed=args.seed,threads=args.threads,skills=args.skills,heights=(args.min_height,args.max_height),minimum_heights=args.skill_min_heights,boundary_rate=args.boundary_rate,execution_weight=args.execution_weight,ascent_weight=args.ascent_weight,launch_speed_weight=args.launch_speed_weight,entry_geometry_weight=args.entry_geometry_weight,entry_curriculum=args.entry_curriculum)
 agent=ActorCritic(phase_exploration=args.phase_exploration);opt=torch.optim.Adam(agent.parameters(),lr=args.lr,eps=1e-5);total=0
 out=ROOT/'training_v2'/args.name
 if out.exists():raise ValueError('Run directory already exists. Resume any saved checkpoint into a NEW --name to preserve the parent.')
 out.mkdir(parents=True)
 source=out/'source';source.mkdir(exist_ok=True)
 for path in [ROOT/'training_v2/sim.py',ROOT/'training_v2/train.py',ROOT/'training_v2/contract.py',ROOT/'training_v2/evaluate.py',ROOT/'src/core/control.js',ROOT/'src/core/physics.js',ROOT/'public/physics/diver.xml',ROOT/'training_v2/water.py',ROOT/'src/core/water.js',ROOT/'training_v2/critic_calibration.py',ROOT/'training_v2/checkpoints.py',ROOT/'training_v2/rewards.py']:
  shutil.copy2(path,source/path.name)
 (source/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.name!='sha256.json'},indent=2))
 contract=runtime_contract(source/'diver.xml');contract['trainingCurriculum']='attainable-pre-entry-replay-v1' if args.entry_curriculum else 'full-dives';contract['trainingSourceSHA256']=json.loads((source/'sha256.json').read_text());contract['stepsOnContract']=0;contract['actorUpdateInteractions']=0;contract['criticCalibrationInteractions']=0;contract['criticCalibrationRetainedSamples']=0
 provenance={'rewardVersion':REWARD_VERSION,'contract':contract,'actionDistribution':'tanh-squashed-gaussian-v1','migration':'Earlier phases used Gaussian actions with environment clipping. This phase uses a tanh change of variables and its bounded-action entropy; old phases are not reclassified.','config':vars(args),'resumeMode':('weights continuation with fresh optimizer' if args.reset_optimizer else 'weights and optimizer continuation')+('; exact update-boundary environment and RNG restoration' if exact else '; environment and RNG freshly initialized with the declared seed'),'launchCreditSemantics':'final-departure-telescoping-v2: recontact debits all earlier launch potential; confirmed final departure receives its own measured stored velocity credit','sourceHashes':json.loads((source/'sha256.json').read_text())}
 if args.resume:
  c=torch.load(args.resume,weights_only=False,map_location='cpu')
  if exact:
   validate_exact(c,provenance['sourceHashes'])
   allowed={'name','resume','resume_mode','steps','target_steps','checkpoint_every_steps','checkpoint_every_seconds','reset_optimizer','allow_physics_migration','critic_calibration_rounds','critic_calibration_lr','critic_calibration_epochs'}
   for key,value in c['config'].items():
    if key not in allowed and key in vars(args) and vars(args)[key]!=value:raise ValueError(f'Exact resume setting changed: {key}; use a named restart')
  if c['model']['actor.0.weight'].shape[1]!=76:
   if not args.allow_physics_migration or not args.reset_optimizer or c.get('contract',{}).get('controlSemantics')!='joint-servo-adduction-v7':raise ValueError('V8 extension requires the exact v7 contract, explicit migration and fresh optimizer')
   agent.migrate_from_v7(c['model'])
  else:agent.load_state_dict(c['model'])
  restore_optimizer(opt,c,agent,args.reset_optimizer)
  if c['model']['logstd'].ndim==1:
   agent.logstd.data[2,1]=-.35;agent.logstd.data[4,0]=-.5
  if args.phase_exploration and c['model']['logstd'].ndim==2:
   with torch.no_grad():agent.logstd[:,0,:3]=math.log(args.board_leg_initial_std)
   provenance['explorationMigration']={'parentShape':list(c['model']['logstd'].shape),'newShape':list(agent.logstd.shape),'description':'The old per-goal Gaussian variance is duplicated. Only loaded-foot-contact hip, knee and ankle exploration is increased. The deterministic actor and flight distribution are unchanged at migration; optimizer is fresh.','initialGroundedLegStd':args.board_leg_initial_std}
  parent_contract=c.get('contract')
  try:
   validate_contract(parent_contract,contract);contract['stepsOnContract']=parent_contract.get('stepsOnContract',0)
   contract['actorUpdateInteractions']=parent_contract.get('actorUpdateInteractions',contract['stepsOnContract']);contract['criticCalibrationInteractions']=parent_contract.get('criticCalibrationInteractions',0);contract['criticCalibrationRetainedSamples']=parent_contract.get('criticCalibrationRetainedSamples',0)
  except ValueError as error:
   if not args.allow_physics_migration:raise ValueError(str(error)+'; pass --allow-physics-migration only for a documented new training phase')
   if not args.reset_optimizer:raise ValueError('Cross-physics migration requires --reset-optimizer; old optimizer state must not be described as weights-only continuation.')
   provenance['physicsMigration']={'parentContract':parent_contract,'parentXMLSHA256':c.get('provenance',{}).get('sourceHashes',{}).get('diver.xml'),'description':('V9 surface-crossing entry score and entry-first curriculum; unchanged body, actuators and water approximation. Fresh optimizer; inherited interactions do not count as v9 experience.')}
  total=c['steps'];provenance['parentSteps']=total;provenance['parentCheckpoint']=str(Path(args.resume).resolve());provenance['parentCheckpointSHA256']=hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()
  for group in opt.param_groups:group['lr']=args.lr
  if args.minimum_std:agent.logstd.data.clamp_(min=float(np.log(args.minimum_std)))
  print('RESUME',total,flush=True)
 else:agent.save_json(out/'initial.json',0,vars(args),contract)
 if args.phase_exploration:
  if not args.board_leg_minimum_std<=args.board_leg_initial_std<=args.board_leg_maximum_std:raise ValueError('Initial grounded-leg standard deviation must lie within its bounds')
  if not args.resume:
   with torch.no_grad():agent.logstd[:,0,:3]=math.log(args.board_leg_initial_std)
  provenance['explorationSemantics']={'name':'goal-and-observed-foot-contact-v1','shape':[6,2,9],'phaseZero':'observation[58] <= 0.5 and either foot-contact flag at61/62 >0.5','phaseOne':'all other observations, including brief contact gaps','groundedLegActions':[0,1,2],'groundedLegBounds':[args.board_leg_minimum_std,args.board_leg_maximum_std],'otherMaximumStd':args.maximum_std,'likelihood':'Conditional tanh-squashed Gaussian with exact change-of-variables log probability and sampled bounded entropy.'}
 cap_exploration(agent,args.maximum_std,args.board_leg_minimum_std,args.board_leg_maximum_std)
 if args.critic_calibration_rounds and not exact:
  if args.resume and not args.reset_optimizer:raise ValueError('Actor-frozen calibration on a resumed policy requires --reset-optimizer; restored critic Adam moments would be stale')
  calibration=calibrate(env,agent,args.critic_calibration_rounds,args.critic_calibration_lr,args.critic_calibration_epochs,out/'critic-calibration.json')
  total+=calibration['trainingInteractions'];contract['stepsOnContract']+=calibration['trainingInteractions'];contract['criticCalibrationInteractions']+=calibration['trainingInteractions'];contract['criticCalibrationRetainedSamples']+=calibration['retainedMCSamples']
  provenance['criticCalibration']={k:v for k,v in calibration.items()if k!='history'}
  provenance['criticCalibration']['accounting']='All actor-frozen calibration training interactions are counted separately from PPO actor-update interactions. Retained MC samples are recorded separately; incomplete tails receive no critic gradient. Disjoint validation interactions are excluded from lineage counts.'
  torch.save({'contract':dict(contract),'actionDistribution':'tanh-squashed-gaussian-v1','model':agent.state_dict(),'optimizer':opt.state_dict(),'steps':total,'config':vars(args),'torchRng':torch.get_rng_state(),'numpyRng':np.random.get_state(),'environmentRng':env.rng.bit_generator.state,'provenance':provenance},out/'calibrated.pt')
  agent.save_json(out/'calibrated-policy.json',total,vars(args),dict(contract))
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
 previous=c.get('trainingState',{}) if args.resume else {}
 if exact:
  restore_environment(env,c['environment']);restore_rng(c['rng'])
 cumulative_before=previous.get('cumulativeElapsedSeconds',0.0)
 history=copy.deepcopy(previous.get('history',[])) if exact else []
 completed=previous.get('episodesCompleted',0) if exact else 0
 updates_before=previous.get('updatesCompleted',0) if exact else 0
 interactions=requested_interactions(args.steps,args.target_steps,total,args.envs*args.horizon)
 started_steps=total;last_save_steps=total;last_save_time=time.monotonic()
 def persist(update_count,reason):
  elapsed=time.monotonic()-segment_start
  state={'segmentStartedUTC':segment_started_utc,'segmentElapsedSeconds':elapsed,'cumulativeElapsedSeconds':cumulative_before+elapsed,'priorElapsedTimeKnown':bool(not args.resume or 'cumulativeElapsedSeconds'in previous),'updatesCompleted':updates_before+update_count,'episodesCompleted':completed,'history':history,'segmentStartSteps':started_steps,'segmentAdditionalTarget':interactions,'requestedAdditionalSteps':args.steps,'requestedCumulativeTarget':args.target_steps,'stopSignal':stop_requested['signal']}
  return save_checkpoint(out,agent,opt,env,contract,vars(args),provenance,total,state,reason)
 persist(0,'initial-boundary')
 obs=torch.from_numpy(env.observe());T=args.horizon;N=args.envs
 ob=torch.zeros(T,N,76);act=torch.zeros(T,N,9);logp=torch.zeros(T,N);rew=torch.zeros(T,N);done=torch.zeros(T,N);value=torch.zeros(T,N)
 start=time.time();updates=interactions//(N*T)
 for update in range(updates):
  episode=[]
  for t in range(T):
   ob[t]=obs
   with torch.no_grad():a,lp,_,v,z=agent.get(obs)
   act[t]=z;logp[t]=lp;value[t]=v
   nxt,r,d,info=env.step(a.numpy());rew[t]=torch.from_numpy(r);done[t]=torch.from_numpy(d.astype(np.float32));obs=torch.from_numpy(nxt);episode.extend(info)
  total+=N*T;contract['stepsOnContract']+=N*T;contract['actorUpdateInteractions']+=N*T;completed+=len(episode)
  with torch.no_grad():nextvalue=agent.critic(obs).squeeze(-1)
  adv=torch.zeros(T,N);gae=torch.zeros(N)
  for t in reversed(range(T)):
   nonterminal=1-done[t];nv=nextvalue if t==T-1 else value[t+1]
   delta=rew[t]+.995*nv*nonterminal-value[t];gae=delta+.995*.95*nonterminal*gae;adv[t]=gae
  ret=adv+value;bobs=ob.flatten(0,1);ba=act.flatten(0,1);blp=logp.flatten();badv=adv.flatten();bret=ret.flatten();bv=value.flatten()
  badv=(badv-badv.mean())/(badv.std()+1e-8);losses=[]
  for epoch in range(args.epochs):
   for ids in torch.randperm(N*T).split(args.batch):
    _,newlp,ent,newv,_=agent.get(bobs[ids],ba[ids]);ratio=(newlp-blp[ids]).exp()
    pg=torch.maximum(-badv[ids]*ratio,-badv[ids]*ratio.clamp(.8,1.2)).mean()
    vl=.5*(newv-bret[ids]).square().mean()
    # Keep raw logits from drifting into the flat region of the bounded latent
    # mean. This regularizer is separate from the squashed action likelihood.
    mean_penalty=(agent.actor(bobs[ids]).abs()-2).clamp_min(0).square().mean()
    loss=pg+.5*vl-args.entropy*ent.mean()+args.mean_penalty*mean_penalty
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(agent.parameters(),.5);opt.step();cap_exploration(agent,args.maximum_std,args.board_leg_minimum_std,args.board_leg_maximum_std);losses.append(float(loss.detach()))
  checkpoint_due=total-last_save_steps>=args.checkpoint_every_steps or time.monotonic()-last_save_time>=args.checkpoint_every_seconds
  if update%10==0 or update==updates-1 or checkpoint_due or stop_requested['signal']:
   stats={'steps':total,'episodes':completed,'seconds':round(time.time()-start,2),'segmentElapsedSeconds':time.monotonic()-segment_start,'loss':round(float(np.mean(losses)),4),'std':agent.logstd.exp().detach().numpy().round(3).tolist(),**episode_statistics(episode,args.skills)}
   history.append(stats);print(json.dumps(stats),flush=True)
   atomic_json(out/'progress.json',history)
  if checkpoint_due or update==updates-1 or stop_requested['signal']:
   persist(update+1,'interrupted' if stop_requested['signal'] else ('completed-target' if update==updates-1 else 'periodic'))
   last_save_steps=total;last_save_time=time.monotonic()
  if stop_requested['signal']:break
 atomic_json(out/'run-status.json',{'status':'interrupted-at-update-boundary' if stop_requested['signal'] else 'completed-target','steps':total,'segmentAdditionalTarget':interactions,'signal':stop_requested['signal'],'latest':'latest.json','finishedUTC':utc_now()})
 env.close()

def parse_args(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--steps','--additional-steps',dest='steps',type=int);p.add_argument('--target-steps',type=int);p.add_argument('--envs',type=int,default=128);p.add_argument('--horizon',type=int,default=160);p.add_argument('--epochs',type=int,default=4);p.add_argument('--batch',type=int,default=1024);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--entropy',type=float,default=.006);p.add_argument('--mean-penalty',type=float,default=.001);p.add_argument('--execution-weight',type=float,default=0);p.add_argument('--ascent-weight',type=float,default=8);p.add_argument('--launch-speed-weight',type=float,default=.5);p.add_argument('--entry-geometry-weight',type=float,default=0);p.add_argument('--maximum-std',type=float,default=1);p.add_argument('--seed',type=int,default=1701);p.add_argument('--skills',nargs='+',type=int,default=[0,1]);p.add_argument('--min-height',type=float,default=5);p.add_argument('--max-height',type=float,default=10);p.add_argument('--skill-min-heights',nargs=6,type=float);p.add_argument('--boundary-rate',type=float,default=.2);p.add_argument('--name',default='ppo-v1');p.add_argument('--reset-optimizer',action='store_true');p.add_argument('--minimum-std',type=float,default=0);p.add_argument('--allow-physics-migration',action='store_true');p.add_argument('--resume');p.add_argument('--critic-calibration-rounds',type=int,default=0);p.add_argument('--critic-calibration-lr',type=float,default=.001);p.add_argument('--critic-calibration-epochs',type=int,default=4);p.add_argument('--phase-exploration',action='store_true');p.add_argument('--board-leg-initial-std',type=float,default=.45);p.add_argument('--board-leg-minimum-std',type=float,default=.35);p.add_argument('--board-leg-maximum-std',type=float,default=.5)
 p.add_argument('--entry-curriculum',action='store_true');p.add_argument('--resume-mode',choices=['exact','restart'],default='exact');p.add_argument('--threads',type=int,default=8);p.add_argument('--checkpoint-every-steps',type=int,default=204800);p.add_argument('--checkpoint-every-seconds',type=float,default=300)
 argv=sys.argv[1:] if argv is None else argv
 args=p.parse_args(argv)
 if args.resume and args.resume_mode=='exact':
  parent=torch.load(args.resume,weights_only=False,map_location='cpu')
  supplied={word.split('=')[0]for word in argv if word.startswith('--')}
  override={'name','resume','resume_mode','steps','target_steps','checkpoint_every_steps','checkpoint_every_seconds','reset_optimizer','allow_physics_migration','critic_calibration_rounds','critic_calibration_lr','critic_calibration_epochs'}
  for key,value in parent.get('config',{}).items():
   if key not in override and hasattr(args,key) and '--'+key.replace('_','-')not in supplied:setattr(args,key,value)
 if args.checkpoint_every_steps<=0 or args.checkpoint_every_seconds<=0:raise ValueError('Checkpoint intervals must be positive')
 return args

if __name__=='__main__':train(parse_args())
