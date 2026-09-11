"""One policy selects a declaration, then drives real joint servos.
No trainer-selected dive targets or motion demonstrations.
"""
import copy
import numpy as np
import mujoco
from engine import Arena as Physics, initial_state, framed_angles, board_forces, ACTION_LOW, ACTION_HIGH, STATE_SPEC
from rules import DIVES,CODES,CODE_INDEX,legal_mask,validate_declaration,POSITIONS
from judge import judge,terminal_reward

EXCLUDED={'roll_state','roll_sensor','ctrl','qadr','vadr','curriculum_attempts','curriculum_successes'}

def rows(engine):
 return {k:v for k,v in vars(engine).items() if isinstance(v,np.ndarray) and v.ndim and v.shape[0]==engine.n and k not in EXCLUDED}

class Arena:
 def __init__(self,n=64,seed=1,threads=4,training=True,practice=.35):
  self.perturb=False;self.n=n;self.training=training;self.rng=np.random.default_rng(seed);self.practice_rate=practice
  self.physics=Physics(n,seed,threads,skills=[0],heights=(1,10),minimum_heights=[1]*6,training=training)
  # The normalized action that reproduces the evaluation stance; a policy whose
  # initial mean holds still is the most neutral prior, not a prescribed motion.
  _,_,stance=initial_state(self.physics.model,self.physics.resetdata,10,0)
  self.initial_action=(2*(stance-ACTION_LOW)/(ACTION_HIGH-ACTION_LOW)-1).astype(np.float32)
  self.previous_actions=np.zeros((n,9),np.float32);self.choosing=np.ones(n,bool);self.declaration=np.zeros(n,int);self.group=np.ones(n,int)
  self.apparatus=np.zeros(n,int);self.height=np.zeros(n);self.used=np.zeros((n,len(CODES)),bool)
  self.auxiliary=np.zeros(n,bool);self.round=np.zeros(n,int);self.schedule=np.zeros((n,6),int);self.routine_points=np.zeros(n)
  self.returns=np.zeros(n);self.practice=np.zeros(n,bool);self.bank=[[] for _ in range(6)]
  self.attempts=np.zeros(6);self.clean=np.zeros(6);self.interactions=0
  self.reset(np.arange(n),new_routine=True)
  self.observation_size=self.observe().shape[-1]
 def close(self):self.physics.close()
 def reset(self,ids,new_routine=False):
  for i in ids:
   if new_routine or self.round[i]>=(1 if self.auxiliary[i] else 6):
    self.apparatus[i]=int(self.rng.integers(2));self.height[i]=float(self.rng.choice([5,7.5,10]) if self.apparatus[i] else self.rng.choice([1,3]))
    groups=[1,2,3,4,5,6] if self.apparatus[i] else [1,2,3,4,5,int(self.rng.integers(1,6))]
    self.schedule[i]=self.rng.permutation(groups);self.auxiliary[i]=self.training and self.rng.random()<.25
    if self.auxiliary[i]:
     weights=.2+1-self.clean/np.maximum(1,self.attempts);weights*=np.array([1,1,1,1,1,self.apparatus[i]])
     self.schedule[i,0]=int(self.rng.choice(np.arange(1,7),p=weights/weights.sum()))
    self.round[i]=0;self.used[i]=False;self.routine_points[i]=0
   self.group[i]=self.schedule[i,self.round[i]];self.choosing[i]=True;self.returns[i]=0;self.practice[i]=False;self.previous_actions[i]=0
   self.physics.reset([i],[dict(skill=0,height=self.height[i],preload=0 if self.apparatus[i] else -.05123,disturbance=0,platform=bool(self.apparatus[i]))])
   self.physics.platform[i]=bool(self.apparatus[i]);self.physics.armstand[i]=False
  return self.observe()
 def mask(self):
  return np.stack([legal_mask(int(self.group[i]),'platform' if self.apparatus[i] else 'springboard',self.height[i],np.asarray(CODES)[self.used[i]]) for i in range(self.n)])
 def observe(self):
  e=self.physics;q=e.state[:,1:1+e.model.nq];v=e.state[:,1+e.model.nq:];s=e.sensors
  # No ballistic future estimate, desired angular speed, or externally assigned
  # maneuver. Intent is visible only after the policy has committed to it.
  intent=np.zeros((self.n,9))
  for i in np.flatnonzero(~self.choosing):
   d=DIVES[self.declaration[i]];intent[i,:5]=[d['sign']*d['turns']/5,d['twists']/5,d['back'],d['armstand'],d['direction']/4];intent[i,5+POSITIONS.index(d['position'])]=1
  # Foot contact flags and the board spring state are physical sensors the
  # athlete has (pressure under the feet, the board moving); balance and the
  # press need them. Nothing here is a future estimate or an assigned target.
  support=np.concatenate([(board_forces(s)[:,[11,14]]>15).astype(float),q[:,0:1]*4,v[:,0:1]/3],axis=1)
  return np.concatenate([q[:,4:8],v[:,1:7]/10,q[:,e.qadr]/3,v[:,e.vadr]/15,e.targets/3,
   s[:,:3]/[4,2,10],s[:,3:6]/10,np.stack([e.phase_theta/(2*np.pi),e.air_twist/(2*np.pi),e.height/10,e.state[:,0]/4,e.released,e.platform,e.armstand,e.water_fraction,e.above_water/10,self.choosing,self.round/6],axis=1),
   support,np.eye(6)[self.group-1],intent,self.used.astype(float),self.previous_actions],axis=1).clip(-10,10).astype(np.float32)
 def declare(self,i,choice):
  apparatus='platform' if self.apparatus[i] else 'springboard'
  d=validate_declaration(choice,int(self.group[i]),apparatus,self.height[i],np.asarray(CODES)[self.used[i]])
  e=self.physics;self.declaration[i]=choice;e.goals[i]=[d['sign']*d['turns'],d['twists'],d['back'],{'C':0,'B':1,'D':2,'A':3}[d['position']]]
  # Legal starting orientation follows the agent's chosen dive. There is no
  # modification to root state after launch.
  e.reset([i],[dict(skill=3 if d['back'] else 0,height=self.height[i],preload=0 if self.apparatus[i] else -.05123,disturbance=0,platform=bool(self.apparatus[i]))])
  e.platform[i]=bool(self.apparatus[i]);e.armstand[i]=d['armstand'];e.headfirst[i]=d['headfirst'];e.height[i]=self.height[i]
  if d['armstand']:
   # Balanced handstand start: hands shoulder-width on the platform, arms straight,
   # trunk pitched so the centre of mass sits over the hands. Holding it and pressing
   # off are learned; the shoulders only carry a bounded servo torque.
   data=e.resetdata;m=e.model;mujoco.mj_setState(m,data,e.state[i],STATE_SPEC)
   yaw=np.pi*d['back'];target=np.array([0,0,1.2,3.14,3.14,0,0,0,.06]);gids=[6,9]
   from engine import ACTUATOR_MAP
   def place(pitch):
    data.qpos[4:8]=[np.cos(pitch/2)*np.cos(yaw/2),np.sin(pitch/2)*np.sin(yaw/2),np.sin(pitch/2)*np.cos(yaw/2),np.cos(pitch/2)*np.sin(yaw/2)]
    data.qpos[e.qadr]=target[ACTUATOR_MAP];data.ctrl[:]=target[ACTUATOR_MAP];data.qpos[0]=0;data.qpos[1]=0;data.qpos[3]=0;mujoco.mj_forward(m,data)
    bottom=min(data.geom_xpos[g,2]-np.linalg.norm(data.geom_xmat[g].reshape(3,3)[2]*m.geom_size[g]) for g in gids)
    data.qpos[3]+=.055-bottom-.0001;data.qpos[1]+=-.2-np.mean(data.geom_xpos[gids,0]);mujoco.mj_forward(m,data)
    return float(data.subtree_com[2][0]-np.mean(data.geom_xpos[gids,0]))
   lo,hi=np.pi-.35,np.pi+.35;offset_lo=place(lo)
   for _ in range(40):
    mid=(lo+hi)/2;offset=place(mid)
    if np.sign(offset)==np.sign(offset_lo):lo,offset_lo=mid,offset
    else:hi=mid
   if abs(place((lo+hi)/2))>2e-3:raise ValueError('Armstand start could not be balanced over the hands')
   mujoco.mj_getState(m,data,e.state[i],STATE_SPEC);e.sensors[i]=data.sensordata;e.targets[i]=target
   e.prev_pitch[i],e.prev_twist[i],_= [x[0] for x in framed_angles(e.state[i,5:9][None])]
  if self.perturb:e.disturbance[i]=30 if i%2 else -30;e.disturbance_time[i]=.75
  self.choosing[i]=False
  # Reachable recovery snapshots originate only from this learner's real dives.
  # Reuse the same self-declaration, apparatus and height, never assign a target.
  g=self.group[i]-1;rate=self.clean[g]/max(1,self.attempts[g]);prob=self.practice_rate*(1-rate)
  eligible=[x for x in self.bank[g] if x['declaration']==choice and x['height']==self.height[i] and x['apparatus']==self.apparatus[i]]
  if self.training and eligible and self.rng.random()<prob:
   snap=eligible[int(self.rng.integers(len(eligible)))];
   for k,x in snap['physics'].items():getattr(e,k)[i]=x
   self.practice[i]=True
 def step(self,choices,actions):
  e=self.physics;chosen=np.flatnonzero(self.choosing)
  # Declare before any integration. Masked actions during declaration do not
  # contribute motor likelihoods or move the body.
  for i in chosen:self.declare(i,int(choices[i]))
  frozen={k:v[chosen].copy() for k,v in rows(e).items()}
  _,r,done,info=e.step(actions,auto_reset=False)
  self.previous_actions[:]=actions;self.previous_actions[chosen]=0
  for k,value in frozen.items():getattr(e,k)[chosen]=value
  r[chosen]=0;done[chosen]=False;info=[x for x in info if x['index'] not in chosen]
  self.interactions+=self.n
  for i in range(self.n):
   if self.training and i not in chosen and not self.practice[i] and not e.banked[i] and e.released[i] and not np.isfinite(e.entry_time[i]) and e.sensors[i,5]<0 and .3<e.above_water[i]<3:
    snap=dict(declaration=int(self.declaration[i]),height=float(self.height[i]),apparatus=int(self.apparatus[i]),physics={k:v[i].copy() for k,v in rows(e).items()})
    bank=self.bank[self.group[i]-1];bank.append(snap)
    if len(bank)>96:bank.pop(0)
    e.banked[i]=True
  completed=[]
  for m in info:
   i=m['index'];d=DIVES[self.declaration[i]];m['positionQuality']=m['positionQualities'][POSITIONS.index(d['position'])]
   score=judge(self.declaration[i],'platform' if self.apparatus[i] else 'springboard',self.height[i],m)
   r[i]+=terminal_reward(score,m);g=self.group[i]-1
   if not self.practice[i]:self.attempts[g]+=1;self.clean[g]+=score['clean']
   self.routine_points[i]+=score['points'];self.used[i,CODE_INDEX[d['code']]]=True;self.round[i]+=1
   completed.append(dict(**score,measurements=m,practice=bool(self.practice[i]),height=float(self.height[i]),apparatus='platform' if self.apparatus[i] else 'springboard',routinePoints=float(self.routine_points[i]),routineFinished=bool(self.round[i]==6 and not self.auxiliary[i]),auxiliaryPractice=bool(self.auxiliary[i]),index=i))
  self.returns+=r
  for record in completed:record['return']=float(self.returns[record['index']])
  if done.any():self.reset(np.flatnonzero(done))
  return self.observe(),r,done,completed
 def state_dict(self):
  return dict(physics={k:v.copy() for k,v in rows(self.physics).items()},arrays={k:v.copy() for k,v in vars(self).items() if isinstance(v,np.ndarray)},bank=copy.deepcopy(self.bank),rng=self.rng.bit_generator.state,physicsRng=self.physics.rng.bit_generator.state,interactions=self.interactions)
 def load_state_dict(self,state):
  for k,v in state['physics'].items():getattr(self.physics,k)[:]=v
  for k,v in state['arrays'].items():getattr(self,k)[:]=v
  self.bank=copy.deepcopy(state['bank']);self.rng.bit_generator.state=state['rng'];self.physics.rng.bit_generator.state=state['physicsRng'];self.interactions=state['interactions']
