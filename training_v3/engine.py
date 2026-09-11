"""Original articulated MuJoCo diving environment; native and browser share diver.xml.
Units: kg, m, s, rad. Native coordinates X forward, Y lateral, Z up; water Z=-height.
The only controller commands are bounded joint position servos. No root wrench is
used by the controller. Board spring contact transfers takeoff momentum physically.
"""
from pathlib import Path
import numpy as np
import mujoco

from mujoco.rollout import Rollout
from water import body_water, immersion

ROOT=Path(__file__).resolve().parents[1]
DT=.02
SUBSTEPS=10
SKILLS=[
 {'id':'101C','code':101,'name':'Forward dive · tuck','turns':.5,'twists':0,'back':0,'shape':0,'height':3},
 {'id':'103C','code':103,'name':'Forward 1½ somersaults · tuck','turns':1.5,'twists':0,'back':0,'shape':0,'height':5},
 {'id':'103B','code':103,'name':'Forward 1½ somersaults · pike','turns':1.5,'twists':0,'back':0,'shape':1,'height':7.5},
 {'id':'201C','code':201,'name':'Back dive · tuck','turns':.5,'twists':0,'back':1,'shape':0,'height':3},
 {'id':'203C','code':203,'name':'Back 1½ somersaults · tuck','turns':1.5,'twists':0,'back':1,'shape':0,'height':5},
 {'id':'5132D','code':5132,'name':'Forward 1½ somersaults · 1 twist','turns':1.5,'twists':1,'back':0,'shape':2,'height':7.5},
]
# Neural actions: common hip/knee/ankle, left/right shoulder pitch, left/right
# shoulder roll, common elbow. Left/right legs stay synchronized, not animated.
ACTION_LOW=np.array([-.45,0,-.55,-.5,-.5,-1.1,-1.1,0,-.15])
ACTION_HIGH=np.array([2.2,2.6,1.35,3.14,3.14,1.1,1.1,2.3,.18])
ACTUATOR_MAP=np.array([0,1,2,0,1,2,3,5,7,4,6,7,8,8])
TARGET_RATE=np.array([8,10,8,10,10,8,8,10,4]) # rad/s servo target slew, actual dynamics uncapped
STATE_SPEC=mujoco.mjtState.mjSTATE_FULLPHYSICS

def quat_up(q):
 w,x,y,z=q.T
 return np.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],axis=1)

def quat_forward(q):
 w,x,y,z=q.T
 return np.stack([1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)],axis=1)

def framed_angles(q):
 """Forward-plane swing and axial twist. Invalid near lateral-axis singularity.
 Sideways tumbling is separately rejected; it cannot earn a forward/back code.
 """
 up=quat_up(q);forward=quat_forward(q)
 side=np.tile([0.,1.,0.],(len(q),1))-up*up[:,1,None]
 side/=np.maximum(np.linalg.norm(side,axis=1,keepdims=True),1e-8)
 ref=np.cross(side,up)
 return np.arctan2(up[:,0],up[:,2]),np.arctan2(np.sum(forward*side,axis=1),np.sum(forward*ref,axis=1)),np.abs(up[:,1])

def board_forces(s):
 return np.linalg.norm(s[...,150:210].reshape(*s.shape[:-1],15,4)[...,1:],axis=-1)

def stand_forces(s):
 return np.linalg.norm(s[...,210:270].reshape(*s.shape[:-1],15,4)[...,1:],axis=-1)

def tuck_geometry(s):
 hands=s[...,22:28].reshape(*s.shape[:-1],2,3)
 knees=s[...,37:43].reshape(*s.shape[:-1],2,3)
 shins=np.stack([s[...,115:118],s[...,136:139]],axis=-2)
 ankles=2*shins-knees
 axis=ankles-knees;fraction=np.clip(np.sum((hands-knees)*axis,axis=-1)/np.maximum(np.sum(axis*axis,axis=-1),1e-9),0,1)
 distance=np.linalg.norm(hands-knees-fraction[...,None]*axis,axis=-1)
 separation=np.maximum(np.linalg.norm(knees[...,0,:]-knees[...,1,:],axis=-1),np.linalg.norm(ankles[...,0,:]-ankles[...,1,:],axis=-1))
 return distance,separation

def leg_geometry(s):
 knees=s[...,37:43].reshape(*s.shape[:-1],2,3);shins=np.stack([s[...,115:118],s[...,136:139]],axis=-2);ankles=2*shins-knees;toes=s[...,16:22].reshape(*s.shape[:-1],2,3)
 w,x,y,z=np.moveaxis(s[...,9:13],-1,0);side=np.stack([2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)],axis=-1)
 differences=np.stack([p[...,0,:]-p[...,1,:]for p in [knees,ankles,toes]],axis=-2)
 gaps=np.linalg.norm(differences,axis=-1);signed=np.sum(differences*side[...,None,:],axis=-1)
 return dict(kneeGap=gaps[...,0],ankleGap=gaps[...,1],toeGap=gaps[...,2],crossedLegs=np.any(signed<=.02,axis=-1),signedLegGaps=signed)

def entry_geometry(s):
 """Actual first-contact foot lines and hand-tip axes; radians and meters."""
 hands=s[...,22:28].reshape(*s.shape[:-1],2,3);knees=s[...,37:43].reshape(*s.shape[:-1],2,3)
 shins=np.stack([s[...,115:118],s[...,136:139]],axis=-2);ankles=2*shins-knees
 toes=s[...,16:22].reshape(*s.shape[:-1],2,3);shin=ankles-knees;toe=toes-ankles
 length=np.linalg.norm(shin,axis=-1)*np.linalg.norm(toe,axis=-1)
 foot=np.where(length>1e-9,np.arccos(np.clip(np.sum(shin*toe,axis=-1)/np.maximum(length,1e-9),-1,1)),np.pi)
 hand_q=np.stack([s[...,83:87],s[...,104:108]],axis=-2)
 hand_up_z=1-2*(hand_q[...,1]**2+hand_q[...,2]**2)
 hand=np.where(np.abs(np.linalg.norm(hand_q,axis=-1)-1)<1e-3,np.arccos(np.clip(hand_up_z,-1,1)),np.pi)
 return dict(**leg_geometry(s),footLineAngles=foot,handAxisAngles=hand,handSeparation=np.linalg.norm(hands[...,0,:]-hands[...,1,:],axis=-1),handHeightGap=np.abs(hands[...,0,2]-hands[...,1,2]))

def entry_posture(joint,geometry):
 hips=joint[..., [0,3]];knees=joint[..., [1,4]];pitch=joint[..., [6,9]]-3.05;roll=joint[..., [7,10]]-np.array([-.3,.3]);elbows=joint[..., [8,11]]
 foot=geometry['footLineAngles'];hand=geometry['handAxisAngles'];gap=geometry['handSeparation'];vertical=geometry['handHeightGap']
 loss=1.25*np.sum(hips**2+knees**2,axis=-1)+.325*np.sum(pitch**2,axis=-1)+.65*np.sum(roll**2,axis=-1)+.5*np.sum(elbows**2,axis=-1)+.55*np.sum(foot**2,axis=-1)+1.5*np.sum(np.maximum(hand-np.pi/12,0)**2,axis=-1)+4*np.maximum(gap-.07,0)**2+20*vertical**2
 limbs=(np.max(np.abs(hips),axis=-1)<.45)&(np.max(np.abs(knees),axis=-1)<.5)&(np.max(np.abs(pitch),axis=-1)<.6)&(np.max(np.abs(roll),axis=-1)<.5)&(np.max(np.abs(elbows),axis=-1)<.5)
 loss+=12*(np.maximum(geometry['kneeGap']-.17,0)**2+np.maximum(geometry['ankleGap']-.13,0)**2+np.maximum(geometry['toeGap']-.11,0)**2)+4*geometry['crossedLegs']
 limits=(np.max(foot,axis=-1)<np.radians(20))&(np.max(hand,axis=-1)<np.pi/6)&(gap<.12)&(vertical<.04)&(geometry['kneeGap']<.19)&(geometry['ankleGap']<.16)&(geometry['toeGap']<.15)&(~geometry['crossedLegs'])
 return np.exp(-loss),limbs,limits,loss

def entry_geometry_error(geometry):
 """Training-only continuous distance from clean physical entry geometry.
 This never changes the independent judge or supplies desired joint commands.
 """
 error=(np.maximum(geometry['handSeparation']-.08,0)/.08)**2+(geometry['handHeightGap']/.04)**2
 error+=.25*np.sum((np.maximum(geometry['handAxisAngles']-np.pi/12,0)/(np.pi/6))**2,axis=-1)
 error+=np.sum((np.maximum(geometry['footLineAngles']-np.pi/18,0)/(np.pi/9))**2,axis=-1)
 error+=(np.maximum(geometry['ankleGap']-.13,0)/.1)**2
 return np.clip(error,0,4)

def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi

def decoded_action(a):return ACTION_LOW+(np.clip(a,-1,1)+1)*.5*(ACTION_HIGH-ACTION_LOW)

def execution_estimate(angle,form,splash,bounces,board_twist,numeric,position,entry,invalid_launch,ascent):
 """The displayed judge, vectorized for reward. Failed declared dives cannot
 collect a clean-entry bonus; difficulty never appears in execution.
 """
 deductions=1.5*(1-np.clip(ascent/.3,0,1))+np.minimum(2,bounces)+np.clip((1-form)*3,0,3)+np.clip(angle/10,0,4)+1.5*splash+np.where(board_twist>.26,np.clip(board_twist,.5,2),0)
 value=np.clip(10-deductions,0,10)
 value=np.where(position&entry,value,np.minimum(value,2))
 value=np.where(numeric&(~invalid_launch),value,0)
 return np.floor(value*2+.5)/2

def initial_state(m,d,height,skill,tilt=.16,preload=-.18,hip=.5,knee=1.,ankle=.5,x=-.13):
 mujoco.mj_resetData(m,d)
 d.mocap_pos[0]=[0,0,-height];d.mocap_quat[0]=[1,0,0,0]
 back=SKILLS[skill]['back']; yaw=np.pi*back
 # Tilt toward water, independent of whether the athlete faces it or faces away.
 d.qpos[4:8]=[np.cos(tilt/2)*np.cos(yaw/2),np.sin(tilt/2)*np.sin(yaw/2),np.sin(tilt/2)*np.cos(yaw/2),np.cos(tilt/2)*np.sin(yaw/2)]
 d.qpos[0]=preload
 targets=np.array([hip,knee,ankle,.3,.3,0,0,0,0])
 qadr=m.jnt_qposadr[m.actuator_trnid[:,0]]
 d.qpos[qadr]=targets[ACTUATOR_MAP];d.ctrl[:]=targets[ACTUATOR_MAP]
 mujoco.mj_forward(m,d)
 # Exact foot-box support plane, including orientation; start in resting contact.
 fg=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in ['foot_L','foot_R']]
 gids=[int(m.body_geomadr[b]) for b in fg]
 bottom=min(d.geom_xpos[g,2]-np.dot(np.abs(d.geom_xmat[g].reshape(3,3)[2]),m.geom_size[g]) for g in gids)
 d.qpos[3]+=preload+.055-bottom-.0001
 d.qpos[1]+=x-np.mean(d.geom_xpos[gids,0])
 mujoco.mj_forward(m,d)
 state=np.empty(mujoco.mj_stateSize(m,STATE_SPEC));mujoco.mj_getState(m,d,state,STATE_SPEC)
 return state,d.sensordata.copy(),targets



class Arena:
 def __init__(self,n=128,seed=123,threads=8,skills=None,heights=(3,10),training=True,minimum_heights=None,boundary_rate=0,execution_weight=0,ascent_weight=8,launch_speed_weight=.5,entry_geometry_weight=0,entry_curriculum=False):
  self.entry_curriculum=entry_curriculum;self.entry_bank=[[] for _ in SKILLS];self.curriculum_attempts=np.zeros(6,int);self.curriculum_successes=np.zeros(6,int)
  self.goals=np.tile([.5,0,0,0.],(n,1));self.headfirst=np.ones(n,bool);self.armstand=np.zeros(n,bool);self.platform=np.zeros(n,bool);self.position_sums=np.zeros((n,4));self.position_ticks=np.zeros(n);self.entry_omega=np.zeros(n)
  self.n=n;self.rng=np.random.default_rng(seed);self.model=mujoco.MjModel.from_xml_path(str(ROOT/'training_v3/diver.xml'))
  self.pool=Rollout(nthread=min(threads,n));self.datas=[mujoco.MjData(self.model) for _ in range(min(threads,n))];self.resetdata=mujoco.MjData(self.model)
  self.models=[self.model]*n;self.ns=mujoco.mj_stateSize(self.model,STATE_SPEC)
  self.state=np.zeros((n,self.ns));self.sensors=np.zeros((n,self.model.nsensordata));self.targets=np.zeros((n,9))
  self.minimum_heights=minimum_heights or [s["height"] for s in SKILLS];self.boundary_rate=boundary_rate;self.execution_weight=execution_weight
  self.ascent_weight=ascent_weight;self.launch_speed_weight=launch_speed_weight;self.entry_geometry_weight=entry_geometry_weight
  self.skill_ids=list(range(len(SKILLS))) if skills is None else skills;self.heights=heights;self.training=training
  self.disturbance=np.zeros(n);self.disturbance_time=np.zeros(n);self.height=np.zeros(n);self.skill=np.zeros(n,dtype=int);self.theta=np.zeros(n);self.twist=np.zeros(n);self.prev_pitch=np.zeros(n);self.prev_twist=np.zeros(n);self.max_lateral=np.zeros(n)
  self.departure_com=np.zeros(n);self.apex_com=np.zeros(n);self.takeoff_vertical_speed=np.zeros(n)
  self.phase_theta=np.zeros(n);self.departure_pitch=np.zeros(n);self.last_foot_contact=np.zeros((n,2));self.foot_departure_gap=np.zeros(n);self.takeoff_tilt_invalid=np.zeros(n,bool);self.air_clear_time=np.zeros(n);self.departure_theta=np.zeros(n);self.departure_twist=np.zeros(n);self.air_theta=np.zeros(n);self.air_twist=np.zeros(n);self.release_theta=np.zeros(n);self.board_invalid=np.zeros(n,bool);self.board_impulse=np.zeros(n);self.board_peak=np.zeros(n);self.preparation_bounces=np.zeros(n,dtype=int);self.takeoff_value=np.zeros(n);self.stand_impulse=np.zeros(n);self.stand_peak=np.zeros(n);self.foot_side_contact=np.zeros(n,bool);self.rotated_recontact=np.zeros(n,bool)
  self.released=np.zeros(n,bool);self.release_time=np.zeros(n);self.release_twist=np.zeros(n);self.max_air=np.zeros(n);self.shape_duration=np.zeros(n);self.shape_peak=np.zeros(n);self.shape_quality=np.zeros(n);self.shape_ticks=np.zeros(n);self.energy=np.zeros(n)
  self.applied_impulse=np.zeros(n);self.push_duration=np.zeros(n);self.first_geometry=np.zeros(n,dtype=int);self.shape_angle=np.zeros(n);self.max_progress=np.zeros(n);self.board_twist=np.zeros(n)
  self.entry_time=np.full(n,np.nan);self.full_entry_time=np.full(n,np.nan);self.first_state=np.zeros((n,self.ns));self.first_sensors=np.zeros((n,self.model.nsensordata));self.entry_max_angle=np.zeros(n);self.entry_min_form=np.ones(n);self.entry_limbs_valid=np.ones(n,bool);self.entry_geometry_valid=np.ones(n,bool);self.entry_samples=np.zeros(n,int);self.entry_worst_geometry=np.zeros((n,9));self.entry_crossed=np.zeros(n,bool);self.above_water=np.zeros(n);self.water_fraction=np.zeros(n)
  self.returns=np.zeros(n);self.max_torque=np.zeros(n);self.max_joint_speed=np.zeros(n)
  self.roll_state=np.empty((n,SUBSTEPS,self.ns));self.roll_sensor=np.empty((n,SUBSTEPS,self.model.nsensordata));self.ctrl=np.zeros((n,SUBSTEPS,self.model.nu+self.model.nbody*6+8))
  self.qadr=self.model.jnt_qposadr[self.model.actuator_trnid[:,0]];self.vadr=self.model.jnt_dofadr[self.model.actuator_trnid[:,0]]
  self.practice=np.zeros(n,bool);self.banked=np.zeros(n,bool)
  self.surface_previous=np.zeros((n,15));self.surface_seen=np.zeros((n,15),bool);self.surface_finished=np.zeros((n,15),bool);self.surface_group_loss=np.zeros((n,6));self.surface_lateral=np.zeros(n)
  self.reset(np.arange(n))
 def close(self):self.pool.close()
 def reset(self,ids,contexts=None):
  for j,i in enumerate(ids):
   s=int(self.rng.choice(self.skill_ids));lo=max(self.heights[0],self.minimum_heights[s]);h=self.rng.uniform(lo,self.heights[1])
   if self.training and self.rng.random()<self.boundary_rate:h=float(self.rng.choice([lo,self.heights[1]]))
   c={} if contexts is None else contexts[j];s=c.get('skill',s);h=c.get('height',h)
   self.skill[i]=s;self.height[i]=h
   self.disturbance[i]=c.get("disturbance",0);self.disturbance_time[i]=c.get("disturbanceTime",self.rng.uniform(.65,1.05))
   initial,sens,target=initial_state(self.model,self.resetdata,h,s,tilt=c.get('tilt',self.rng.uniform(.10,.22) if self.training else .16),preload=c.get('preload',self.rng.uniform(-.20,-.16) if self.training else -.18),x=c.get('x',self.rng.uniform(-.16,-.10) if self.training else -.13))
   self.state[i]=initial;self.sensors[i]=sens;self.targets[i]=target
   self.theta[i]=0;self.twist[i]=0;self.prev_pitch[i]=np.arctan2(quat_up(initial[5:9][None])[0,0],quat_up(initial[5:9][None])[0,2])
   self.applied_impulse[i]=0;self.push_duration[i]=0;self.first_geometry[i]=0;self.shape_angle[i]=0;self.max_progress[i]=0;self.board_twist[i]=0
   self.prev_twist[i]=framed_angles(initial[5:9][None])[1][0];self.max_lateral[i]=0;self.shape_duration[i]=0;self.shape_peak[i]=0
   self.departure_com[i]=0;self.apex_com[i]=0;self.takeoff_vertical_speed[i]=0
   self.phase_theta[i]=0;self.departure_pitch[i]=0;self.last_foot_contact[i]=0;self.foot_departure_gap[i]=0;self.takeoff_tilt_invalid[i]=False;self.air_clear_time[i]=0;self.departure_theta[i]=0;self.departure_twist[i]=0;self.air_theta[i]=0;self.air_twist[i]=0;self.release_theta[i]=0;self.board_invalid[i]=False;self.board_impulse[i]=0;self.board_peak[i]=0;self.preparation_bounces[i]=0;self.takeoff_value[i]=0;self.stand_impulse[i]=0;self.stand_peak[i]=0;self.foot_side_contact[i]=False;self.rotated_recontact[i]=False
   self.entry_time[i]=np.nan;self.full_entry_time[i]=np.nan;self.entry_max_angle[i]=0;self.entry_min_form[i]=1;self.entry_limbs_valid[i]=True;self.entry_geometry_valid[i]=True;self.entry_samples[i]=0;self.entry_worst_geometry[i]=0;self.entry_crossed[i]=False
   self.practice[i]=False;self.banked[i]=False;self.surface_previous[i]=0;self.surface_seen[i]=False;self.surface_finished[i]=False;self.surface_group_loss[i]=0;self.surface_lateral[i]=0
   self.released[i]=False;self.release_time[i]=0;self.release_twist[i]=0;self.max_air[i]=0;self.shape_quality[i]=0;self.shape_ticks[i]=0;self.energy[i]=0;self.returns[i]=0;self.max_torque[i]=0;self.max_joint_speed[i]=0;self.position_sums[i]=0;self.position_ticks[i]=0;self.entry_omega[i]=0
   if self.entry_curriculum and contexts is None and self.entry_bank[s]:
    rate=self.curriculum_successes[s]/max(1,self.curriculum_attempts[s])
    probability=.75 if self.curriculum_attempts[s]<128 or rate<.5 else .5 if rate<.7 else .25 if rate<.85 else .1
    if self.rng.random()<probability:
     snapshot=self.entry_bank[s][int(self.rng.integers(len(self.entry_bank[s])))]
     for key,value in snapshot.items():getattr(self,key)[i]=value
     self.practice[i]=True;self.banked[i]=True;self.returns[i]=0
  self.water_measurements()
  return self.observe()
 def goal(self):
  return self.goals
 def observe(self):
  q=self.state[:,1:1+self.model.nq];v=self.state[:,1+self.model.nq:];s=self.sensors;g=self.goal();up=quat_up(q[:,4:8])
  # Remaining ballistic time is an observable estimate, never an action planner.
  distance=np.maximum(s[:,2]+self.height-.7,0);vz=s[:,5]
  tgo=np.clip((vz+np.sqrt(vz*vz+19.62*distance))/9.81,.05,3)
  obs=np.concatenate([q[:,4:8],v[:,1:7]/10,q[:,self.qadr[:12]]/3,v[:,self.vadr[:12]]/15,
    self.targets[:,:8]/3,s[:,:3]/np.array([4,2,10]),s[:,3:6]/10,s[:,6:9]/70,
    np.stack([self.phase_theta/(2*np.pi),self.air_twist/(2*np.pi),(g[:,0]*2*np.pi-self.phase_theta)/(2*np.pi),(g[:,1]*2*np.pi-self.air_twist)/(2*np.pi),tgo/2,self.height/10,self.state[:,0]/3,self.released.astype(float),q[:,0]*4,v[:,0]/3],axis=1),
    (board_forces(s)[:,[11,14]]>15).astype(float),g/np.array([2.5,2,1,2]),q[:,self.qadr[12:]]/.25,v[:,self.vadr[12:]]/4,self.targets[:,8,None]/.25,np.stack([np.isfinite(self.entry_time),np.where(np.isfinite(self.entry_time),(self.state[:,0]-self.entry_time)/.8,0),self.above_water/2,self.water_fraction],axis=1)],axis=1)
  return np.clip(obs,-8,8).astype(np.float32)
 def water_measurements(self):
  geo=self.sensors[:,45:150].reshape(self.n,15,7)
  fraction,extent,_=immersion(self.model.geom_type[1:16],self.model.geom_size[1:16],geo[:,:,:3],geo[:,:,3:],-self.height[:,None])
  self.above_water=np.max(geo[:,:,2]+extent+self.height[:,None],axis=1)
  self.water_fraction=np.sum(fraction*self.model.body_mass[self.model.geom_bodyid[1:16]],axis=1)/70
 def water_rollout(self,ids):
  # Rollout's sensor output is pre-integration. A second disposable substep
  # obtains the *forward-updated* sensors of the first state. Only its first
  # physical transition is retained. This avoids stale velocity feedback.
  m=self.model;count=len(ids);state=self.state[ids].copy();sensor=self.sensors[ids].copy()
  states=np.empty((count,SUBSTEPS,self.ns));sensors=np.empty((count,SUBSTEPS,m.nsensordata));after=np.empty_like(sensors)
  control=self.ctrl[ids,:2].copy();body=m.geom_bodyid[1:16]
  for k in range(SUBSTEPS):
   geo=sensor[:,45:150].reshape(count,15,7);vel=sensor[:,270:360].reshape(count,15,6)
   result=body_water(m.geom_type[1:16],m.geom_size[1:16],geo[:,:,:3],geo[:,:,3:],geo[:,:,:3],vel[:,:,:3],vel[:,:,3:],m.body_mass[body],-self.height[ids,None])
   force=np.zeros((count,m.nbody,6));force[:,body,:3]=result['force'];force[:,body,3:]=result['torque']
   control[:,:,m.nu:m.nu+m.nbody*6]=force.reshape(count,1,m.nbody*6)
   st,se=self.pool.rollout([m]*count,self.datas,state,control,skip_checks=True,control_spec=3904,nstep=2,state=np.empty((count,2,self.ns)),sensordata=np.empty((count,2,m.nsensordata)))
   states[:,k]=st[:,0];sensors[:,k]=se[:,0];after[:,k]=se[:,1]
   state=st[:,0].copy();sensor=se[:,1].copy()
  return states,sensors,after
 def track_entry_batch(self,ids,state,sensor):
  if len(ids)==0:return
  geo=sensor[:,45:150].reshape(len(ids),15,7)
  fraction,extent,_=immersion(self.model.geom_type[1:16],self.model.geom_size[1:16],geo[:,:,:3],geo[:,:,3:],-self.height[ids,None])
  # Score each anatomical group while it crosses the surface. Once a group
  # is fully submerged, later movement cannot worsen that group's entry.
  active=(fraction>0)&~self.surface_finished[ids]
  geometry=entry_geometry(sensor)
  hands=np.any(active[:,[5,8]],axis=1);legs=np.any(active[:,[9,10,11,12,13,14]],axis=1)
  trunk=np.any(active[:,[0,1,2]],axis=1)
  feet=active[:,[11,14]];hand_parts=active[:,[5,8]]
  shins=active[:,[10,13]];thighs=active[:,[9,12]]
  shoulders=active[:,[3,6]];forearms=active[:,[4,7]]
  knee_cross=np.any(thighs|shins,axis=1);ankle_cross=np.any(shins|feet,axis=1);toe_cross=np.any(feet,axis=1)
  metrics=np.concatenate([geometry['footLineAngles'],geometry['handAxisAngles'],np.stack([geometry[key]for key in ['handSeparation','handHeightGap','kneeGap','ankleGap','toeGap']],axis=1)],axis=1)
  mask=np.concatenate([feet,hand_parts,np.stack([hands,hands,knee_cross,ankle_cross,toe_cross],axis=1)],axis=1)
  self.entry_worst_geometry[ids]=np.maximum(self.entry_worst_geometry[ids],np.where(mask,metrics,0))
  self.entry_crossed[ids]|=geometry['crossedLegs']&legs
  head=self.headfirst[ids]
  angle=np.degrees(np.arccos(np.clip(np.where(head,-1,1)*quat_up(state[:,5:9])[:,2],-1,1)))
  self.entry_max_angle[ids]=np.maximum(self.entry_max_angle[ids],np.where(trunk|legs,angle,0))
  joint=state[:,1+self.qadr]
  hips=joint[:,[0,3]];knees=joint[:,[1,4]]
  pitch=joint[:,[6,9]]-np.where(head,3.05,0)[:,None];roll=joint[:,[7,10]]-head[:,None]*np.array([-.3,.3]);elbows=joint[:,[8,11]]
  losses=np.stack([
   np.sum(1.25*hips**2*thighs,axis=1),
   np.sum(1.25*knees**2*shins,axis=1)+12*np.maximum(geometry['kneeGap']-.17,0)**2*knee_cross,
   np.sum(.55*geometry['footLineAngles']**2*feet,axis=1)+12*(np.maximum(geometry['ankleGap']-.13,0)**2*ankle_cross+np.maximum(geometry['toeGap']-.11,0)**2*toe_cross)+4*geometry['crossedLegs']*toe_cross,
   np.sum((.325*pitch**2+.65*roll**2)*shoulders,axis=1),
   np.sum(.5*elbows**2*forearms,axis=1),
   np.sum(1.5*np.maximum(geometry['handAxisAngles']-np.pi/12,0)**2*hand_parts,axis=1)+(4*np.maximum(geometry['handSeparation']-.07,0)**2+20*geometry['handHeightGap']**2)*hands],axis=1)
  losses[:,-1]*=head
  self.surface_group_loss[ids]=np.maximum(self.surface_group_loss[ids],losses)
  self.entry_min_form[ids]=np.exp(-self.surface_group_loss[ids].sum(axis=1))
  limbs=np.all((~thighs|(np.abs(hips)<.45))&(~shins|(np.abs(knees)<.5))&(~shoulders|((np.abs(pitch)<.6)&(np.abs(roll)<.5)))&(~forearms|(np.abs(elbows)<.5)),axis=1)
  hand_ok=np.all(~hand_parts|(geometry['handAxisAngles']<np.pi/6),axis=1)&(~hands|((geometry['handSeparation']<.12)&(geometry['handHeightGap']<.04)))
  leg_ok=np.all(~feet|(geometry['footLineAngles']<np.radians(20)),axis=1)&(~knee_cross|(geometry['kneeGap']<.19))&(~ankle_cross|(geometry['ankleGap']<.16))&(~toe_cross|((geometry['toeGap']<.15)&~geometry['crossedLegs']))
  self.entry_geometry_valid[ids]&=(~head|hand_ok)&leg_ok
  self.entry_limbs_valid[ids]&=limbs
  velocity=sensor[:,270:360].reshape(len(ids),15,6)[:,:,3:]
  lateral=np.max(np.where(active,np.linalg.norm(velocity[:,:,:2],axis=2),0),axis=1)
  self.surface_lateral[ids]=np.maximum(self.surface_lateral[ids],lateral)
  self.surface_seen[ids]|=active;self.surface_finished[ids]|=fraction>=1;self.surface_previous[ids]=fraction
  self.entry_samples[ids]+=1
  complete=np.all(fraction==1,axis=1)
  fresh=complete&~np.isfinite(self.full_entry_time[ids]);self.full_entry_time[ids[fresh]]=state[fresh,0]
 def step(self,action,auto_reset=True):
  started_wet=np.isfinite(self.entry_time)
  old_phase_theta=self.phase_theta.copy();old_air_theta=self.air_theta.copy();old_air_twist=self.air_twist.copy();old_invalid=self.board_invalid.copy();old_impulse=self.board_impulse.copy();old_takeoff_value=self.takeoff_value.copy();old_stand_impulse=self.stand_impulse.copy()
  old_state=self.state.copy();old_t=self.state[:,0].copy();old_s=self.sensors.copy();old_theta=self.theta.copy();old_twist=self.twist.copy();g=self.goal()
  desired=decoded_action(action);self.targets+=np.clip(desired-self.targets,-TARGET_RATE*DT,TARGET_RATE*DT)
  self.ctrl[:,:,:self.model.nu]=self.targets[:,None,ACTUATOR_MAP]
  self.ctrl[:,:,self.model.nu:]=0
  push=self.disturbance*((old_t>=self.disturbance_time-1e-8)&(old_t<self.disturbance_time+.12-1e-8)&self.released&(~started_wet))
  self.ctrl[:,:,self.model.nu+3*6]=push[:,None]
  offset=self.model.nu+self.model.nbody*6
  self.ctrl[:,:,offset]=self.platform[:,None]
  offset+=1
  self.ctrl[:,:,offset+2]=-self.height[:,None];self.ctrl[:,:,offset+3]=1
  st=self.roll_state;se=self.roll_sensor;dry=np.flatnonzero(~started_wet);wet=np.flatnonzero(started_wet);wet_after=None
  if len(dry):
   dst,dse=self.pool.rollout([self.model]*len(dry),self.datas,self.state[dry],self.ctrl[dry],skip_checks=True,control_spec=3904,nstep=SUBSTEPS,state=np.empty((len(dry),SUBSTEPS,self.ns)),sensordata=np.empty((len(dry),SUBSTEPS,self.model.nsensordata)))
   st[dry]=dst;se[dry]=dse
  if len(wet):
   wst,wse,wet_after=self.water_rollout(wet);st[wet]=wst;se[wet]=wse
  geom=se[:,:,45:150].reshape(self.n,SUBSTEPS,15,7)
  gq=geom[:,:,:,3:];w,x,y,z=np.moveaxis(gq,-1,0)
  row=np.stack([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],axis=-1)
  sizes=self.model.geom_size[1:16];types=self.model.geom_type[1:16]
  extent=np.sqrt(np.sum((row*sizes)**2,axis=-1))
  for j,typ in enumerate(types):
   if typ==2:extent[:,:,j]=sizes[j,0]
   elif typ==3:extent[:,:,j]=sizes[j,0]+sizes[j,1]*np.abs(row[:,:,j,2])
   elif typ==6:extent[:,:,j]=np.sum(np.abs(row[:,:,j])*sizes[j],axis=-1)
  low=np.min(geom[:,:,:,2]-extent,axis=2)
  in_pool=(geom[:,:,:,0]>=-.44)&(geom[:,:,:,0]<=11.34)&(np.abs(geom[:,:,:,1])<=3.56)
  low=np.min(np.where(in_pool,geom[:,:,:,2]-extent,np.inf),axis=2)
  contact_steps=low<=-self.height[:,None]
  water=np.any(contact_steps,axis=1)&(~started_wet)
  # MuJoCo sensors describe the pre-integration state. Score exactly the first
  # sampled contact state, never a later pose at the end of a20ms policy tick.
  for i in np.flatnonzero(water):
   k=int(np.argmax(contact_steps[i]));self.first_geometry[i]=int(np.argmin(np.where(in_pool[i,k],geom[i,k,:,2]-extent[i,k],np.inf)))+1;entry_state=old_state[i].copy() if k==0 else st[i,k-1].copy()
   st[i,k:]=entry_state;se[i,k:]=se[i,k].copy()
   self.entry_time[i]=entry_state[0];self.first_state[i]=entry_state;self.first_sensors[i]=se[i,k]
  # Track only the scored interval; later underwater motion is not judged.
  if len(wet):
   tracking=~np.isfinite(self.full_entry_time[wet])
   for k in range(SUBSTEPS):
    active=tracking&(st[wet,k,0]<=self.entry_time[wet]+.8+1e-9);self.track_entry_batch(wet[active],st[wet[active],k],wet_after[active,k])
    for j,i in enumerate(wet):
     if tracking[j] and ((np.isfinite(self.full_entry_time[i]) and st[i,k,0]>=self.full_entry_time[i]-1e-9) or st[i,k,0]>=self.entry_time[i]+.8-1e-9):
      tracking[j]=False
      st[i,k:]=st[i,k].copy();se[i,k:]=wet_after[j,k].copy();wet_after[j,k:]=wet_after[j,k].copy()
  self.state[:]=st[:,-1];self.sensors[:]=se[:,-1]
  if len(wet):self.sensors[wet]=wet_after[:,-1]
  fresh=np.flatnonzero(water);self.track_entry_batch(fresh,self.state[fresh],self.sensors[fresh])
  self.water_measurements()
  duration=self.state[:,0]-old_t;self.applied_impulse+=push*duration;self.push_duration+=(push!=0)*duration
  q=self.state[:,1:1+self.model.nq];v=self.state[:,1+self.model.nq:];s=self.sensors
  # Accumulate observed orientation continuously, at all ten physics substeps.
  ups=quat_up(st[:,:,5:9].reshape(-1,4)).reshape(self.n,SUBSTEPS,3)
  p,t,l=framed_angles(st[:,:,5:9].reshape(-1,4));pitches=p.reshape(self.n,SUBSTEPS);twists=t.reshape(self.n,SUBSTEPS)
  self.max_lateral=np.maximum(self.max_lateral,l.reshape(self.n,SUBSTEPS).max(axis=1)*(~started_wet))
  delta=wrap(np.diff(np.concatenate([self.prev_pitch[:,None],pitches],axis=1),axis=1))
  trajectory_theta=old_theta[:,None]+np.cumsum(delta,axis=1)
  twist_path=old_twist[:,None]+np.cumsum(wrap(np.diff(np.concatenate([self.prev_twist[:,None],twists],axis=1),axis=1)),axis=1)
  hand_distance,leg_separation=tuck_geometry(se)
  forces=board_forces(se);support=stand_forces(se);nonfoot=np.ones(15,bool);nonfoot[[11,14]]=False
  physical_dt=np.diff(np.concatenate([old_t[:,None],st[:,:,0]],axis=1),axis=1)
  joints=st[:,:,1+self.qadr];target_angle=g[:,0]*2*np.pi
  # Ground state comes only from actual athlete↔board contacts, never self-touch.
  # Upright foot-only preparation bounces are allowed. Nonfoot assistance and
  # a rotated recontact invalidate the dive. Skill progress resets on recontact.
  for k in range(SUBSTEPS):
   active=(physical_dt[:,k]>0)&(~started_wet)
   allowed=np.zeros((self.n,15),bool);allowed[:,[11,14]]=~self.armstand[:,None];allowed[:,[5,8]]=self.armstand[:,None]
   nonfoot_force=np.sum(np.where(allowed,0,forces[:,k]),axis=1)
   self.board_impulse+=nonfoot_force*physical_dt[:,k]
   self.board_peak=np.maximum(self.board_peak,np.max(np.where(allowed,0,forces[:,k]),axis=1)*active)
   self.stand_impulse+=np.sum(support[:,k],axis=1)*physical_dt[:,k]
   self.stand_peak=np.maximum(self.stand_peak,np.max(support[:,k],axis=1)*active)
   # A foot pressing the side/underside is not an allowable top takeoff.
   foot_low=geom[:,k,[11,14],2] < st[:,k,1,None]+.04
   self.foot_side_contact|=np.any(foot_low&(forces[:,k,[11,14]]>15),axis=1)&active&~self.armstand
   grounded=np.sum(se[:,k,150:210:4],axis=1)>0
   foot_contacts=np.where(self.armstand[:,None],se[:,k,[170,182]]>0,se[:,k,[194,206]]>0)
   self.last_foot_contact=np.where(foot_contacts&active[:,None],st[:,k,0,None],self.last_foot_contact)
   self.takeoff_tilt_invalid|=grounded&active&(np.where(self.armstand,np.abs(np.abs(pitches[:,k])-np.pi),np.abs(pitches[:,k]))>np.pi/3)
   recontact=self.released&grounded&active
   self.preparation_bounces+=recontact
   self.rotated_recontact|=recontact&(np.where(self.armstand,-ups[:,k,2],ups[:,k,2])<.87)
   self.board_invalid|=(self.board_impulse>.1)|(self.board_peak>15)|(self.stand_impulse>.1)|(self.stand_peak>15)|self.rotated_recontact|self.foot_side_contact
   self.board_twist=np.where(grounded&active,np.maximum(self.board_twist,np.abs(twist_path[:,k])),self.board_twist)
   reset=grounded&active
   self.air_clear_time[reset]=0;self.departure_com[reset]=se[reset,k,2];self.apex_com[reset]=se[reset,k,2]
   self.released[reset]=False;self.shape_angle[reset]=0;self.max_progress[reset]=0;self.max_lateral[reset]=0
   self.shape_duration[reset]=0;self.shape_quality[reset]=0;self.shape_ticks[reset]=0;self.shape_peak[reset]=0
   first_clear=(self.air_clear_time==0)&(~grounded)&active
   self.departure_com[first_clear]=se[first_clear,k,2];self.apex_com[first_clear]=se[first_clear,k,2];self.takeoff_vertical_speed[first_clear]=se[first_clear,k,5]
   self.departure_pitch[first_clear]=pitches[first_clear,k]
   self.departure_theta[first_clear]=trajectory_theta[first_clear,k];self.departure_twist[first_clear]=twist_path[first_clear,k]
   self.apex_com=np.where((~grounded)&active,np.maximum(self.apex_com,se[:,k,2]),self.apex_com)
   self.air_clear_time+=(~grounded)*physical_dt[:,k]
   newly=(~self.released)&(self.air_clear_time>=.04-1e-8)&(st[:,k,0]>.08)&active
   self.released[newly]=True;self.release_time[newly]=st[newly,k,0]-self.air_clear_time[newly]+.002
   self.release_theta[newly]=self.departure_theta[newly];self.release_twist[newly]=self.departure_twist[newly]
   self.foot_departure_gap[newly]=np.abs(self.last_foot_contact[newly,0]-self.last_foot_contact[newly,1])
   self.takeoff_tilt_invalid|=newly&(np.where(self.armstand,np.abs(np.abs(self.departure_pitch)-np.pi),np.abs(self.departure_pitch))>np.pi/3)
   self.board_invalid|=self.takeoff_tilt_invalid|(newly&(self.foot_departure_gap>.04+1e-8))
   self.air_theta=np.where(started_wet,self.air_theta,np.where(self.released,trajectory_theta[:,k]-self.release_theta,0))
   self.phase_theta=np.where(started_wet,self.phase_theta,np.where(self.released,self.air_theta+np.where(self.armstand,wrap(self.departure_pitch-np.pi),self.departure_pitch),0))
   self.air_twist=np.where(started_wet,self.air_twist,np.where(self.released,twist_path[:,k]-self.release_twist,0))
   self.max_lateral=np.maximum(self.max_lateral,np.abs(ups[:,k,1])*self.released*(~started_wet))
  self.theta=np.where(started_wet,old_theta,trajectory_theta[:,-1]);self.twist=np.where(started_wet,old_twist,twist_path[:,-1])
  self.prev_pitch=pitches[:,-1];self.prev_twist=twists[:,-1]
  self.max_air=np.maximum(self.max_air,s[:,2]);up=ups[:,-1]
  joint=q[:,self.qadr];hip=joint[:,0];knee=joint[:,1];arms=(joint[:,6]+joint[:,9])*.5
  contacted=np.isfinite(self.entry_time)
  done=np.isfinite(self.full_entry_time)|(contacted&(self.state[:,0]-self.entry_time>=.8-1e-9))|((~contacted)&(self.state[:,0]>=3.6))|(~np.isfinite(self.state).all(axis=1))
  geometry=entry_geometry(s);straight,limbs,geo_ok,loss=entry_posture(joint,geometry)
  self.entry_omega=np.where(contacted,np.maximum(self.entry_omega,np.linalg.norm(s[:,13:16],axis=1)),self.entry_omega)
  tuck=np.exp(-((hip-1.4)/.8)**2-((knee-2)/.8)**2)*np.exp(-(np.max(tuck_geometry(s)[0],axis=1)/.45)**2)
  pike=np.exp(-((hip-1.5)/.8)**2-(knee/.35)**2)
  straight_flight=np.exp(-(hip/.35)**2-(knee/.35)**2)
  quality=np.stack([straight_flight,pike,tuck,np.maximum.reduce([straight_flight,pike,tuck])],axis=1)
  progress=self.phase_theta/np.where(np.abs(g[:,0])>1e-6,g[:,0]*2*np.pi,1)
  assess=self.released&~contacted&(((progress>.15)&(progress<.8))|(g[:,0]==0))
  self.position_sums+=quality*assess[:,None];self.position_ticks+=assess
  # Shape-agnostic small physical regularizers, no desired rotation or twist.
  reward=-.0005*np.mean((v[:,self.vadr]/12)**2,axis=1)
  reward-=.001*np.sum(geometry['footLineAngles']**2,axis=1)*self.released*~contacted
  reward-=.002*np.maximum(geometry['ankleGap']-.13,0)*self.released*~contacted
  reward-=.2*(self.board_invalid&~old_invalid)
  reward=np.where(started_wet,0,reward)
  info=[]
  for i in np.flatnonzero(done):
   qs=self.first_state[i] if contacted[i] else self.state[i]
   first_angle=float(np.degrees(np.arccos(np.clip((-1 if self.headfirst[i] else 1)*quat_up(qs[5:9][None])[0,2],-1,1))))
   w=self.entry_worst_geometry[i];js=self.first_sensors[i] if contacted[i] else s[i]
   info.append(dict(index=int(i),rotation=float(self.phase_theta[i]/(2*np.pi)),twist=float(self.air_twist[i]/(2*np.pi)),
    boardInvalid=bool(self.board_invalid[i]),water=bool(contacted[i]),fullEntryComplete=bool(np.isfinite(self.full_entry_time[i])),
    firstGeometry=int(self.first_geometry[i]),x=float(js[0]),firstContactAngle=first_angle,
    entryAngle=float(max(first_angle,self.entry_max_angle[i])),form=float(self.entry_min_form[i]),
    entryGeometryValid=bool(self.entry_geometry_valid[i]),entryLimbsValid=bool(self.entry_limbs_valid[i]),
    entryGeometryWorst=dict(footLineAngles=np.degrees(w[:2]).tolist(),handAxisAngles=np.degrees(w[2:4]).tolist(),
      **{key:float(w[4+j]) for j,key in enumerate(['handSeparation','handHeightGap','kneeGap','ankleGap','toeGap'])},crossedLegs=bool(self.entry_crossed[i])),
    entryArmPositionValid=bool(np.max(np.abs(qs[1+self.qadr[[6,9]]]- (3.05 if self.headfirst[i] else 0)))<.8),
    positionQualities=(self.position_sums[i]/max(1,self.position_ticks[i])).tolist(),
    ascent=float(max(0,self.apex_com[i]-self.departure_com[i])),preparationBounces=int(self.preparation_bounces[i]),
    surfaceLateralSpeed=float(self.surface_lateral[i]),entryAngularSpeed=float(self.entry_omega[i]),maxLateral=float(self.max_lateral[i]),
    takeoffVerticalSpeed=float(self.takeoff_vertical_speed[i]),time=float(self.state[i,0])))
  if auto_reset and done.any():self.reset(np.flatnonzero(done))
  return self.observe(),reward.astype(np.float32),done,info
