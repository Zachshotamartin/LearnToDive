import unittest
import copy
import numpy as np
import torch
from rules import *
from judge import judge,terminal_reward
from environment import Arena
from policy import Policy

class RulesTests(unittest.TestCase):
 def test_table(self):
  self.assertEqual(difficulty(IDS['101C'],'springboard',3),1.4)
  self.assertEqual(difficulty(IDS['103B'],'platform',10),1.6)
  self.assertEqual(difficulty(IDS['5132D'],'springboard',3),2.1)
  with self.assertRaises(ValueError):difficulty(IDS['101C'],'platform',6)
 def test_groups_and_repeat(self):
  self.assertEqual(DIVES[IDS['301C']]['sign'],-1)
  self.assertEqual(DIVES[IDS['401C']]['back'],1)
  self.assertFalse(legal_mask(1,'platform',10,[101])[IDS['101C']])
  self.assertFalse(legal_mask(6,'springboard',3).any())
 def metrics(self):
  return dict(rotation=.5,twist=0,boardInvalid=False,water=True,fullEntryComplete=True,firstGeometry=6,x=1,
   maxLateral=0,firstContactAngle=0,entryAngle=0,ascent=.4,preparationBounces=0,positionQuality=1,form=1,
   entryGeometryWorst=dict(footLineAngles=[0,0],ankleGap=.12,crossedLegs=False,handSeparation=.07,handHeightGap=0),
   surfaceLateralSpeed=0,entryGeometryValid=True,entryLimbsValid=True)
 def test_tail_flop_and_wrong_declaration(self):
  m=self.metrics();good=judge(IDS['101C'],'platform',10,m);self.assertEqual(good['execution'],10)
  m['entryAngle']=65;bad=judge(IDS['101C'],'platform',10,m);self.assertFalse(bad['clean']);self.assertLess(bad['execution'],good['execution'])
  m=self.metrics();wrong=judge(IDS['103C'],'platform',10,m);self.assertEqual(wrong['points'],0);self.assertLess(terminal_reward(wrong,m),0)
 def test_training_reward_stays_dense_when_execution_is_clipped(self):
  base=self.metrics();base.update(ascent=0,positionQuality=0,form=0,entryAngle=40,firstContactAngle=40);base['entryGeometryWorst']['footLineAngles']=[45,45]
  better=judge(IDS['101C'],'platform',10,base);self.assertEqual(better['execution'],0)
  m=self.metrics();m.update(ascent=0,positionQuality=0,form=0,entryAngle=40,firstContactAngle=40,surfaceLateralSpeed=3);m['entryGeometryWorst'].update(footLineAngles=[45,45],handSeparation=.4)
  worse=judge(IDS['101C'],'platform',10,m);self.assertEqual(worse['execution'],0)
  self.assertGreater(terminal_reward(better,base),terminal_reward(worse,m))
  invalid=dict(base,water=False);self.assertLess(terminal_reward(judge(IDS['101C'],'platform',10,invalid),invalid),terminal_reward(better,base))
  self.assertGreater(terminal_reward(judge(IDS['101C'],'platform',10,self.metrics()),self.metrics()),terminal_reward(better,base))
 def test_no_failed_reward_saturation(self):
  m=self.metrics();m['rotation']=.1;a=judge(IDS['103C'],'platform',10,m);r=terminal_reward(a,m)
  m['entryAngle']=80;b=judge(IDS['103C'],'platform',10,m);self.assertLess(terminal_reward(b,m),r)

class IntegrationTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);self.e=Arena(4,seed=4,threads=1)
 def tearDown(self):self.e.close()
 def test_selection_physics_and_resume(self):
  e=self.e;choices=e.mask().argmax(1);before=e.physics.state.copy()
  o,r,d,_=e.step(choices,np.zeros((4,9)));np.testing.assert_array_equal(e.physics.state[:,0],0)
  self.assertFalse(e.choosing.any());self.assertTrue(np.isfinite(o).all())
  state=e.state_dict();a=np.full((4,9),.2);x=e.step(choices,a)
  e.load_state_dict(state);y=e.step(choices,a)
  for i in range(3):np.testing.assert_allclose(x[i],y[i],rtol=0,atol=0)
 def test_policy_likelihood_and_popart(self):
  e=self.e;m=Policy(e.observation_size,(32,32));o=torch.tensor(e.observe());mask=torch.tensor(e.mask());choose=torch.tensor(e.choosing)
  out=m(o,mask,choose);again=m(o,mask,choose,out['raw'],out['choice'])
  torch.testing.assert_close(out['logp'],again['logp']);v=out['value'].detach()
  m.update_value_scale(torch.tensor([10.,20.,40.,80.]));torch.testing.assert_close(v,m(o,mask,choose)['value'],atol=1e-5,rtol=1e-5)
  again=m(o,mask,choose)
  loss=-again['logp'].mean()+again['value'].square().mean();loss.backward()
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
 def test_rigid_platform(self):
  e=self.e;e.apparatus[:]=1;e.height[:]=5;e.group[:]=1;e.used[:]=False
  e.step(np.full(4,IDS['101C']),np.zeros((4,9)))
  for _ in range(20):e.step(np.zeros(4,int),np.zeros((4,9)))
  self.assertLess(np.abs(e.physics.state[:,1]).max(),.002)
if __name__=='__main__':unittest.main()

class MeasurementTests(unittest.TestCase):
 """Physical drops that probe judge measurements; nothing here trains or prescribes motion."""
 def drop(self,vy,vz=-9.2,height=3.):
  import mujoco
  from engine import Arena as Physics,ACTION_LOW,ACTION_HIGH,ACTUATOR_MAP,STATE_SPEC
  p=Physics(1,seed=2,threads=1,skills=[0],heights=(1,10),minimum_heights=[1]*6,training=False);m=p.model;d=p.resetdata
  p.reset([0],[dict(skill=0,height=height,preload=0,disturbance=0)]);p.platform[0]=True;p.headfirst[0]=True
  target=np.array([0,0,1.35,np.pi,np.pi,-.3,.3,0,.06]);mujoco.mj_resetData(m,d);d.mocap_pos[0]=[0,0,-height]
  d.qpos[p.qadr]=target[ACTUATOR_MAP];d.ctrl[:]=target[ACTUATOR_MAP];d.qpos[4:8]=[0,0,1,0];d.qpos[1]=2.;d.qpos[3]=0;mujoco.mj_forward(m,d)
  low=min(d.geom_xpos[g,2]-np.dot(np.abs(d.geom_xmat[g].reshape(3,3)[2]),m.geom_size[g]) for g in range(1,16))
  d.qpos[3]+=-height+.03-low;d.qvel[:]=0;d.qvel[2]=vy;d.qvel[3]=vz;mujoco.mj_forward(m,d)
  mujoco.mj_getState(m,d,p.state[0],STATE_SPEC);p.sensors[0]=d.sensordata;p.targets[0]=target;p.released[0]=True
  a=(2*(target-ACTION_LOW)/(ACTION_HIGH-ACTION_LOW)-1)[None];info=None
  for _ in range(120):
   _,_,done,info=p.step(a,auto_reset=False)
   if done[0]:break
  p.close();return info[0]
 def test_surface_lateral_speed_is_sideways_linear_speed(self):
  still=self.drop(0.);moving=self.drop(2.)
  # Limbs deflect sideways under the impact, so a vertical drop is not exactly zero,
  # but it must be metres per second, not the tens of radians per second of the old reading.
  self.assertLess(still['surfaceLateralSpeed'],1.5)
  self.assertGreater(moving['surfaceLateralSpeed']-still['surfaceLateralSpeed'],1.)
  self.assertLess(moving['surfaceLateralSpeed'],3.6)
  self.assertTrue(still['fullEntryComplete']);self.assertLess(still['entryAngle'],6)
 def hop(self,vz,gap=.2):
  import mujoco
  from engine import Arena as Physics,ACTION_LOW,ACTION_HIGH,STATE_SPEC
  p=Physics(1,seed=2,threads=1,skills=[0],heights=(1,10),minimum_heights=[1]*6,training=False);m=p.model;d=p.resetdata
  p.reset([0],[dict(skill=0,height=10,preload=0,disturbance=0)]);p.platform[0]=True
  mujoco.mj_setState(m,d,p.state[0],STATE_SPEC);d.qpos[3]+=gap;d.qvel[:]=0;d.qvel[3]=vz;mujoco.mj_forward(m,d)
  mujoco.mj_getState(m,d,p.state[0],STATE_SPEC);p.sensors[0]=d.sensordata
  a=(2*(p.targets[0]-ACTION_LOW)/(ACTION_HIGH-ACTION_LOW)-1)[None]
  for _ in range(60):
   _,_,done,_=p.step(a,auto_reset=False)
   if done[0]:break
  bounces=int(p.preparation_bounces[0]);p.close();return bounces
 def test_instability_reset_ends_the_dive_instead_of_teleporting(self):
  from engine import Arena as Physics
  p=Physics(1,seed=2,threads=1,skills=[0],heights=(1,10),minimum_heights=[1]*6,training=False)
  p.reset([0],[dict(skill=0,height=10,preload=0,disturbance=0,platform=True)]);p.platform[0]=True
  p.state[0,2:5]=[2.,0.,-3.];p.state[0,26]=1e12   # state = [time, qpos(22), qvel(21)]: root z velocity beyond mjMAXVAL
  _,_,done,info=p.step(np.zeros((1,9)),auto_reset=False)
  self.assertTrue(done[0]);self.assertGreaterEqual(p.diverged_events,1);self.assertFalse(info[0]['water']);p.close()
 def test_preparation_bounce_requires_an_upward_hop(self):
  self.assertEqual(self.hop(-.2),0)   # dropped onto the platform: contact regained without a hop
  self.assertEqual(self.hop(.15,gap=.01),0)  # a wobble of the board or a toe lift is not a hop either
  self.assertGreaterEqual(self.hop(1.8,gap=.02),1)  # launched upward, lands again: one hop
 def test_distance_is_a_deduction_and_only_an_unsafe_dive_is_capped(self):
  m=RulesTests.metrics(RulesTests());m['x']=.3;s=judge(IDS['101C'],'platform',10,m)
  self.assertTrue(s['valid']);self.assertAlmostEqual(s['deductions']['distance'],1.2);self.assertGreater(s['execution'],2)
  m['x']=.1;s=judge(IDS['101C'],'platform',10,m);self.assertTrue(s['valid']);self.assertLessEqual(s['execution'],2)
 def test_position_deduction_matches_rule_range(self):
  m=RulesTests.metrics(RulesTests());m['positionQuality']=.5
  self.assertAlmostEqual(judge(IDS['101C'],'platform',10,m)['deductions']['position'],1.)
  m['positionQuality']=0.;self.assertAlmostEqual(judge(IDS['101C'],'platform',10,m)['deductions']['position'],2.)

class StanceTests(unittest.TestCase):
 """The environment must make a real takeoff physically reachable; the controller still has to learn it."""
 def physics(self,height=10,**context):
  from engine import Arena as Physics
  p=Physics(1,seed=3,threads=1,skills=[0],heights=(1,10),minimum_heights=[1]*6,training=False)
  p.reset([0],[dict(skill=0,height=height,preload=0,disturbance=0,**context)]);p.platform[0]=True;p.headfirst[0]=True;p.goals[0]=[.5,0,0,3]
  return p
 def test_standing_start_is_flat_footed_balanced_and_not_over_the_edge(self):
  import mujoco
  from engine import ACTION_LOW,ACTION_HIGH,STATE_SPEC
  p=self.physics();m=p.model;d=mujoco.MjData(m);mujoco.mj_setState(m,d,p.state[0],STATE_SPEC);mujoco.mj_forward(m,d)
  gids=[int(m.body_geomadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)]) for n in ['foot_L','foot_R']]
  for g in gids:self.assertLess(abs(d.geom_xmat[g].reshape(3,3)[2,0]),1e-3)
  front=max(d.geom_xpos[g,0]+np.abs(d.geom_xmat[g].reshape(3,3)[0])@m.geom_size[g] for g in gids);rear=min(d.geom_xpos[g,0]-np.abs(d.geom_xmat[g].reshape(3,3)[0])@m.geom_size[g] for g in gids)
  com=float(d.subtree_com[2][0]);self.assertAlmostEqual(front,.03,places=3);self.assertGreater(com,rear);self.assertLess(com,front)
  a=(2*(p.targets[0]-ACTION_LOW)/(ACTION_HIGH-ACTION_LOW)-1)[None]
  for _ in range(150):
   p.step(a,auto_reset=False)
   if p.released[0]:break
  self.assertGreater(float(p.state[0,0]),1.0,'an unbalanced start topples before a controller could act');p.close()
 def test_forward_countermovement_takeoff_is_physically_reachable(self):
  # Open-loop feasibility probe (from a bounded random search); never a training target.
  from engine import ACTION_LOW,ACTION_HIGH
  def act(t):return (2*(np.asarray(t)-ACTION_LOW)/(ACTION_HIGH-ACTION_LOW)-1)[None]
  p=self.physics(lean=0.,hip=.12,knee=.2,toeOver=.03)
  crouch=np.array([.67,1.73,-.55,.01,.01,0,0,0,0.]);extend=np.array([.32,0.,-.13,3.1,3.1,-.3,.3,0,.06]);straight=np.array([0,0,1.2,3.14,3.14,-.3,.3,0,.06])
  info=None;t_ext=None
  for _ in range(220):
   t=p.state[0,0]
   if t<.1:target=p.targets[0]
   elif t<.52:target=crouch
   elif not p.released[0]:
    t_ext=t if t_ext is None else t_ext;target=extend.copy()
    if t-t_ext>=.25:target[2]=1.3
   else:target=straight
   _,_,done,rows=p.step(act(target),auto_reset=False)
   if rows:info=rows[0];break
  ascent=float(max(0,p.apex_com[0]-p.departure_com[0]));p.close()
  self.assertIsNotNone(info);self.assertTrue(info['water']);self.assertFalse(info['boardInvalid']);self.assertGreater(info['x'],.4)
  self.assertGreater(ascent,.2);self.assertGreater(info['takeoffVerticalSpeed'],2.)

class PriorTests(unittest.TestCase):
 def test_untrained_policy_holds_the_reset_stance_and_observes_support(self):
  torch.manual_seed(1);e=Arena(2,seed=6,threads=1,training=False)
  self.assertEqual(e.observation_size,223)
  o=e.observe();support=o[:,4+6+14+14+9+3+3+11:4+6+14+14+9+3+3+11+4]
  self.assertTrue((support[:,:2]==1).all(),'both feet start supported on springboard and platform alike')
  self.assertTrue(np.all(np.abs(support[:,2]-np.where(e.apparatus==1,0,-.05123*4))<1e-3))
  m=Policy(e.observation_size,(32,32),initial_action=e.initial_action)
  previous=np.zeros(9,np.float32)
  for _ in range(12):
   obs=torch.tensor(np.concatenate([o[:1,:-9],previous[None]],axis=1));out=m(obs,torch.tensor(e.mask()[:1]),torch.tensor(e.choosing[:1]),deterministic=True)
   previous=out['action'][0].detach().numpy()
  np.testing.assert_allclose(previous,np.clip(e.initial_action,-.95,.95),atol=.06)
  e.close()
