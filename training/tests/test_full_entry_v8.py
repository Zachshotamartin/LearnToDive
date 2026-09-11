"""V8 regressions: append-only migration, whole-entry boundary and launch credit."""
import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,mujoco
from train import ActorCritic
from sim import Arena,initial_state,ACTUATOR_MAP

class FullEntryTests(unittest.TestCase):
 def test_append_only_observations_preserve_every_old_action_at_migration(self):
  torch.manual_seed(718);old=ActorCritic(obs=72,phase_exploration=True);new=ActorCritic(phase_exploration=True)
  original={k:v.clone()for k,v in old.state_dict().items()};new.migrate_from_v7(original)
  obs=torch.randn(48,72);extra=torch.randn(48,4)*10
  torch.testing.assert_close(old.mean_action(obs),new.mean_action(torch.cat([obs,extra],1)),rtol=1e-5,atol=2e-7)
  self.assertEqual(torch.count_nonzero(new.actor[0].weight[:,72:]).item(),0)
  self.assertEqual(torch.count_nonzero(new.critic[0].weight[:,72:]).item(),0)
  torch.testing.assert_close(new.logstd,old.logstd,rtol=0,atol=0)
  for k,v in original.items():torch.testing.assert_close(v,old.state_dict()[k],rtol=0,atol=0)
 def test_velocity_sensor_append_does_not_reinterpret_any_old_sensor(self):
  e=Arena(1,training=False)
  try:
   self.assertEqual(e.observe().shape,(1,76));self.assertEqual(e.model.nsensordata,360)
   d=e.resetdata;d.qvel[1:7]=[.7,-.3,-4.,.4,.8,.2];mujoco.mj_forward(e.model,d)
   vel=d.sensordata[270:360].reshape(15,6)
   for i,b in enumerate(e.model.geom_bodyid[1:16]):
    v=np.empty(6);mujoco.mj_objectVelocity(e.model,d,1,int(b),v,0)
    np.testing.assert_allclose(d.xipos[b],d.geom_xpos[i+1],atol=1e-14)
    np.testing.assert_allclose(vel[i],np.r_[v[3:],v[:3]],atol=1e-13)
  finally:e.close()
 def test_last_geometry_boundary_not_first_hands_contact(self):
  e=Arena(1,training=False)
  try:
   m,d=e.model,e.resetdata;e.height[0]=7.5;initial_state(m,d,7.5,0)
   d.qpos[1:4]=[2,0,-7.5+.4];d.qpos[4:8]=[0,0,1,0]
   target=np.array([0,0,1.35,2.8,2.8,-.24,.24,0,.04]);d.qpos[e.qadr]=target[ACTUATOR_MAP];d.ctrl[:]=target[ACTUATOR_MAP]
   mujoco.mj_forward(m,d);state=np.empty(e.ns);mujoco.mj_getState(m,d,state,15)
   e.entry_time[0]=0;e.track_entry_batch(np.array([0]),state[None],d.sensordata[None])
   self.assertTrue(np.isnan(e.full_entry_time[0]));self.assertLess(e.entry_max_angle[0],1e-6)
   # Actual physical pose tilted45° before last feet crossing must be retained.
   a=np.pi/4;d.qpos[4:8]=[np.sin(a/2),0,np.cos(a/2),0];mujoco.mj_forward(m,d);mujoco.mj_getState(m,d,state,15)
   e.track_entry_batch(np.array([0]),state[None],d.sensordata[None]);self.assertGreater(e.entry_max_angle[0],44.99)
   # Last geometry now fully underwater. The boundary is not a COM threshold.
   d.qpos[3]-=2;d.time=.3;mujoco.mj_forward(m,d);mujoco.mj_getState(m,d,state,15)
   e.track_entry_batch(np.array([0]),state[None],d.sensordata[None]);self.assertAlmostEqual(e.full_entry_time[0],.3)
   e.full_entry_time[0]=np.nan;d.qpos[1]=20;mujoco.mj_forward(m,d);mujoco.mj_getState(m,d,state,15)
   e.track_entry_batch(np.array([0]),state[None],d.sensordata[None]);self.assertTrue(np.isnan(e.full_entry_time[0]))
  finally:e.close()
 def test_contact_cancels_prior_takeoff_credit_without_awarding_a_new_launch(self):
  e=Arena(1,training=False)
  try:
   # A previous preparation hop's potential must be debited once feet reload.
   e.takeoff_value[0]=2.75
   command=np.zeros((1,9));e.step(command,auto_reset=False)
   self.assertFalse(e.released[0]);self.assertEqual(e.takeoff_value[0],0)
  finally:e.close()

class CriticCalibrationTests(unittest.TestCase):
 def test_calibration_changes_only_value_weights_and_uses_completed_mc_targets(self):
  import tempfile,json
  from critic_calibration import calibrate,actor_hash
  torch.manual_seed(718);agent=ActorCritic(phase_exploration=True);before=actor_hash(agent)
  value_before=[p.detach().clone()for p in agent.critic.parameters()]
  env=Arena(8,training=True)
  try:
   with tempfile.TemporaryDirectory() as d:
    report=calibrate(env,agent,1,.001,2,Path(d)/'calibration.json')
    self.assertEqual(actor_hash(agent),before);self.assertEqual(report['actorBeforeSHA256'],report['actorAfterSHA256'])
    self.assertTrue(any(not torch.equal(a,b)for a,b in zip(value_before,agent.critic.parameters())))
    self.assertFalse(report['bootstrap']);self.assertTrue(report['heldoutIncompleteTailsExcluded'])
    self.assertEqual(report['trainingInteractions'],1920);self.assertEqual(report['validationInteractions'],1920);self.assertGreater(report['retainedMCSamples'],0);self.assertLess(report['retainedMCSamples'],report['trainingInteractions'])
    self.assertTrue(np.isfinite(report['before']['all']['rmse']));self.assertTrue(np.isfinite(report['after']['all']['rmse']))
  finally:env.close()


class MatchedHeadEntryTests(unittest.TestCase):
 def test_unbraced_motor_held_entry_is_physically_reachable_at_three_speeds(self):
  from water import apply_native
  for speed in [7,10,13]:
   e=Arena(1,training=False);m=e.model;d=e.resetdata
   try:
    initial_state(m,d,7.5,0);target=np.array([0,0,1.31,3.07,3.07,-.24,.24,0,.04]);d.qpos[e.qadr]=target[ACTUATOR_MAP];d.ctrl[:]=target[ACTUATOR_MAP]
    d.qpos[1:4]=[2,0,-6];angle=np.pi-.1;d.qpos[4:8]=[np.cos(angle/2),0,np.sin(angle/2),0];d.qvel[:]=0;mujoco.mj_forward(m,d)
    water=apply_native(m,d,7.5);geo=d.sensordata[45:150].reshape(15,7);d.qpos[3]+=-7.5-np.min(geo[:,2]-water['extent'])+.002
    d.qvel[5]=1;mujoco.mj_forward(m,d);d.qvel[5]=15/d.sensordata[7];mujoco.mj_forward(m,d);d.qvel[1:4]+=np.array([0,0,-speed])-d.sensordata[3:6];mujoco.mj_forward(m,d)
    self.assertEqual(d.ncon,0);complete=False
    # Initial-condition feasibility only: no further qpos/qvel writes, no labels
    # or diagnostic controls used in training. Real motors/forces integrate.
    for _ in range(400):
     mujoco.mj_forward(m,d);water=apply_native(m,d,7.5)
     self.assertFalse(any(3 in c.geom for c in d.contact),'No invisible head bracing is allowed in this witness')
     from sim import entry_geometry,entry_posture,quat_up
     geometry=entry_geometry(d.sensordata);form,limbs,limits,_=entry_posture(d.qpos[e.qadr],geometry)
     self.assertGreater(form,.55);self.assertTrue(limbs and limits);self.assertLess(np.degrees(np.arccos(np.clip(-quat_up(d.qpos[4:8][None])[0,2],-1,1))),35)
     self.assertLessEqual(np.max(abs(d.actuator_force)),180+1e-8)
     if np.all(water['fraction']==1):complete=True;break
     mujoco.mj_step(m,d)
    self.assertTrue(complete)
   finally:e.close()

if __name__=='__main__':unittest.main()
