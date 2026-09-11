"""Numerical regressions for the actual bounded action probability distribution."""
import unittest,sys,math,tempfile,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.distributions import Normal
from train import ActorCritic,cap_exploration,restore_optimizer
from evaluate import actor,run
from sim import execution_estimate,entry_geometry_error,Arena,ACTION_LOW,ACTION_HIGH
from qualify import catalog_errors
from sim import SKILLS

torch.set_num_threads(1)
class BoundedPolicyTests(unittest.TestCase):
 def test_release_export_requires_evidence_before_reading_or_mutating_assets(self):
  from export_assets import export
  with self.assertRaisesRegex(ValueError,'frozen native/WASM qualification'):
   export('does-not-exist-policy.json','does-not-exist-initial.json','release')
 def test_phase_distribution_migration_preserves_old_actor_and_flight_likelihood(self):
  torch.manual_seed(8127);old=ActorCritic();phase=ActorCritic(phase_exploration=True)
  original=copy.deepcopy(old.state_dict());phase.load_state_dict(original)
  obs=torch.randn(16,76)*.1;obs[:,63:67]=torch.tensor([.2,0,0,0]);obs[:,58]=1
  latent=torch.randn(16,9);a,lp,_,_,_=old.get(obs,latent);b,newlp,_,_,_=phase.get(obs,latent)
  torch.testing.assert_close(phase.mean_action(obs),old.mean_action(obs),rtol=0,atol=0)
  torch.testing.assert_close(newlp,lp,rtol=0,atol=0);torch.testing.assert_close(a,b,rtol=0,atol=0)
  with torch.no_grad():phase.logstd[:,0,:3]=math.log(.45)
  cap_exploration(phase,.7,.35,.5)
  torch.testing.assert_close(phase.logstd[:,1],old.logstd,rtol=0,atol=0)
  torch.testing.assert_close(phase.get(obs,latent)[1],lp,rtol=0,atol=0)
  for key,value in original.items():torch.testing.assert_close(value,old.state_dict()[key],rtol=0,atol=0)
 def test_phase_selection_requires_observed_foot_load_and_no_confirmed_flight(self):
  obs=torch.zeros(6,76);obs[:,58]=torch.tensor([0,0,0,1,1,0]);obs[:,61]=torch.tensor([1,0,0,1,0,0]);obs[:,62]=torch.tensor([0,1,0,0,0,1])
  torch.testing.assert_close(ActorCritic.exploration_phase(obs),torch.tensor([0,0,1,1,1,0]))
  torch.manual_seed(55);agent=ActorCritic(phase_exploration=True);cap_exploration(agent,.22,.35,.5)
  with torch.no_grad():agent.logstd[:,0,:3]=math.log(.45)
  z=torch.randn(6,9);mu=agent.mean_logits(obs);std=agent.logstd[torch.zeros(6,dtype=torch.long),ActorCritic.exploration_phase(obs)].exp()
  expected=(Normal(mu,std).log_prob(z)-ActorCritic.log_jacobian(z)).sum(-1)
  _,lp,_,_,_=agent.get(obs,z);torch.testing.assert_close(lp,expected)
  _,again,entropy,_,_=agent.get(obs,z);torch.testing.assert_close((again-lp).exp(),torch.ones(6))
  self.assertTrue(torch.isfinite(entropy).all());(-again.mean()).backward()
  self.assertGreater(float(agent.logstd.grad[:,0,:3].abs().sum()),0);self.assertGreater(float(agent.logstd.grad[:,1].abs().sum()),0)
 def test_phase_optimizer_requires_explicit_fresh_migration_and_resumes_immutably(self):
  torch.manual_seed(73);old=ActorCritic();old_opt=torch.optim.Adam(old.parameters(),lr=.0001)
  old.logstd.sum().backward();old_opt.step()
  checkpoint={'model':copy.deepcopy(old.state_dict()),'optimizer':copy.deepcopy(old_opt.state_dict())}
  phase=ActorCritic(phase_exploration=True);phase.load_state_dict(checkpoint['model']);opt=torch.optim.Adam(phase.parameters(),lr=.0001)
  with self.assertRaisesRegex(ValueError,'reset-optimizer'):restore_optimizer(opt,checkpoint,phase,False)
  restore_optimizer(opt,checkpoint,phase,True);self.assertEqual(len(opt.state),0)
  phase.logstd.sum().backward();opt.step();new_checkpoint={'model':copy.deepcopy(phase.state_dict()),'optimizer':copy.deepcopy(opt.state_dict())}
  frozen=copy.deepcopy(new_checkpoint);resumed=ActorCritic(phase_exploration=True);resumed.load_state_dict(new_checkpoint['model']);resumed_opt=torch.optim.Adam(resumed.parameters(),lr=.0001)
  restore_optimizer(resumed_opt,new_checkpoint,resumed,False)
  resumed.logstd.sum().backward();resumed_opt.step()
  for key,state in new_checkpoint['optimizer']['state'].items():
   for name,value in state.items():torch.testing.assert_close(value,frozen['optimizer']['state'][key][name],rtol=0,atol=0)
  self.assertEqual(tuple(resumed.logstd.shape),(6,2,9));self.assertGreater(float(resumed_opt.state[resumed.logstd]['step']),float(opt.state[phase.logstd]['step']))
 def test_launch_reward_uses_departure_velocity_before_confirmation_gravity_loss(self):
  env=Arena(1,training=False,launch_speed_weight=1.5)
  env.reset(np.array([0]),[dict(skill=0,height=7.5,tilt=.16,preload=-.18,x=-.13,disturbance=0)])
  target=np.array([-.1,0,1.3,1.5,1.5,0,0,0,0]);command=(2*(target-ACTION_LOW)/(ACTION_HIGH-ACTION_LOW)-1)[None]
  try:
   for _ in range(80):
    env.step(command,auto_reset=False)
    if env.released[0]:break
   self.assertTrue(env.released[0]);self.assertFalse(env.board_invalid[0])
   departure=env.takeoff_vertical_speed[0];current=env.sensors[0,5]
   self.assertGreater(departure-current,.3)
   momentum=.7*np.exp(-((env.sensors[0,7]-25)/25)**2)
   expected=.15*np.clip(env.sensors[0,3],0,2)+1.5*np.clip(departure,0,4)+momentum
   # Free-flight horizontal momentum changes only by numerical integration;
   # the old delayed-velocity reward differs by more than half a reward unit.
   self.assertAlmostEqual(env.takeoff_value[0],expected,delta=.02)
   delayed=.15*np.clip(env.sensors[0,3],0,2)+1.5*np.clip(current,0,4)+momentum
   self.assertGreater(env.takeoff_value[0]-delayed,.4)
  finally:env.close()
 def test_qualification_refuses_runtime_height_range_below_the_measured_range(self):
  heights=[s['height']for s in SKILLS];catalog=[dict(id=s['id'],minHeight=h)for s,h in zip(SKILLS,heights)]
  self.assertEqual(catalog_errors(catalog,heights),[])
  narrower=heights.copy();narrower[4]=7.5
  self.assertTrue(any(SKILLS[4]['id']in error for error in catalog_errors(catalog,narrower)))
  self.assertTrue(catalog_errors(catalog[:-1],heights))
 def test_refinement_caps_noise_without_changing_deterministic_actions(self):
  torch.manual_seed(71);agent=ActorCritic();obs=torch.randn(8,76);before=agent.mean_action(obs).detach().clone()
  cap_exploration(agent,.22)
  self.assertLessEqual(float(agent.logstd.exp().max().detach()),.220001)
  torch.testing.assert_close(agent.mean_action(obs),before)
  with self.assertRaises(ValueError):cap_exploration(agent,0)
  with self.assertRaises(ValueError):cap_exploration(agent,2)
 def test_entry_training_loss_uses_geometry_and_keeps_all_dimensions_bounded(self):
  clean={'handSeparation':np.array([.07]),'handHeightGap':np.array([0.]),'handAxisAngles':np.array([[.1,.1]]),'footLineAngles':np.array([[.1,.1]]),'ankleGap':np.array([.12])}
  self.assertEqual(float(entry_geometry_error(clean)[0]),0)
  for name,value in [('handSeparation',np.array([.15])),('handHeightGap',np.array([.05])),('handAxisAngles',np.array([[.8,.1]])),('footLineAngles',np.array([[.8,.1]])),('ankleGap',np.array([.25]))]:
   error=entry_geometry_error({**clean,name:value})[0];self.assertGreater(error,0);self.assertLessEqual(error,4)
 def test_reward_refinement_cannot_change_physics_or_judge(self):
  from evaluate import baseline
  a=Arena(1,training=False);b=Arena(1,training=False,ascent_weight=24,launch_speed_weight=1.5,entry_geometry_weight=6)
  context=[dict(skill=0,height=3,tilt=.16,preload=-.18,x=-.13,disturbance=0)];a.reset(np.array([0]),context);b.reset(np.array([0]),context)
  different=False
  try:
   for _ in range(181):
    command=baseline(a);_,ra,done,ia=a.step(command,auto_reset=False);_,rb,_,ib=b.step(command,auto_reset=False)
    np.testing.assert_array_equal(a.state,b.state);np.testing.assert_array_equal(a.observe(),b.observe());different|=bool(np.any(ra!=rb))
    if done[0]:
     for key in ['entryAngle','form','geometryLimits','ascent','boardInvalid','shapeFraction','valid']:self.assertEqual(ia[0][key],ib[0][key])
     break
   self.assertTrue(different)
  finally:a.close();b.close()
 def test_change_of_variables_density(self):
  mu=torch.tensor([[.2,-.4]],dtype=torch.float64);std=torch.tensor([[.6,.8]],dtype=torch.float64);z=torch.tensor([[.5,-1.1]],dtype=torch.float64)
  d=Normal(mu,std);logp,_=ActorCritic.statistics(d,z,False)
  reference=(d.log_prob(z)-torch.log(1-torch.tanh(z)**2)).sum(-1)
  torch.testing.assert_close(logp,reference,rtol=1e-12,atol=1e-12)
  # Stable even when floating-point tanh itself rounds to exactly ±1.
  self.assertTrue(torch.isfinite(ActorCritic.log_jacobian(torch.tensor([-50.,50.]))).all())
 def test_ppo_reuses_latent_sample_without_clipping_or_inverse_tanh(self):
  torch.manual_seed(47);agent=ActorCritic();obs=torch.randn(32,76)*.1;obs[:,63:67]=torch.tensor([.2,0.,0.,0.])
  a,old,_,_,z=agent.get(obs);_,new,entropy,_,again=agent.get(obs,z.detach())
  torch.testing.assert_close((new-old).exp(),torch.ones_like(old))
  torch.testing.assert_close(a,torch.tanh(z));torch.testing.assert_close(again,z.detach())
  self.assertTrue((a.abs()<=1).all());self.assertTrue(torch.isfinite(entropy).all())
  (-new.mean()-.01*entropy.mean()).backward()
  self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()for p in agent.parameters()))
 def test_entropy_is_for_the_bounded_action(self):
  torch.manual_seed(71);mu=torch.full((100000,1),1.8,requires_grad=True);std=torch.full((100000,1),.7,requires_grad=True);d=Normal(mu,std)
  _,entropy=ActorCritic.statistics(d,d.sample());estimate=entropy.mean()
  self.assertLess(float(estimate.detach()),math.log(2))
  self.assertLess(float(estimate.detach()),float(d.entropy().mean().detach())-1)
  (-estimate).backward();self.assertTrue(torch.isfinite(mu.grad).all());self.assertTrue(torch.isfinite(std.grad).all())
 def test_exported_deterministic_mean_transform(self):
  torch.manual_seed(71);agent=ActorCritic();obs=torch.randn(8,76)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'policy.json';agent.save_json(p,0,{'seed':71});infer,model=actor(p)
   self.assertEqual(model['actionDistribution'],'tanh-squashed-gaussian-v1')
   np.testing.assert_allclose(infer(obs.numpy()),agent.mean_action(obs).detach().numpy(),rtol=2e-5,atol=1e-7)
 def test_evaluation_keeps_starting_parameters_separate_from_outcome_position(self):
  context=dict(skill=0,height=5.7,tilt=.17,preload=-.18,x=-.143,disturbance=0)
  rows,_=run([context],mode='baseline')
  self.assertEqual(rows[0]['parameters'],context)
  self.assertNotEqual(rows[0]['x'],context['x'])
  if rows[0]['boardInvalid']:
   self.assertFalse(rows[0]['valid']);self.assertEqual(rows[0]['execution'],0)
 def test_execution_reward_obeys_declared_position_and_launch_gates(self):
  # Identical attractive entries: only the first actually completes the task.
  score=execution_estimate(np.full(4,2.),np.full(4,.97),np.full(4,.1),np.zeros(4),np.zeros(4),np.array([True,True,False,True]),np.array([True,False,True,True]),np.ones(4,bool),np.array([False,False,False,True]),np.full(4,.3))
  np.testing.assert_array_equal(score,[9.5,2,0,0])
  # No difficulty term can reward attempting a harder failed dive.
  self.assertGreater(score[0],score[1])
 def test_refinement_reward_matches_the_displayed_judge_on_actual_physics(self):
  contexts=[dict(skill=s,height=h,tilt=.16,preload=-.18,x=-.13,disturbance=0)for s,h in [(0,5),(1,7.5),(3,5),(5,10)]]
  rows,_=run(contexts,mode='baseline')
  for row in rows:
   value=execution_estimate(row['entryAngle'],row['form'],row['splash'],row['preparationBounces'],row['boardTwist'],row['numericCorrect'] and row['water'],row['shapeCorrect'],row['validEntry'],np.bool_(row['boardInvalid']),row['ascent'])
   self.assertEqual(float(value),row['execution'])
 def test_v7_migration_preserves_eight_actions_and_zeroes_five_new_input_columns(self):
  torch.manual_seed(71);old=ActorCritic(obs=67,actions=8);new=ActorCritic(obs=72);new.migrate_from_v6(old.state_dict())
  obs=torch.randn(32,67);extra=torch.randn(32,5)
  torch.testing.assert_close(old.mean_action(obs),new.mean_action(torch.cat([obs,extra],1))[:,:8],rtol=1e-5,atol=2e-7)
  self.assertEqual(torch.count_nonzero(new.actor[0].weight[:,67:]).item(),0)
  self.assertEqual(torch.count_nonzero(new.critic[0].weight[:,67:]).item(),0)
  # The new head is independently initialized rather than copying an old skill output.
  self.assertGreater(torch.count_nonzero(new.actor[-1].weight[8]).item(),90)
  action=new.mean_action(torch.zeros(1,72))[0,8].item()
  self.assertAlmostEqual(-.15+(action+1)*.5*.33,0,places=6)
if __name__=='__main__':unittest.main()
