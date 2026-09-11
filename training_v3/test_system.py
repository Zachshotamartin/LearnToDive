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
