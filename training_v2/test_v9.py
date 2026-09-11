import unittest,copy
import numpy as np,mujoco
from sim import Arena,initial_state
from checkpoints import capture_environment,restore_environment
from rewards import whole_entry_terms
class V9(unittest.TestCase):
 def test_physical_entry_and_submerged_limbs(self):
  e=Arena(1,threads=1,training=False);d=mujoco.MjData(e.model)
  try:
   initial_state(e.model,d,7.5,1);d.qpos[1:4]=[2,0,-7.5];d.qpos[4:8]=[0,0,1,0]
   d.qpos[e.qadr]=[0,0,1.35,0,0,1.35,2.8,-.24,0,2.8,.24,0,.04,.04]
   mujoco.mj_forward(e.model,d);e.height[0]=7.5
   state=e.state[:1].copy();state[0,1:1+e.model.nq]=d.qpos
   e.track_entry_batch(np.array([0]),state,d.sensordata[None].copy())
   # Once all parts crossed, arbitrary underwater arm/leg changes cannot change entry.
   e.surface_finished[:]=True;before=(e.entry_min_form.copy(),e.entry_worst_geometry.copy())
   d.qpos[e.qadr[6]]=0;d.qpos[e.qadr[2]]=0;mujoco.mj_forward(e.model,d);state[0,1:1+e.model.nq]=d.qpos
   e.track_entry_batch(np.array([0]),state,d.sensordata[None].copy())
   np.testing.assert_array_equal(before[0],e.entry_min_form);np.testing.assert_array_equal(before[1],e.entry_worst_geometry)
  finally:e.close()
 def test_restorable_curriculum_and_rng(self):
  a=Arena(2,threads=1,entry_curriculum=True);b=Arena(2,threads=1,entry_curriculum=True)
  try:
   for _ in range(8):a.step(np.zeros((2,9)))
   a.entry_bank[0]=[{'state':a.state[0].copy()}];a.curriculum_attempts[0]=13
   restore_environment(b,capture_environment(a))
   for _ in range(4):
    left=a.step(np.zeros((2,9)));right=b.step(np.zeros((2,9)))
    for i in range(3):np.testing.assert_array_equal(left[i],right[i])
   np.testing.assert_array_equal(a.entry_bank[0][0]['state'],b.entry_bank[0][0]['state'])
  finally:a.close();b.close()
 def test_alignment_cost_order(self):
  angles=np.array([0.,5,15,30,60]);terms=whole_entry_terms(angles,np.ones(5),np.zeros(5))
  self.assertTrue(np.all(np.diff(terms['quality'])<0));self.assertTrue(np.all(np.diff(terms['angleCost'])>0))
if __name__=='__main__':unittest.main()
