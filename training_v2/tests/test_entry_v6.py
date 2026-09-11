"""Reachability, anti-exploit and physical contract regressions for v6."""
import unittest,sys,tempfile,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,mujoco
from sim import ROOT,initial_state,entry_geometry,entry_posture,ACTION_HIGH,ACTION_LOW,TARGET_RATE,ACTUATOR_MAP
from contract import runtime_contract,validate_contract
from evaluate import actor
from export_assets import export

class PhysicalEntryTests(unittest.TestCase):
 def setUp(self):
  self.m=mujoco.MjModel.from_xml_path(str(ROOT/'public/physics/diver.xml'));self.d=mujoco.MjData(self.m)
  self.ids=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]]
 def pose(self,ankle=1.35):
  initial_state(self.m,self.d,7.5,1);self.d.qpos[1:4]=[2,0,4];self.d.qpos[4:8]=[0,0,1,0]
  target=np.array([0,0,ankle,0,0,ankle,2.8,-.24,0,2.8,.24,0,.04,.04])
  self.d.qpos[self.ids]=target;self.d.ctrl[:]=target;mujoco.mj_forward(self.m,self.d)
  return target
 def metrics(self):
  mujoco.mj_forward(self.m,self.d);g=entry_geometry(self.d.sensordata)
  return g,entry_posture(self.d.qpos[self.ids],g)
 def test_contact_free_reachable_kinematic_entry(self):
  self.pose();g,(form,limbs,limits,_)=self.metrics()
  self.assertEqual(self.d.ncon,0);self.assertTrue(limbs and limits);self.assertGreater(form,.85)
  self.assertLess(max(np.degrees(g['footLineAngles'])),5);self.assertLess(g['handSeparation'],.12)
 def test_motor_driven_100ms_entry_correction_without_pose_override(self):
  # Start 0.25 rad short of the target, then use only the actual servos for 100 ms.
  # No qpos/qvel writes occur after this initial condition; target slew is the
  # same 8 rad/s as the learned controller and torque limits are unchanged.
  self.pose(1.1);targets=np.array([0,0,1.1,2.8,2.8,-.24,.24,0.,.04]);desired=targets.copy();desired[2]=1.35
  peak=0
  for tick in range(5):
   targets+=np.clip(desired-targets,-TARGET_RATE*.02,TARGET_RATE*.02)
   self.d.ctrl[:]=targets[ACTUATOR_MAP]
   for _ in range(10):
    mujoco.mj_step(self.m,self.d);peak=max(peak,np.max(np.abs(self.d.actuator_force)));self.assertEqual(self.d.ncon,0)
  g,(form,limbs,limits,_)=self.metrics()
  self.assertTrue(limbs and limits);self.assertGreater(form,.8);self.assertLess(peak,180.00001)
  self.assertGreater(min(self.d.qpos[self.ids[[2,5]]]),1.2)
 def test_neutral_feet_fail_actual_form_even_with_straight_legs(self):
  self.pose(0);g,(form,limbs,limits,_)=self.metrics()
  self.assertGreater(min(np.degrees(g['footLineAngles'])),80);self.assertFalse(limits);self.assertLess(form,.2)
 def test_each_actual_hand_and_foot_is_independent(self):
  self.pose();g,(_,_,_,_)=self.metrics();j=self.d.qpos[self.ids]
  for key,value in [('footLineAngles',np.radians([0,21])),('handAxisAngles',np.radians([0,31])),('handSeparation',.121),('handHeightGap',.041)]:
   bad={**g,key:value};self.assertFalse(entry_posture(j,bad)[2],key)
 def test_contract_rejects_legacy_and_mismatched_physics_or_actions(self):
  expected=runtime_contract();validate_contract(expected)
  for key,bad in [('xmlSHA256','0'*64),('judgeVersion','v5'),('controlSemantics','old'),('actionHigh',[2.2,2.6,.75,3.14,3.14,1.1,1.1,2.3])]:
   with self.assertRaisesRegex(ValueError,key):validate_contract({**expected,key:bad})
  with self.assertRaises(ValueError):validate_contract(None)
 def test_native_actor_and_export_refuse_old_checkpoint_before_writing_assets(self):
  with tempfile.TemporaryDirectory() as directory:
   old=Path(directory)/'old.json';initial=ROOT/'training/initial-v7.json'
   data=json.loads(initial.read_bytes());data.pop('contract');old.write_text(json.dumps(data))
   with self.assertRaises(ValueError):actor(old)
   _,diagnostic=actor(old,allow_cross_physics=True);self.assertTrue(diagnostic['crossPhysicsBaseline'])
   before=(ROOT/'public/asset-manifest.json').read_bytes()
   with self.assertRaises(ValueError):export(old,initial,'development')
   self.assertEqual(before,(ROOT/'public/asset-manifest.json').read_bytes())
 def test_actual_adduction_reaches_collision_consistent_leg_closure(self):
  self.pose();self.d.qpos[self.ids[-2:]]=0;self.d.ctrl[-2:]=0;mujoco.mj_forward(self.m,self.d)
  start=entry_geometry(self.d.sensordata);self.assertAlmostEqual(start['ankleGap'],.21,places=7)
  # Bounded motor target, no pose writes after this initial condition.
  self.d.ctrl[-2:]=.08;peak=0
  for _ in range(100):
   mujoco.mj_step(self.m,self.d);peak=max(peak,np.max(np.abs(self.d.actuator_force[-2:])));self.assertEqual(self.d.ncon,0)
  g,(_,_,limits,_)=self.metrics();self.assertTrue(limits);self.assertFalse(g['crossedLegs'])
  self.assertLess(g['kneeGap'],.19);self.assertLess(g['ankleGap'],.16);self.assertLess(g['toeGap'],.15);self.assertLessEqual(peak,80)
 def test_crossed_feet_cannot_earn_closure_credit(self):
  self.pose();s=self.d.sensordata.copy();s[16:19],s[19:22]=s[19:22].copy(),s[16:19].copy()
  g=entry_geometry(s);self.assertTrue(g['crossedLegs']);self.assertFalse(entry_posture(self.d.qpos[self.ids],g)[2])
if __name__=='__main__':unittest.main()
