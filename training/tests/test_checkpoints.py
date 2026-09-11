"""Actual update-boundary interruption/resume and immutable checkpoint tests."""
import unittest,tempfile,subprocess,sys,os,json,time,signal,hashlib,copy
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'training'))
from checkpoints import requested_interactions,register_evaluation,capture_environment,restore_environment
from sim import Arena

def equal_tree(test,a,b):
 if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  test.assertEqual(a.keys(),b.keys())
  for key in a:equal_tree(test,a[key],b[key])
 elif isinstance(a,(tuple,list)):
  test.assertEqual(len(a),len(b))
  for x,y in zip(a,b):equal_tree(test,x,y)
 else:test.assertEqual(a,b)

class CheckpointTests(unittest.TestCase):
 def command(self,out,*extra):
  return [sys.executable,str(ROOT/'training/train.py'),'--name',str(out),'--envs','4','--threads','1','--horizon','80','--batch','160','--epochs','1','--skills','0','--checkpoint-every-steps','320',*extra]
 def run_training(self,cmd,log):
  with open(log,'w')as output:
   result=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},stdout=output,stderr=subprocess.STDOUT)
  self.assertEqual(result.returncode,0,Path(log).read_text()[-4000:])
 def test_real_update_boundary_resume_matches_uninterrupted_and_extends_completed_target(self):
  with tempfile.TemporaryDirectory()as d:
   d=Path(d);full=d/'full';part=d/'part';continued=d/'continued'
   self.run_training(self.command(full,'--additional-steps','640'),d/'full.log')
   self.run_training(self.command(part,'--additional-steps','320'),d/'part.log')
   parent=part/'latest.pt';before=hashlib.sha256(parent.read_bytes()).hexdigest()
   self.run_training([sys.executable,str(ROOT/'training/train.py'),'--name',str(continued),'--resume',str(parent),'--resume-mode','exact','--target-steps','640'],d/'continued.log')
   a=torch.load(full/'latest.pt',weights_only=False);b=torch.load(continued/'latest.pt',weights_only=False)
   for key in ['model','optimizer','rng','environment']:equal_tree(self,a[key],b[key])
   self.assertEqual(b['steps'],640);self.assertEqual(b['trainingState']['updatesCompleted'],2)
   self.assertGreater(b['trainingState']['cumulativeElapsedSeconds'],b['trainingState']['segmentElapsedSeconds'])
   self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(),before)
   self.assertEqual(len(list((full/'checkpoints').glob('step-*'))),3)
   self.assertEqual(json.loads((continued/'run-status.json').read_text())['status'],'completed-target')
 def test_exact_after_fresh_optimizer_restart_does_not_repeat_segment_operations(self):
  with tempfile.TemporaryDirectory()as d:
   d=Path(d);a=d/'a';b=d/'b';c=d/'c'
   self.run_training(self.command(a,'--additional-steps','320'),d/'a.log')
   self.run_training(self.command(b,'--resume',str(a/'latest.pt'),'--resume-mode','restart','--reset-optimizer','--additional-steps','320'),d/'b.log')
   self.run_training([sys.executable,str(ROOT/'training/train.py'),'--name',str(c),'--resume',str(b/'latest.pt'),'--additional-steps','320'],d/'c.log')
   saved=torch.load(c/'latest.pt',weights_only=False);self.assertFalse(saved['config']['reset_optimizer']);self.assertEqual(saved['steps'],960)
 def test_sigterm_saves_a_complete_update_and_parent_is_not_overwritten(self):
  with tempfile.TemporaryDirectory()as d:
   d=Path(d);run=d/'signal';log=(d/'signal.log').open('w');p=subprocess.Popen(self.command(run,'--additional-steps','100000'),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   try:
    deadline=time.monotonic()+30
    while not(run/'latest.json').exists()and time.monotonic()<deadline:
     if p.poll()is not None:break
     time.sleep(.02)
    self.assertIsNone(p.poll());p.send_signal(signal.SIGTERM);self.assertEqual(p.wait(timeout=30),0)
    status=json.loads((run/'run-status.json').read_text());self.assertEqual(status['status'],'interrupted-at-update-boundary');self.assertEqual(status['steps']%320,0)
    self.assertGreaterEqual(status['steps'],320)
   finally:
    if p.poll()is None:p.kill();p.wait()
    log.close()
 def test_targets_are_additional_or_explicit_cumulative_at_saved_boundaries(self):
  self.assertEqual(requested_interactions(640,None,9000,320),640)
  self.assertEqual(requested_interactions(None,9640,9000,320),640)
  self.assertEqual(requested_interactions(1,None,9000,320),320)
  with self.assertRaises(ValueError):requested_interactions(None,9000,9000,320)
  with self.assertRaises(ValueError):requested_interactions(320,9320,9000,320)
 def test_actual_frozen_baseline_can_remain_best_after_a_worse_evaluation(self):
  base=ROOT/'training/diagnostics/v8-matched-head-zero';failed=ROOT/'training/diagnostics/v8-matched-head-first-2048k'
  if not base.exists():self.skipTest('Local preserved regression evidence not part of the portable test fixture')
  with tempfile.TemporaryDirectory()as d:
   self.assertTrue(register_evaluation(d,base));before=Path(d,'best.json').read_bytes()
   self.assertFalse(register_evaluation(d,failed));self.assertEqual(Path(d,'best.json').read_bytes(),before)

if __name__=='__main__':unittest.main()
