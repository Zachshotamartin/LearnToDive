import contextlib,io,tempfile,unittest
from pathlib import Path
import torch
from train import train,parser
class TrainingTests(unittest.TestCase):
 def test_exact_resume_and_finite_gradients(self):
  with tempfile.TemporaryDirectory() as folder:
   def args(name,steps,resume=None):
    a=parser().parse_args(['--output',str(Path(folder)/name),'--envs','4','--threads','1','--horizon','32','--widths','32','32','--batch','64','--epochs','1','--steps',str(steps),'--evaluate-every','0','--archive-every','1']);a.resume=resume;return a
   with contextlib.redirect_stdout(io.StringIO()):
    train(args('whole',512));train(args('split',256));train(args('split',256,str(Path(folder)/'split/latest.pt')))
   a=torch.load(Path(folder)/'whole/latest.pt',weights_only=False);b=torch.load(Path(folder)/'split/latest.pt',weights_only=False)
   for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],atol=0,rtol=0)
   self.assertEqual(a['training']['steps'],b['training']['steps'])
 def test_every_apparatus_category_has_legal_choices(self):
  from rules import legal_mask
  for app,heights,groups in [('springboard',[1,3],range(1,6)),('platform',[5,7.5,10],range(1,7))]:
   for h in heights:
    for g in groups:self.assertTrue(legal_mask(g,app,h).any(),(app,h,g))
if __name__=='__main__':unittest.main()

class LossTests(unittest.TestCase):
 def test_ppo_terms_weight_every_decision_once(self):
  from losses import ppo_terms
  n=100;logp=torch.zeros(n);old=torch.zeros(n);adv=torch.ones(n);choosing=torch.zeros(n,dtype=torch.bool);choosing[:2]=True
  selected=torch.ones(n,dtype=torch.bool);entropy=torch.cat([torch.full((2,),3.),torch.full((98,),-4.)])
  loss,kl=ppo_terms(logp,old,adv,choosing,selected,entropy,motor_entropy=.006,declaration_entropy=.01)
  expected=-1+ (.006*(-4.)*98+.01*3.*2)/n*-1  # surrogate mean of -1 minus the two entropy bonuses
  self.assertAlmostEqual(float(loss),-1-(.006*(-4.)*98+.01*3.*2)/n,places=6);self.assertEqual(float(kl),0.)
  adv=torch.cat([torch.full((2,),50.),torch.ones(98)])
  loss,_=ppo_terms(logp,old,adv,choosing,selected,entropy,motor_entropy=0,declaration_entropy=0)
  self.assertAlmostEqual(float(loss),-(50.*2+98)/n,places=5)  # per-sample, not one half per group
  loss,kl=ppo_terms(logp,old,adv,choosing,torch.zeros(n,dtype=torch.bool),entropy)
  self.assertEqual(float(loss),0.);self.assertEqual(float(kl),0.)
 def test_training_sources_exclude_tooling_and_cover_the_learning_problem(self):
  from train import hashes
  from engine import TRAINING_SOURCES
  names=set(hashes())
  self.assertEqual(names,set(TRAINING_SOURCES))
  for tooling in ['run_suite.py','audit_physics.py','check_browser_parity.py','extract_difficulty.py']:self.assertNotIn(tooling,names)
  for essential in ['engine.py','water.py','judge.py','rules.py','policy.py','losses.py','train.py','environment.py','difficulty.json','diver.xml']:self.assertIn(essential,names)
