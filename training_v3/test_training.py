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
