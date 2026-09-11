import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewards import whole_entry_terms

def merit(angle,form,geometry):
 t=whole_entry_terms(angle,form,geometry)
 return 24*t['quality']-t['angleCost']-t['formCost']-t['geometryCost']
class RewardTests(unittest.TestCase):
 def test_clean_entry_beats_partial_shape_or_tilt_regardless_of_duration(self):
  clean=merit(5,.95,.05)
  self.assertGreater(clean,merit(42,.95,.05));self.assertGreater(clean,merit(5,.1,3.6))
 def test_form_savings_cannot_mask_known_large_angle_regression(self):
  self.assertGreater(merit(35,.1,3.6),merit(42,.2,2.8))
 def test_components_are_monotone_and_geometry_is_bounded(self):
  angles=np.array([0,10,35,60,90]);terms=whole_entry_terms(angles,np.ones(5),np.zeros(5))
  self.assertTrue(np.all(np.diff(terms['angleCost'])>0));self.assertTrue(np.all(np.diff(terms['quality'])<0))
  self.assertEqual(whole_entry_terms(0,1,100)['geometryCost'],8)
  self.assertEqual(whole_entry_terms(0,1,0)['geometryCost'],0)
if __name__=='__main__':unittest.main()
