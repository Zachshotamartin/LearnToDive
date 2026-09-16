"""Limits, continuous feedback, and a lossless transition into the new cohort."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rules import DIVES, IDS, in_practice_scope, legal_mask
from motor_curriculum import MotorCurriculum
from environment import Arena
from test_completion_first import target, ledger
from test_rotation_progress import RotationProgressTests
from train import Trainer


class BoundedFeedbackTests(unittest.TestCase):
    def test_all_conditions_and_nonrepeat_choices(self):
        self.assertTrue(in_practice_scope(DIVES[IDS['614C']]))
        self.assertFalse(in_practice_scope(DIVES[IDS['105C']]))
        self.assertTrue(any(d['twists'] > 1 and not in_practice_scope(d) for d in DIVES))
        for apparatus, heights in [('platform', [5, 7.5, 10]), ('springboard', [1, 3])]:
            for height in heights:
                for group in range(1, 7 if apparatus == 'platform' else 6):
                    choices = np.flatnonzero(legal_mask(group, apparatus, height))
                    self.assertGreater(len(choices), 0)
                    for j in choices:
                        d = DIVES[j]
                        self.assertLessEqual(d['turns'], 2 if group == 6 else 1.5)
                        self.assertLessEqual(d['twists'], 1)
                        repeated = legal_mask(group, apparatus, height, [d['code']])
                        self.assertFalse(any(repeated[k] for k, row in enumerate(DIVES) if row['code'] == d['code']))

    def test_continuity_through_judge_threshold_and_pose_recognition(self):
        for code in ('103C', '203C', '303C', '403C', '5132D', '614C'):
            m = target(code)
            sign = DIVES[IDS[code]]['sign']
            # Numeric valid changes at .25; motor reward must not jump there.
            a = ledger(dict(m, rotation=m['rotation']-sign*(.25+1e-7)), code)
            b = ledger(dict(m, rotation=m['rotation']-sign*(.25-1e-7)), code)
            self.assertGreater(b['total'], a['total'])
            self.assertLess(b['total']-a['total'], 1e-4)
            unrecognized = copy.deepcopy(m)
            unrecognized['positionRecognition']['fractions'] = [0, 0, 0]
            self.assertEqual(ledger(m, code)['total'], ledger(unrecognized, code)['total'])
            # Independent smooth position deductions still penalize poor poses.
            unrecognized['motorPositionQuality'] = 0
            self.assertLess(ledger(unrecognized, code)['total'], ledger(m, code)['total'])

    def test_moving_toward_each_target_beats_not_rotating(self):
        for code in ('103C', '303C', '5132D', '614C'):
            d = DIVES[IDS[code]]
            scores = [ledger(dict(target(code), rotation=d['sign']*d['turns']*fraction,
                                  twist=d['twists']*fraction), code)['total']
                      for fraction in (0, .01, .1, .5, .9, 1.)]
            self.assertTrue(all(a < b for a,b in zip(scores,scores[1:])))
            for sign in (-1, 1):
                self.assertAlmostEqual(ledger(dict(target('5132D'),twist=sign*.7))['total'],
                                       ledger(dict(target('5132D'),twist=.7))['total'])

    def test_goal_practice_excludes_stationary_air_task(self):
        env = MotorCurriculum(Arena(16, 18, 1, rotation_progress=True), goal_practice=True, direction_practice=True)
        try:
            seen = set()
            for _ in range(50):
                env.assign(np.arange(env.n))
                seen.update(env.task)
            self.assertEqual(seen, {0, 1, 2, 4})
        finally:
            env.close()

    def test_fresh_random_weights_zero_progress_then_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            args=RotationProgressTests().args(root/'new',True)
            args.random_motor_init=True
            new=Trainer(args)
            try:
                self.assertEqual(new.state['steps'],0)
                self.assertEqual(new.state['updates'],0)
                self.assertFalse(new.optimizer.state)
                self.assertFalse(new.env.goal_visits.any())
                self.assertFalse(new.env.visits.any())
                self.assertFalse(new.env.goal_mastery.any())
                for _ in range(5):new.update()
                latest=copy.deepcopy(new.persist('paused'))
                self.assertGreater(new.state['steps'],0)
            finally:
                new.env.close()
            a=RotationProgressTests().args(root/'new',True)
            a.random_motor_init=True; a.resume=str(root/'new/latest.pt')
            resumed=Trainer(a)
            try:
                self.assertEqual(resumed.state['steps'],latest['training']['steps'])
                for k,v in latest['model'].items():
                    torch.testing.assert_close(resumed.policy.state_dict()[k],v,rtol=0,atol=0)
                for pid,values in latest['optimizer']['state'].items():
                    for key,v in values.items():
                        torch.testing.assert_close(resumed.optimizer.state_dict()['state'][pid][key],v,rtol=0,atol=0)
                self.assertFalse(np.isnan(resumed.env.observe()).any())
            finally:
                resumed.env.close()

if __name__=='__main__': unittest.main()
