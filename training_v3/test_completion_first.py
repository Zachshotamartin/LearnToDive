"""Behavioral guarantees for completion-first rewards and safe continuation."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from judge import judge
from rules import DIVES, IDS
from test_judge import measurements
from test_direction_practice import ContinuationTests
from training_reward import reward_components
from train import Trainer, train


def target(code='5132D'):
    d = DIVES[IDS[code]]
    return dict(measurements(), rotation=d['sign'] * d['turns'], twist=d['twists'],
                firstGeometry=6 if d['headfirst'] else 12,
                positionRecognition=dict(samples=30, fractions=[float(p == d['position']) for p in 'ABC']))


def ledger(m, code='5132D'):
    score = judge(IDS[code], 'platform', 10, m)
    before = copy.deepcopy(score)
    result = reward_components(score, m, 'completion-first')
    assert score == before
    return result


class CompletionRewardTests(unittest.TestCase):
    def test_bad_form_correct_position_completion_beats_perfect_form_wrong_dive(self):
        for code in ('103C', '203C', '303C', '403C', '5132D', '612C'):
            m = target(code)
            bad = copy.deepcopy(m)
            bad.update(entryAngle=175, firstContactAngle=175, positionQuality=0,
                       ascent=0, takeoffVerticalSpeed=-20, departureLean=180, surfaceLateralSpeed=100)
            bad['entryFaultLosses'] = {k: 1e12 for k in m['entryFaultLosses']}
            wrong = dict(m, rotation=0, twist=0)
            good = ledger(m, code)
            completed = ledger(bad, code)
            failed = ledger(wrong, code)
            self.assertEqual(completed['stage'], 'execution-refinement')
            self.assertGreater(completed['total'], failed['total'] + 10)
            self.assertLess(completed['executionDeductionTotal'], 4)
            self.assertGreater(good['total'], completed['total'])
            self.assertEqual(failed['credits']['rotationProgress'], 0)

    def test_more_accurate_flips_and_twists_help_before_completion(self):
        for key in ('rotation', 'twist'):
            values = []
            for count in (0, .2, .4):
                m = target()
                m[key] = count
                values.append(ledger(m)['total'])
            self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
        m = target()
        baseline = ledger(m)['total']
        self.assertLess(ledger(dict(m, rotation=m['rotation']+1))['total'], baseline)
        self.assertLess(ledger(dict(m, twist=m['twist']+1))['total'], baseline)
        self.assertLess(ledger(dict(m, rotation=-m['rotation']))['total'], baseline)

    def test_execution_faults_stay_independent_even_for_failed_dives(self):
        for rotation in (0, 1.5):
            m = target()
            m['rotation'] = rotation
            baseline = ledger(m)
            for key in m['entryFaultLosses']:
                bad = copy.deepcopy(m)
                bad['entryFaultLosses'][key] = .5
                first = ledger(bad)
                bad['entryFaultLosses'][key] = 5
                worse = ledger(bad)
                self.assertGreater(baseline['total'], first['total'])
                self.assertGreater(first['total'], worse['total'])
                for other in baseline['deductions']:
                    if other != key:
                        self.assertEqual(baseline['deductions'][other], worse['deductions'][other])
                self.assertAlmostEqual(worse['total'], sum(worse['credits'].values())-sum(worse['deductions'].values()))

    def test_crashing_never_claims_completed_difficulty(self):
        for fault in ({'water': False}, {'boardInvalid': True}, {'fullEntryComplete': False}):
            m = target()
            m.update(fault)
            row = ledger(m)
            self.assertLessEqual(sum(row['credits'].values()), 16)
            self.assertLess(row['total'], 0)

    def test_no_merely_declared_difficulty_credit(self):
        # The same stationary attempt cannot earn a high DD bonus by naming it.
        m = dict(target(), rotation=0, twist=0)
        self.assertEqual(ledger(m)['credits'], {'rotationProgress': 0.})
        completed = ledger(target())
        self.assertGreater(completed['credits']['rotationProgress'], completed['executionBudget'] * 3)


class CompletionContinuationTests(unittest.TestCase):
    def test_preserves_model_optimizer_counters_and_motor_readiness(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            old_args = ContinuationTests().args(root/'old', 480, direction=True)
            train(old_args)
            old = torch.load(root/'old/latest.pt', weights_only=False)
            a = ContinuationTests().args(root/'new', direction=True)
            a.reward_mode = 'completion-first'
            a.continue_from = str(root/'old/latest.pt')
            t = Trainer(a)
            try:
                self.assertEqual(t.state['steps'], old['training']['steps'])
                self.assertEqual(t.state['updates'], old['training']['updates'])
                for k, v in old['model'].items():
                    torch.testing.assert_close(t.policy.state_dict()[k], v, rtol=0, atol=0)
                for k, vs in old['optimizer']['state'].items():
                    for name, v in vs.items():
                        torch.testing.assert_close(t.optimizer.state_dict()['state'][k][name], v, rtol=0, atol=0)
                for name in ('visits', 'readiness', 'level', 'direction_readiness', 'goal_visits'):
                    np.testing.assert_array_equal(getattr(t.env, name), old['environment']['curriculum'][name])
                self.assertTrue(t.env.choosing.all())
                self.assertFalse(t.env.goal_mastery.any())
                self.assertFalse(t.env.base.phase_previous.any())
                self.assertFalse(any(t.env.base.bank))
                torch.testing.assert_close(torch.get_rng_state(), old['torchRNG'], rtol=0, atol=0)
                before = copy.deepcopy(t.policy.state_dict())
                t.update()
                self.assertEqual(t.state['steps'], 640)
                for k, v in before.items():
                    if not k.startswith(('value', 'critic_trunk.', 'input_')):
                        torch.testing.assert_close(t.policy.state_dict()[k], v, rtol=0, atol=0)
                self.assertEqual(t.state['criticWarmupRemaining'], 3)
            finally:
                t.env.close()

    def test_exact_resume_of_new_objective_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            def args(name, steps):
                a = ContinuationTests().args(root/name, steps, direction=True)
                a.reward_mode = 'completion-first'
                return a
            train(args('whole', 640))
            train(args('split', 320))
            a = args('split', 320)
            a.resume = str(root/'split/latest.pt')
            train(a)
            x, y = [torch.load(root/n/'latest.pt', weights_only=False) for n in ('whole','split')]
            for k, v in x['model'].items():
                torch.testing.assert_close(v, y['model'][k], rtol=0, atol=0)

    def test_refuses_architecture_or_physics_changes(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            train(ContinuationTests().args(root/'old', 160, direction=True))
            original = torch.load(root/'old/latest.pt', weights_only=False)
            for file in ('diver.xml', 'policy.py', 'judge.py'):
                bad = copy.deepcopy(original)
                bad['contract']['sourceHashes'][file] = 'incompatible'
                torch.save(bad, root/'bad.pt')
                a = ContinuationTests().args(root/('new-'+file), direction=True)
                a.reward_mode = 'completion-first'
                a.continue_from = str(root/'bad.pt')
                with self.assertRaises(ValueError):
                    Trainer(a)


if __name__ == '__main__':
    unittest.main()
