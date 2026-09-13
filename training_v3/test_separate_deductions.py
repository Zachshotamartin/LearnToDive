"""Scoring and physical-entry regressions for independent fault credit."""
import copy
import unittest

from entry_faults import ENTRY_BUDGETS
from judge import judge, terminal_reward
from rules import IDS
from test_judge import measurements
from test_measurements import drop
from training_reward import reward_components


class SeparateDeductionTests(unittest.TestCase):
    def score(self, m):
        return judge(IDS['101C'], 'platform', 10, m)

    def test_every_entry_fault_changes_failed_dive_reward_independently(self):
        for name in ENTRY_BUDGETS:
            with self.subTest(fault=name):
                good = measurements()
                good['rotation'] = 0  # Execution is already zero for all variants.
                previous_reward = terminal_reward(self.score(good), good)
                previous_deductions = self.score(good)['deductions']
                for severity in [.1, 1., 10., 100.]:
                    bad = copy.deepcopy(good)
                    bad['entryFaultLosses'][name] = severity
                    score = self.score(bad)
                    self.assertEqual(score['execution'], 0.)
                    reward = terminal_reward(score, bad)
                    self.assertLess(reward, previous_reward)
                    changed = {key for key in previous_deductions if score['deductions'][key] != previous_deductions[key]}
                    self.assertEqual(changed, {name})
                    previous_reward = reward

    def test_no_aggregate_duplicate_and_exact_reward_reconciliation(self):
        m = measurements()
        m.update(rotation=0, ascent=0, takeoffVerticalSpeed=-1, boardInvalid=True)
        m['entryFaultLosses'] = {key: 2. for key in ENTRY_BUDGETS}
        score = self.score(m)
        self.assertFalse({'takeoff', 'entryForm', 'hands', 'legs', 'feet'} & set(score['deductions']))
        self.assertAlmostEqual(score['rawExecution'], 10 - sum(score['deductions'].values()))
        self.assertAlmostEqual(score['execution'], max(0, score['rawExecution']) - sum(score['scoreAdjustments'].values()))
        for mode in ['v12', 'continuous-entry']:
            ledger = reward_components(score, m, mode)
            self.assertEqual(set(ENTRY_BUDGETS) - set(ledger['deductions']), set())
            self.assertAlmostEqual(ledger['total'], sum(ledger['credits'].values()) - sum(ledger['deductions'].values()))
        # The previous pooled diagnostic is informational, never another cost.
        changed = dict(m, form=0.)
        self.assertEqual(terminal_reward(self.score(changed), changed), terminal_reward(score, m))

    def test_arm_cap_is_reported_and_bad_arms_cannot_be_clean(self):
        m = measurements()
        m['entryArmPositionValid'] = False
        score = self.score(m)
        self.assertEqual(score['execution'], 4.5)
        self.assertEqual(score['scoreAdjustments']['armPositionCap'], 5.5)
        self.assertFalse(score['clean'])

    def test_missing_or_nonfinite_faults_fail_instead_of_silently_passing(self):
        for value in [None, float('nan'), -1.]:
            m = measurements()
            if value is None:
                del m['entryFaultLosses']['entryElbowBend']
            else:
                m['entryFaultLosses']['entryElbowBend'] = value
            with self.assertRaises(ValueError):
                self.score(m)

    def test_physical_arm_placement_survives_full_entry_measurement(self):
        straight = drop(0.)
        arms_down = drop(0., arm_pitch=0.)
        elbows_bent = drop(0., elbow=1.5)
        for item in [straight, arms_down, elbows_bent]:
            self.assertTrue(item['water'])
            self.assertEqual(set(item['entryFaultLosses']), set(ENTRY_BUDGETS))
        self.assertGreater(arms_down['entryFaultLosses']['entryShoulderPitch'],
                           straight['entryFaultLosses']['entryShoulderPitch'] + 1.)
        self.assertGreater(elbows_bent['entryFaultLosses']['entryElbowBend'],
                           straight['entryFaultLosses']['entryElbowBend'] + .5)


if __name__ == '__main__':
    unittest.main()
