"""Independent invariants for the alternative training signal."""
import unittest
from judge import judge, terminal_reward
from rules import IDS
from test_judge import measurements
from training_reward import alignment_cost, reward_components


class RewardComparisonTests(unittest.TestCase):
    def test_bad_angles_keep_an_incentive_without_unbounded_cost(self):
        costs = [alignment_cost(a) for a in [0, 30, 60, 80, 90, 120, 180]]
        self.assertTrue(all(a < b for a, b in zip(costs, costs[1:])))
        self.assertEqual(costs[0], 0)
        self.assertEqual(costs[-1], 1.5)

    def test_failed_dives_ordered_and_judge_unchanged(self):
        values = []
        for angle in [80, 100, 120]:
            m = measurements()
            m.update(rotation=0, firstContactAngle=angle, entryAngle=angle)
            score = judge(IDS['101C'], 'platform', 10, m)
            before = repr(score)
            self.assertEqual(score['execution'], 0)
            self.assertAlmostEqual(reward_components(score, m)['total'], terminal_reward(score, m))
            values.append(reward_components(score, m, 'continuous-entry')['total'])
            self.assertEqual(repr(score), before)
        self.assertTrue(values[0] > values[1] > values[2])
        self.assertTrue(all(v <= 0 for v in values))

    def test_takeoff_and_position_remain_independently_valuable(self):
        m = measurements()
        m.update(rotation=0, entryAngle=100, firstContactAngle=100)
        def reward(row):
            return reward_components(judge(IDS['101C'], 'platform', 10, row), row, 'continuous-entry')['total']
        self.assertGreater(reward(m), reward(dict(m, ascent=0, takeoffVerticalSpeed=-1)))
        self.assertGreater(reward(m), reward(dict(m, positionQuality=0)))
