"""Rules table and judge behaviour on synthetic measurements."""
import unittest

from entry_faults import ENTRY_BUDGETS

from judge import judge, terminal_reward
from rules import DIVES, IDS, difficulty, legal_mask


def measurements():
    """A perfect forward dive as the engine would report it."""
    return dict(rotation=.5, twist=0, boardInvalid=False, water=True, fullEntryComplete=True, firstGeometry=6, x=1,
                entryFaultLosses={name: 0. for name in ENTRY_BUDGETS}, entryArmPositionValid=True,
                maxLateral=0, firstContactAngle=0, entryAngle=0, ascent=.4, preparationBounces=0, positionQuality=1,
                form=1, entryGeometryWorst=dict(footLineAngles=[0, 0], ankleGap=.12, crossedLegs=False,
                                                handSeparation=.07, handHeightGap=0),
                surfaceLateralSpeed=0, entryGeometryValid=True, entryLimbsValid=True, departureLean=5, takeoffVerticalSpeed=2.5)


class RulesTests(unittest.TestCase):
    def test_table(self):
        self.assertEqual(difficulty(IDS['101C'], 'springboard', 3), 1.4)
        self.assertEqual(difficulty(IDS['103B'], 'platform', 10), 1.6)
        self.assertEqual(difficulty(IDS['5132D'], 'springboard', 3), 2.1)
        with self.assertRaises(ValueError):
            difficulty(IDS['101C'], 'platform', 6)

    def test_groups_and_repeat(self):
        self.assertEqual(DIVES[IDS['301C']]['sign'], -1)
        self.assertEqual(DIVES[IDS['401C']]['back'], 1)
        self.assertFalse(legal_mask(1, 'platform', 10, [101])[IDS['101C']])
        self.assertFalse(legal_mask(6, 'springboard', 3).any())

    def test_every_apparatus_category_has_legal_choices(self):
        for apparatus, heights, groups in [('springboard', [1, 3], range(1, 6)), ('platform', [5, 7.5, 10], range(1, 7))]:
            for height in heights:
                for group in groups:
                    self.assertTrue(legal_mask(group, apparatus, height).any(), (apparatus, height, group))


class JudgeTests(unittest.TestCase):
    def test_tail_flop_and_wrong_declaration(self):
        m = measurements()
        good = judge(IDS['101C'], 'platform', 10, m)
        self.assertEqual(good['execution'], 10)
        m['entryAngle'] = 65
        bad = judge(IDS['101C'], 'platform', 10, m)
        self.assertFalse(bad['clean'])
        self.assertLess(bad['execution'], good['execution'])
        m = measurements()
        wrong = judge(IDS['103C'], 'platform', 10, m)
        self.assertEqual(wrong['points'], 0)
        self.assertLess(terminal_reward(wrong, m), 0)

    def test_training_reward_stays_dense_when_execution_is_clipped(self):
        base = measurements()
        base.update(ascent=0, positionQuality=0, form=0, entryAngle=40, firstContactAngle=40)
        base['entryFaultLosses'].update({name: 4. for name in ENTRY_BUDGETS})
        better = judge(IDS['101C'], 'platform', 10, base)
        self.assertEqual(better['execution'], 0)
        m = measurements()
        m.update(ascent=0, positionQuality=0, form=0, entryAngle=40, firstContactAngle=40, surfaceLateralSpeed=3)
        m['entryFaultLosses'].update({name: 4. for name in ENTRY_BUDGETS})
        m['entryFaultLosses']['entryHandSeparation'] = 8.
        worse = judge(IDS['101C'], 'platform', 10, m)
        self.assertEqual(worse['execution'], 0)
        self.assertGreater(terminal_reward(better, base), terminal_reward(worse, m))
        invalid = dict(base, water=False)
        self.assertLess(terminal_reward(judge(IDS['101C'], 'platform', 10, invalid), invalid), terminal_reward(better, base))
        perfect = measurements()
        self.assertGreater(terminal_reward(judge(IDS['101C'], 'platform', 10, perfect), perfect), terminal_reward(better, base))

    def test_no_failed_reward_saturation(self):
        m = measurements()
        m['rotation'] = .1
        short = judge(IDS['103C'], 'platform', 10, m)
        reward = terminal_reward(short, m)
        m['entryAngle'] = 80
        flat = judge(IDS['103C'], 'platform', 10, m)
        self.assertLess(terminal_reward(flat, m), reward)

    def test_distance_is_a_deduction_and_only_an_unsafe_dive_is_capped(self):
        m = measurements()
        m['x'] = .3
        score = judge(IDS['101C'], 'platform', 10, m)
        self.assertTrue(score['valid'])
        self.assertAlmostEqual(score['deductions']['distance'], 1.2)
        self.assertGreater(score['execution'], 2)
        m['x'] = .1
        score = judge(IDS['101C'], 'platform', 10, m)
        self.assertTrue(score['valid'])
        self.assertLessEqual(score['execution'], 2)

    def test_takeoff_height_speed_and_lean_change_the_reward_at_equal_entry(self):
        jump = measurements()
        fall = measurements()
        fall.update(ascent=0, takeoffVerticalSpeed=-1.5, departureLean=60)
        jumped = judge(IDS['101C'], 'platform', 10, jump)
        fell = judge(IDS['101C'], 'platform', 10, fall)
        names = ['takeoffHeight', 'takeoffLean', 'preparationBounces']
        self.assertGreater(sum(fell['deductions'][k] for k in names), sum(jumped['deductions'][k] for k in names) + 2)
        # A missing takeoff must cost at least as much as a thirty-degree entry error;
        # before v12 it cost less than a fifth of one.
        flat = measurements()
        flat.update(entryAngle=30, firstContactAngle=30)
        self.assertGreaterEqual(terminal_reward(jumped, jump) - terminal_reward(fell, fall),
                                terminal_reward(jumped, jump) - terminal_reward(judge(IDS['101C'], 'platform', 10, flat), flat))

    def test_position_deduction_matches_rule_range(self):
        m = measurements()
        m['positionQuality'] = .5
        self.assertAlmostEqual(judge(IDS['101C'], 'platform', 10, m)['deductions']['position'], 1.)
        m['positionQuality'] = 0.
        self.assertAlmostEqual(judge(IDS['101C'], 'platform', 10, m)['deductions']['position'], 2.)


if __name__ == '__main__':
    unittest.main()
