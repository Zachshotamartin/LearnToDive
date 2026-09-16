"""The conjunctive credit: rotation only pays through a vertical entry; every phase keeps its slope."""
import unittest

import numpy as np

from judge import judge
from motor_objective import conjunctive_components, entry_quality, rotation_progress, SAFETY_COST, COMPLETION_BONUS
from rules import IDS
from test_judge import measurements


def reward(dive='101A', **changes):
    m = measurements()
    m.update(changes)
    if 'entryAngle' in changes and 'firstContactAngle' not in changes:
        m['firstContactAngle'] = changes['entryAngle']
    score = judge(IDS[dive], 'platform', 10, m)
    return conjunctive_components(score, m, armstand=False), score


class ConjunctiveRewardTests(unittest.TestCase):
    def test_entry_quality_scale(self):
        self.assertEqual(entry_quality(0), 1.)
        self.assertAlmostEqual(entry_quality(20), .5)
        self.assertGreater(entry_quality(180), 0.)

    def test_flat_landing_spinner_earns_less_than_a_vertical_half_somersault(self):
        spinner, _ = reward(entryAngle=85)
        vertical, _ = reward(entryAngle=5)
        self.assertLess(spinner['total'], vertical['total'])
        self.assertLess(spinner['credits']['rotationThroughEntry'], .3 * vertical['credits']['rotationThroughEntry'])

    def test_credit_is_monotone_in_entry_angle_and_rotation_error(self):
        totals = [reward(entryAngle=angle)[0]['total'] for angle in (0, 10, 20, 45, 87, 150)]
        self.assertEqual(totals, sorted(totals, reverse=True))
        by_rotation = [reward(rotation=r)[0]['total'] for r in (.5, .45, .35, .2, 0.)]
        self.assertEqual(by_rotation, sorted(by_rotation, reverse=True))
        self.assertAlmostEqual(rotation_progress(reward(rotation=.5)[1]), 1.)
        self.assertLess(rotation_progress(reward(rotation=-.5)[1]), 0.)

    def test_a_jump_beats_a_fall_at_equal_entry(self):
        jump, _ = reward()
        fall, _ = reward(ascent=0, takeoffVerticalSpeed=-1.5, departureLean=60)
        self.assertGreater(jump['total'] - fall['total'], 3.)

    def test_a_missing_takeoff_costs_as_much_as_a_thirty_degree_entry(self):
        perfect, _ = reward()
        fall, _ = reward(ascent=0, takeoffVerticalSpeed=-1.5, departureLean=60)
        tilted, _ = reward(entryAngle=30)
        self.assertGreaterEqual(perfect['total'] - fall['total'], .8 * (perfect['total'] - tilted['total']))
        # From a vertical fall, a jump that still lands thirty degrees off must already pay more.
        fall_vertical, _ = reward(ascent=0, takeoffVerticalSpeed=-1.5, departureLean=60, entryAngle=0)
        jump_tilted, _ = reward(entryAngle=30)
        self.assertGreater(jump_tilted['total'], fall_vertical['total'])

    def test_physical_failures_cost_more_than_the_maximum_credit(self):
        self.assertGreaterEqual(SAFETY_COST, COMPLETION_BONUS)
        missed, _ = reward(water=False, fullEntryComplete=False)
        best, _ = reward()
        self.assertLess(missed['total'], best['total'] - 2 * SAFETY_COST)
        self.assertGreater(best['total'], 10.)
        self.assertLessEqual(best['credits']['rotationThroughEntry'], COMPLETION_BONUS)

    def test_alignment_keeps_its_slope_past_the_judge_cap(self):
        sixty, _ = reward(entryAngle=60)
        ninety, _ = reward(entryAngle=90)
        self.assertGreater(ninety['deductions']['entryAlignment'], sixty['deductions']['entryAlignment'])


if __name__ == '__main__':
    unittest.main()
