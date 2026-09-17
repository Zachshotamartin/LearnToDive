"""The takeoff practice task starts as a small hop and widens toward the competition takeoff."""
import unittest

import numpy as np

from environment import Arena
from motor_curriculum import (BOARD_INVALID_COST, MotorCurriculum, TAKEOFF_LEAN_WEIGHT, TAKEOFF_RISE, TAKEOFF_SPEED, takeoff_cost,
                              takeoff_lean, takeoff_targets)


class TakeoffLadderTests(unittest.TestCase):
    def test_targets_interpolate_from_a_hop_to_the_competition_takeoff(self):
        self.assertEqual(takeoff_targets(0.), (TAKEOFF_RISE[0], TAKEOFF_SPEED[0]))
        self.assertEqual(takeoff_targets(1.), (TAKEOFF_RISE[1], TAKEOFF_SPEED[1]))
        self.assertEqual(takeoff_targets(.5), (.2, 2.))
        self.assertEqual(takeoff_targets(2.), (TAKEOFF_RISE[1], TAKEOFF_SPEED[1]))

    def test_a_small_hop_passes_the_first_rung_and_fails_the_last(self):
        hop = takeoff_cost(.12, 1.3, False, 0.)
        self.assertLess(hop, .7)
        self.assertGreater(takeoff_cost(.12, 1.3, False, 1.), .7)
        self.assertAlmostEqual(float(takeoff_cost(.3, 2.8, False, 1.)), 0.)
        self.assertAlmostEqual(float(takeoff_cost(0., 0., False, 0.)), 1.5)
        self.assertAlmostEqual(float(takeoff_cost(.3, 2.8, True, 1.)), BOARD_INVALID_COST)
        np.testing.assert_allclose(takeoff_cost([.1, .3], [1.2, 2.8], [False, False], .5), [.5 + .5 * .4, 0.])

    def test_leaning_off_the_edge_costs_more_the_further_past_vertical_it_goes(self):
        upright = takeoff_cost(.12, 1.3, False, 0., lean=np.radians(20))
        self.assertLess(upright, .7, 'a slightly leaning valid hop still passes the first rung')
        self.assertAlmostEqual(float(upright), TAKEOFF_LEAN_WEIGHT * 20 / 90)
        rolled = takeoff_cost(0., -.7, True, 0., lean=np.radians(100))
        self.assertGreater(rolled, takeoff_cost(0., -.7, True, 0., lean=np.radians(60)))
        self.assertGreater(takeoff_cost(0., -.7, True, 0., lean=np.radians(60)), takeoff_cost(0., -.7, True, 0., lean=0.))
        self.assertAlmostEqual(float(takeoff_cost(0., 0., False, 0., lean=np.pi)), 1.5 + 2 * TAKEOFF_LEAN_WEIGHT)

    def test_live_costs_follow_the_current_level(self):
        arena = Arena(4, 7, 1, training=False, practice=0)
        env = MotorCurriculum(arena, enabled=False)
        try:
            physics = env.base.physics
            env.task[:] = 1
            physics.armstand[:] = False
            physics.board_invalid[:] = False
            physics.departure_com[:] = 0.
            physics.apex_com[:] = .12
            physics.takeoff_vertical_speed[:] = 1.3
            physics.released[:] = True
            physics.departure_pitch[:] = np.radians(10)
            physics.prev_pitch[:] = np.radians(170)
            np.testing.assert_allclose(takeoff_lean(physics), np.radians(10))
            env.level[1] = 0.
            self.assertTrue((env.costs() < .7).all())
            env.level[1] = 1.
            self.assertTrue((env.costs() > .7).all())
            physics.released[:] = False
            np.testing.assert_allclose(takeoff_lean(physics), np.radians(170))
            env.level[1] = 0.
            self.assertTrue((env.costs() > .7).all(), 'still on the board and rolled past horizontal')
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
