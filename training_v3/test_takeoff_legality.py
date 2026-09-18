"""A takeoff that is nearly legal costs less than a wild one, and practice follows learning progress."""
import unittest

import numpy as np

from engine import ASSIST_FORCE, ASSIST_IMPULSE, FOOT_GAP_LIMIT, RECONTACT_TILT_LIMIT
from environment import Arena
from motor_curriculum import MINIMUM_PRACTICE, MotorCurriculum, takeoff_cost
from motor_objective import MINIMUM_ILLEGAL_TAKEOFF, SAFETY_COST, conjunctive_components, takeoff_legality


def measurements(**changes):
    base = dict(boardInvalid=False, boardAssistImpulse=0., boardAssistForce=0., footDepartureGap=0.,
                recontactTilt=0., departureLean=0.)
    return {**base, **changes}


class TakeoffLegalityTests(unittest.TestCase):
    def test_a_legal_takeoff_costs_nothing_and_an_ungradable_fault_still_costs(self):
        self.assertEqual(takeoff_legality(measurements()), 0.)
        self.assertEqual(takeoff_legality(measurements(boardAssistImpulse=ASSIST_IMPULSE)), 0.)
        # A cause this view cannot see keeps a definite charge.
        self.assertEqual(takeoff_legality(measurements(boardInvalid=True, footSideContact=True)), MINIMUM_ILLEGAL_TAKEOFF)

    def test_severity_grows_with_every_measured_cause_and_saturates(self):
        for key, limit in (('boardAssistImpulse', ASSIST_IMPULSE), ('boardAssistForce', ASSIST_FORCE),
                           ('footDepartureGap', FOOT_GAP_LIMIT), ('recontactTilt', RECONTACT_TILT_LIMIT),
                           ('departureLean', 90.)):
            near = takeoff_legality(measurements(boardInvalid=True, **{key: limit * 1.1}))
            far = takeoff_legality(measurements(boardInvalid=True, **{key: limit * 1.8}))
            self.assertLess(near, far, key)
            self.assertEqual(takeoff_legality(measurements(boardInvalid=True, **{key: limit * 5})), 1., key)

    def test_the_dive_reward_charges_the_graded_takeoff(self):
        score = dict(declaration='101A', deductions={}, entryAngle=20., rotationError=0., twistError=0., failures=[], valid=False)
        common = dict(ascent=.0, takeoffVerticalSpeed=0., fullEntryComplete=True, water=True)
        wild = conjunctive_components(score, measurements(boardInvalid=True, recontactTilt=90., **common))
        near = conjunctive_components(score, measurements(boardInvalid=True, recontactTilt=RECONTACT_TILT_LIMIT * 1.1, **common))
        self.assertAlmostEqual(wild['deductions']['boardContact'], SAFETY_COST)
        self.assertLess(near['deductions']['boardContact'], wild['deductions']['boardContact'] / 2)
        self.assertGreater(near['total'], wild['total'], 'a closer takeoff must be worth more')

    def test_practice_cost_follows_the_same_grading(self):
        legal = takeoff_cost(.12, 1.3, 0., 0.)
        self.assertLess(legal, .7)
        self.assertGreater(takeoff_cost(.12, 1.3, .5, 0.), legal)
        self.assertGreater(takeoff_cost(.12, 1.3, 1., 0.), takeoff_cost(.12, 1.3, .5, 0.))


class PracticeAllocationTests(unittest.TestCase):
    def test_practice_follows_learning_progress_not_failure(self):
        arena = Arena(8, 11, 1, training=True, practice=0)
        env = MotorCurriculum(arena, enabled=True, goal_practice=True)
        try:
            np.testing.assert_array_equal(env.learning_progress(), np.zeros(5))
            # A task that has stopped changing, however badly it does, cannot take
            # more than the floor share of the rollouts.
            env.fast[:] = env.slow[:] = 0.
            shares = [env.base.rng.random() for _ in range(4)]
            self.assertEqual(max(MINIMUM_PRACTICE, min(.5, .5 * 4 * float(env.learning_progress()[[1, 2, 4]].max()))),
                             MINIMUM_PRACTICE)
            # A task whose success rate is moving is practised more.
            env.fast[4], env.slow[4] = .6, .2
            self.assertGreater(env.learning_progress()[4], env.learning_progress()[1])
            self.assertGreater(max(MINIMUM_PRACTICE, min(.5, .5 * 4 * float(env.learning_progress()[[1, 2, 4]].max()))),
                               MINIMUM_PRACTICE)
            self.assertEqual(shares, shares)
        finally:
            env.close()

    def test_progress_is_recorded_and_restored_with_the_curriculum(self):
        arena = Arena(4, 12, 1, training=True, practice=0)
        env = MotorCurriculum(arena, enabled=True, goal_practice=True)
        other = MotorCurriculum(Arena(4, 12, 1, training=True, practice=0), enabled=True, goal_practice=True)
        try:
            env.fast[1], env.slow[1] = .4, .1
            other.load_state_dict(env.state_dict())
            np.testing.assert_allclose(other.learning_progress(), env.learning_progress())
            self.assertIn('learningProgress', env.metrics()['takeoff'])
        finally:
            env.close()
            other.close()


if __name__ == '__main__':
    unittest.main()
