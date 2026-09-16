"""Mastery stages: scope masks, routines inside the scope, the widening rule and the stance width."""
import unittest

import numpy as np

import mastery
from environment import Arena
from rules import DIVES, IDS


class ScopeTests(unittest.TestCase):
    def test_first_stage_scope(self):
        scope = mastery.stage(0)
        allowed = mastery.scope_mask(scope, DIVES)
        self.assertTrue(allowed[IDS['101A']])
        self.assertTrue(allowed[IDS['201C']])
        self.assertFalse(allowed[IDS['103C']], 'one and a half somersaults are beyond the first stage')
        self.assertFalse(allowed[IDS['401A']], 'inward dives are beyond the first stage')
        self.assertFalse(allowed[IDS['101B']], 'pike is beyond the first stage')
        self.assertEqual(mastery.routine_groups(scope, 1), [1, 2])
        self.assertEqual(mastery.heights_for(scope, 1), (10,))

    def test_final_stage_covers_everything_the_practice_scope_allows(self):
        scope = mastery.stage(mastery.FINAL_STAGE)
        self.assertTrue(mastery.scope_mask(scope, DIVES)[IDS['5132D']])
        self.assertTrue(mastery.scope_mask(scope, DIVES)[IDS['624C']])
        self.assertEqual(mastery.routine_groups(scope, 0), [1, 2, 3, 4, 5])
        self.assertEqual(scope['spread'], 1.)

    def test_progression_advances_after_consecutive_clean_evaluations_only(self):
        p = mastery.Progression(0)
        self.assertFalse(p.observe(.4, 100))
        self.assertFalse(p.observe(.4, 200))
        self.assertFalse(p.observe(.1, 300), 'a miss resets the streak')
        self.assertFalse(p.observe(.4, 400))
        self.assertFalse(p.observe(.4, 500))
        self.assertTrue(p.observe(.4, 600))
        self.assertEqual(p.index, 1)
        self.assertEqual(p.history[-1]['steps'], 600)
        p = mastery.Progression(mastery.FINAL_STAGE)
        for steps in range(5):
            self.assertFalse(p.observe(1., steps))
        self.assertEqual(p.index, mastery.FINAL_STAGE)
        with self.assertRaises(ValueError):
            mastery.stage(mastery.FINAL_STAGE + 1)


class StagedArenaTests(unittest.TestCase):
    def test_routines_and_masks_stay_inside_the_stage(self):
        e = Arena(6, seed=9, threads=1, stage=0)
        try:
            self.assertTrue((e.apparatus == 1).all())
            self.assertTrue((e.height == 10).all())
            self.assertTrue(np.isin(e.schedule[:, :2], [1, 2]).all())
            self.assertTrue((e.routine_length == 2).all())
            self.assertEqual(e.physics.stance_spread, .25)
            self.assertEqual(e.auxiliary_rate, .6)
            allowed = mastery.scope_mask(mastery.stage(0), DIVES)
            self.assertTrue((e.mask() <= allowed[None]).all())
            self.assertTrue(e.mask().any(axis=1).all(), 'every athlete keeps a legal declaration')
            e.set_stage(2)
            self.assertEqual(e.physics.stance_spread, .75)
            state = e.state_dict()
            self.assertEqual(state['stage'], 2)
            e.set_stage(0)
            e.load_state_dict(state)
            self.assertEqual(e.stage_index, 2)
        finally:
            e.close()

    def test_no_stage_keeps_the_full_scope(self):
        e = Arena(4, seed=3, threads=1)
        try:
            self.assertIsNone(e.stage_index)
            self.assertEqual(e.physics.stance_spread, 1.)
            self.assertTrue((e.routine_length == 6).all())
        finally:
            e.close()


if __name__ == '__main__':
    unittest.main()
