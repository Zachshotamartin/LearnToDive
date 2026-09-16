"""Position quality must keep a slope toward the shape even far from it."""
import unittest

import numpy as np

from positions import position_qualities


class PositionTests(unittest.TestCase):
    def test_peaks_at_each_shape_and_never_flattens(self):
        hips = np.array([0., 1.5, 1.4, 1.4, 1.5])
        knees = np.array([0., 0., 2., 2., 0.])
        hands = np.array([1., 1., 0., 1., 1.])
        q = position_qualities(hips, knees, hands)
        self.assertEqual(q.shape, (5, 4))
        self.assertEqual(q[0, 0], 1.)          # straight
        self.assertEqual(q[1, 1], 1.)          # pike
        self.assertEqual(q[2, 2], 1.)          # tuck with hands on the shins
        self.assertLess(q[3, 2], q[2, 2])      # hands away from the shins is a worse tuck
        np.testing.assert_array_equal(q[:, 3], q[:, :3].max(axis=1))
        # A body bent 80 degrees at the hips still gets a measurable slope toward straight.
        far = position_qualities(np.array([1.4, 1.3]), np.zeros(2), np.ones(2))[:, 0]
        self.assertGreater(far[1], far[0])
        self.assertGreater(far[1] - far[0], 1e-3)


if __name__ == '__main__':
    unittest.main()
