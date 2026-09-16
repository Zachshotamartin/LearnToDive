"""The environment must make a real takeoff physically reachable; the controller still has to learn it."""
import unittest

import mujoco
import numpy as np

from engine import STATE_SPEC
from probes import action_for, single_athlete


def standing(height=10, **context):
    p = single_athlete(seed=3)
    p.reset([0], [dict(skill=0, height=height, preload=0, disturbance=0, **context)])
    p.platform[0] = True
    p.headfirst[0] = True
    p.goals[0] = [.5, 0, 0, 3]
    return p


class StanceTests(unittest.TestCase):
    def test_standing_start_is_flat_footed_balanced_and_not_over_the_edge(self):
        p = standing()
        m = p.model
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, p.state[0], STATE_SPEC)
        mujoco.mj_forward(m, d)
        feet = [int(m.body_geomadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)]) for n in ['foot_L', 'foot_R']]
        for g in feet:
            self.assertLess(abs(d.geom_xmat[g].reshape(3, 3)[2, 0]), 1e-3)
        front = max(d.geom_xpos[g, 0] + np.abs(d.geom_xmat[g].reshape(3, 3)[0]) @ m.geom_size[g] for g in feet)
        rear = min(d.geom_xpos[g, 0] - np.abs(d.geom_xmat[g].reshape(3, 3)[0]) @ m.geom_size[g] for g in feet)
        com = float(d.subtree_com[2][0])
        self.assertAlmostEqual(front, .03, places=3)
        self.assertGreater(com, rear)
        self.assertLess(com, front)
        action = action_for(p.targets[0])
        for _ in range(150):
            p.step(action, auto_reset=False)
            if p.released[0]:
                break
        self.assertGreater(float(p.state[0, 0]), 1.0, 'an unbalanced start topples before a controller could act')
        p.close()

    def test_forward_countermovement_takeoff_is_physically_reachable(self):
        # Open-loop feasibility probe (from a bounded random search); never a training target.
        p = standing(lean=0., hip=.12, knee=.2, toeOver=.03)
        crouch = np.array([.67, 1.73, -.55, .01, .01, 0, 0, 0, 0.])
        extend = np.array([.32, 0., -.13, 3.1, 3.1, -.3, .3, 0, .06])
        straight = np.array([0, 0, 1.2, 3.14, 3.14, -.3, .3, 0, .06])
        info = None
        extension_start = None
        for _ in range(220):
            t = p.state[0, 0]
            if t < .1:
                target = p.targets[0]
            elif t < .52:
                target = crouch
            elif not p.released[0]:
                extension_start = t if extension_start is None else extension_start
                target = extend.copy()
                if t - extension_start >= .25:
                    target[2] = 1.3
            else:
                target = straight
            _, _, done, rows = p.step(action_for(target), auto_reset=False)
            if rows:
                info = rows[0]
                break
        ascent = float(max(0, p.apex_com[0] - p.departure_com[0]))
        p.close()
        self.assertIsNotNone(info)
        self.assertTrue(info['water'])
        self.assertFalse(info['boardInvalid'])
        self.assertGreater(info['x'], .4)
        self.assertGreater(ascent, .2)
        self.assertGreater(info['takeoffVerticalSpeed'], 2.)


if __name__ == '__main__':
    unittest.main()
