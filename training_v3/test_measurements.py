"""Physical drops that probe judge measurements; nothing here trains or prescribes motion."""
import unittest

import mujoco
import numpy as np

from engine import ACTUATOR_MAP, STATE_SPEC
from probes import action_for, single_athlete


def drop(vy, vz=-9.2, height=3.):
    """Drop a straight head-first athlete into the water with the given velocity; returns the episode info."""
    p = single_athlete(seed=2)
    m, d = p.model, p.resetdata
    p.reset([0], [dict(skill=0, height=height, preload=0, disturbance=0)])
    p.platform[0] = True
    p.headfirst[0] = True
    target = np.array([0, 0, 1.35, np.pi, np.pi, -.3, .3, 0, .06])
    mujoco.mj_resetData(m, d)
    d.mocap_pos[0] = [0, 0, -height]
    d.qpos[p.qadr] = target[ACTUATOR_MAP]
    d.ctrl[:] = target[ACTUATOR_MAP]
    d.qpos[4:8] = [0, 0, 1, 0]
    d.qpos[1] = 2.
    d.qpos[3] = 0
    mujoco.mj_forward(m, d)
    low = min(d.geom_xpos[g, 2] - np.dot(np.abs(d.geom_xmat[g].reshape(3, 3)[2]), m.geom_size[g]) for g in range(1, 16))
    d.qpos[3] += -height + .03 - low
    d.qvel[:] = 0
    d.qvel[2] = vy
    d.qvel[3] = vz
    mujoco.mj_forward(m, d)
    mujoco.mj_getState(m, d, p.state[0], STATE_SPEC)
    p.sensors[0] = d.sensordata
    p.targets[0] = target
    p.released[0] = True
    info = None
    for _ in range(120):
        _, _, done, info = p.step(action_for(target), auto_reset=False)
        if done[0]:
            break
    p.close()
    return info[0]


def hop(vz, gap=.2):
    """Lift the standing athlete off a platform with the given vertical speed; returns counted bounces."""
    p = single_athlete(seed=2)
    m, d = p.model, p.resetdata
    p.reset([0], [dict(skill=0, height=10, preload=0, disturbance=0)])
    p.platform[0] = True
    mujoco.mj_setState(m, d, p.state[0], STATE_SPEC)
    d.qpos[3] += gap
    d.qvel[:] = 0
    d.qvel[3] = vz
    mujoco.mj_forward(m, d)
    mujoco.mj_getState(m, d, p.state[0], STATE_SPEC)
    p.sensors[0] = d.sensordata
    action = action_for(p.targets[0])
    for _ in range(60):
        _, _, done, _ = p.step(action, auto_reset=False)
        if done[0]:
            break
    bounces = int(p.preparation_bounces[0])
    p.close()
    return bounces


class MeasurementTests(unittest.TestCase):
    def test_surface_lateral_speed_is_sideways_linear_speed(self):
        still = drop(0.)
        moving = drop(2.)
        # Limbs deflect sideways under the impact, so a vertical drop is not exactly zero,
        # but it must be metres per second, not the tens of radians per second of the old reading.
        self.assertLess(still['surfaceLateralSpeed'], 1.5)
        self.assertGreater(moving['surfaceLateralSpeed'] - still['surfaceLateralSpeed'], 1.)
        self.assertLess(moving['surfaceLateralSpeed'], 3.6)
        self.assertTrue(still['fullEntryComplete'])
        self.assertLess(still['entryAngle'], 6)

    def test_instability_reset_ends_the_dive_instead_of_teleporting(self):
        p = single_athlete(seed=2)
        p.reset([0], [dict(skill=0, height=10, preload=0, disturbance=0, platform=True)])
        p.platform[0] = True
        p.state[0, 2:5] = [2., 0., -3.]
        p.state[0, 26] = 1e12  # state = [time, qpos(22), qvel(21)]: root z velocity beyond mjMAXVAL
        _, _, done, info = p.step(np.zeros((1, 9)), auto_reset=False)
        self.assertTrue(done[0])
        self.assertGreaterEqual(p.diverged_events, 1)
        self.assertFalse(info[0]['water'])
        p.close()

    def test_preparation_bounce_requires_an_upward_hop(self):
        self.assertEqual(hop(-.2), 0)                   # dropped onto the platform: contact regained without a hop
        self.assertEqual(hop(.15, gap=.01), 0)          # a wobble of the board or a toe lift is not a hop either
        self.assertGreaterEqual(hop(1.8, gap=.02), 1)   # launched upward, lands again: one hop


if __name__ == '__main__':
    unittest.main()
