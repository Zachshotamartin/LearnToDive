"""Declared body position and real feet-first/head-first arm regressions."""
import copy
import unittest
import numpy as np
import mujoco
from geometry import ACTUATOR_MAP, entry_arm_assessment
from engine import STATE_SPEC
from probes import single_athlete, action_for
from positions import position_qualities, recognized_positions
from judge import judge
from rules import IDS, difficulty
from test_judge import measurements
from motor_objective import live_errors, terminal_components


def arm_pose(pitch, headfirst=False, elbow=0., roll=0.):
    p = single_athlete(seed=2)
    p.reset([0], [dict(skill=0, height=3, preload=0, disturbance=0)])
    p.platform[0] = True
    p.headfirst[0] = headfirst
    d, m = p.resetdata, p.model
    mujoco.mj_setState(m, d, p.state[0], STATE_SPEC)
    target = np.array([0, 0, 1.35, pitch, pitch, -.3 if headfirst else roll, .3 if headfirst else -roll, elbow, .06])
    d.qpos[p.qadr] = target[ACTUATOR_MAP]
    d.ctrl[:] = target[ACTUATOR_MAP]
    d.qpos[4:8] = [0, 0, 1, 0] if headfirst else [1, 0, 0, 0]
    d.qpos[1] = 2
    mujoco.mj_forward(m, d)
    low = min(d.geom_xpos[g, 2] - np.dot(np.abs(d.geom_xmat[g].reshape(3, 3)[2]), m.geom_size[g]) for g in range(1, 16))
    d.qpos[3] += -3 + .03 - low
    d.qvel[:] = 0
    d.qvel[3] = -5
    mujoco.mj_forward(m, d)
    mujoco.mj_getState(m, d, p.state[0], STATE_SPEC)
    p.sensors[0] = d.sensordata
    p.targets[0] = target
    p.released[0] = True
    return p, target


class DeclaredFormTests(unittest.TestCase):
    def test_real_pose_arm_references_and_severity(self):
        for head, pitch, elbow, roll, valid, cap in [
            (False, 0, 0, 0, True, False),
            (False, 3.05, 0, 0, False, True),
            (False, 0, 1.2, 0, False, False),
            (False, 0, 0, .8, False, False),
            (True, 3.05, 0, 0, True, False),
            (True, 0, 0, 0, False, True),
        ]:
            with self.subTest(head=head, pitch=pitch, elbow=elbow, roll=roll):
                p, target = arm_pose(pitch, head, elbow, roll)
                try:
                    good, severe = entry_arm_assessment(p.state[:, 1+p.qadr], p.sensors, p.headfirst)
                    self.assertEqual(bool(good[0]), valid)
                    self.assertEqual(bool(severe[0]), cap)
                    if valid:
                        err = live_errors(p)
                        self.assertAlmostEqual(err['shoulderPitch'][0], 0, places=5)
                        self.assertAlmostEqual(err['shoulderRoll'][0], 0, places=5)
                finally: p.close()

    def test_real_feet_first_drops_charge_wrong_arms_separately(self):
        results = []
        for pitch, elbow in [(0, 0), (3.05, 0), (0, 1.2)]:
            p, target = arm_pose(pitch, elbow=elbow)
            try:
                for _ in range(200):
                    _, _, done, info = p.step(action_for(target), auto_reset=False)
                    if done[0]: break
                self.assertTrue(done[0])
                self.assertTrue(info[0]['water'])
                results.append(info[0])
            finally: p.close()
        self.assertGreater(results[1]['entryFaultLosses']['entryShoulderPitch'], results[0]['entryFaultLosses']['entryShoulderPitch']+1)
        self.assertGreater(results[2]['entryFaultLosses']['entryElbowBend'], results[0]['entryFaultLosses']['entryElbowBend']+.2)
        self.assertTrue(results[1]['entryArmPositionCap'])
        self.assertFalse(results[1]['entryArmPositionValid'])

    def test_pike_knee_fault_is_not_a_different_position_or_dd(self):
        hips=np.array([[1.5,1.5]]*3);knees=np.array([[0,0],[.5,.5],[2,2]])
        np.testing.assert_array_equal(recognized_positions(hips,knees), [1,1,2])
        q=position_qualities(hips[:,0],knees[:,0],np.zeros(3),hips[:,1],knees[:,1])
        self.assertGreater(q[0,1],q[1,1]);self.assertGreater(q[1,1],q[2,1])
        scores=[]
        for i, recognized in enumerate([1,1,2]):
            m=measurements();m.update(rotation=1.5,positionQuality=float(q[i,1]),positionRecognition=dict(samples=30,fractions=np.eye(3)[recognized].tolist()))
            scores.append(judge(IDS['103B'],'platform',10,m))
        self.assertFalse(scores[1]['wrongBodyPosition'])
        self.assertEqual(scores[1]['scoreAdjustments']['positionCap'],0)
        self.assertGreater(scores[0]['execution'],scores[1]['execution'])
        self.assertTrue(scores[2]['wrongBodyPosition']);self.assertLessEqual(scores[2]['execution'],2)
        self.assertTrue(scores[1]['completedDeclaration'])
        self.assertFalse(scores[2]['completedDeclaration'])
        self.assertTrue(all(s['difficulty']==difficulty(IDS['103B'],'platform',10) for s in scores))
        self.assertNotEqual(difficulty(IDS['103B'],'platform',10),difficulty(IDS['103C'],'platform',10))

    def test_wrong_body_position_keeps_progress_but_costs_execution(self):
        m=measurements()
        m.update(rotation=1.5,positionQuality=.2,positionRecognition=dict(samples=30,fractions=[0,0,1]))
        score=judge(IDS['103B'],'platform',10,m)
        self.assertTrue(score['valid'])  # Rules cap it; they do not fail the dive outright.
        reward=terminal_components(score,m,completion_first=True)
        self.assertFalse(score['completedDeclaration'])
        self.assertEqual(reward['credits']['rotationProgress'],16.)
        self.assertNotIn('difficulty',reward['credits'])
        m.update(positionQuality=.4,positionRecognition=dict(samples=30,fractions=[0,1,0]))
        improved=terminal_components(judge(IDS['103B'],'platform',10,m),m,completion_first=True)
        self.assertEqual(improved['credits'],reward['credits'])
        self.assertGreater(improved['total'],reward['total'])

    def test_one_bent_knee_cannot_hide_behind_the_straight_other_leg(self):
        q=position_qualities(np.array([1.5]),np.array([0.]),np.zeros(1),np.array([1.5]),np.array([1.]))
        self.assertLess(q[0,1],.5)
        self.assertEqual(recognized_positions(np.array([[1.5,1.5]]),np.array([[0.,1.]]))[0],-1)

    def test_mild_arm_fault_does_not_inherit_the_overhead_cap(self):
        m=measurements();m.update(entryArmPositionValid=False,entryArmPositionCap=False)
        m['entryFaultLosses']['entryElbowBend']=1.
        s=judge(IDS['101C'],'platform',10,m)
        self.assertEqual(s['scoreAdjustments']['armPositionCap'],0)
        self.assertLess(s['execution'],10);self.assertGreater(s['execution'],4.5)
        self.assertFalse(s['clean'])

if __name__=='__main__': unittest.main()
