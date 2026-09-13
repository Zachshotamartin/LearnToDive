"""Direction conventions, task feedback, and checkpoint-preserving phase migration."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from engine import STATE_SPEC
from environment import Arena
from geometry import framed_angles
from motor_curriculum import MotorCurriculum
from motor_objective import rotation_direction_cost, potential_components
from rules import IDS, DIVES
from stance import orientation
from train import Trainer, parser, train
from evaluation import evaluate_targets
from policy import Policy


class DirectionTests(unittest.TestCase):
    def test_world_axis_agrees_with_judge_for_both_facings(self):
        e = Arena(1, 38, 1, training=False)
        try:
            model = e.physics.model
            for code in ('101C', '201C', '301C', '401C'):
                declaration = DIVES[IDS[code]]
                sign = declaration['sign']
                d = mujoco.MjData(model)
                # Measurement unit test only: a freely rotating straight body.
                # Feasibility is tested separately from actual board controls.
                d.qpos[1:4] = [3, 0, 4]
                d.qpos[4:8] = orientation(0, np.pi * declaration['back'])
                # MuJoCo free-joint angular velocity uses the body-local frame.
                d.qvel[5] = sign * (-1 if declaration['back'] else 1)
                mujoco.mj_forward(model, d)
                self.assertGreater(sign * d.sensordata[7], 0)
                before = framed_angles(d.qpos[4:8][None])[0][0]
                mujoco.mj_step(model, d)
                after = framed_angles(d.qpos[4:8][None])[0][0]
                self.assertGreater(sign * (after-before), 0)
        finally:
            e.close()

    def test_wrong_momentum_is_costly_and_extra_spin_is_not_a_bonus(self):
        for turns in (-1.5, -.5, .5, 1.5):
            sign = np.sign(turns)
            costs = [rotation_direction_cost(sign*x, turns) for x in (-100, -12, 0, 6, 12, 120)]
            self.assertTrue(all(a >= b for a, b in zip(costs, costs[1:])))
            self.assertEqual(costs[-1], costs[-2])
            self.assertGreaterEqual(rotation_direction_cost(sign*120, turns, False), 1)

    def test_takeoff_balances_directions_and_keeps_policy_choosing(self):
        e = MotorCurriculum(Arena(16, 29, 1, practice=0), direction_practice=True)
        try:
            for _ in range(30):
                e.base.reset(np.arange(e.n), new_routine=True)
                e.assign(np.arange(e.n))
                self.assertTrue(e.choosing.all())
                self.assertTrue(e.mask().any(1).all())
            self.assertLessEqual(np.ptp(e.direction_assignments), 1)
            self.assertGreater(e.direction_assignments.min(), 0)
            e.task[:] = 1
            short = e.horizons().copy()
            e.level[1] = 1
            self.assertTrue((e.horizons() > short).all())
            self.assertEqual(e.observe().shape[1], 233)
        finally:
            e.close()

    def test_wrong_direction_cannot_pass_jump_only_success(self):
        e = MotorCurriculum(Arena(1, 25, 1, practice=0), direction_practice=True)
        try:
            e.group[:], e.height[:], e.apparatus[:] = 3, 10, 1
            e.base.declare(0, IDS['301C'])
            e.task[:] = 1
            p = e.physics
            p.released[:] = True
            p.apex_com[:] = p.departure_com + .5
            p.takeoff_vertical_speed[:] = 3
            p.takeoff_angular_momentum[:, 1] = 25
            p.phase_theta[:] = np.pi
            wrong = e.costs()[0]
            p.takeoff_angular_momentum[:, 1] = -25
            p.phase_theta[:] = -.2
            correct = e.costs()[0]
            self.assertGreater(wrong, .7)
            self.assertLess(correct, .7)
            self.assertEqual(potential_components(e.base)['takeoffDirection'][0], 0)
        finally:
            e.close()

    def test_assigned_targets_get_motor_experience_without_selector_gradients(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = ContinuationTests().args(Path(tmp)/'goal', direction=True)
            args.horizon = 250
            t = Trainer(args)
            try:
                t.env.task[:] = 0
                t.env.assigned_goal[:] = True
                t.env.enabled = False
                batch = t.collect()
                self.assertFalse(batch['learn'][0][batch['choosing'][0]].any())
                assigned = [r for r in batch['episodes'] if r.get('assignedPracticeGoal')]
                self.assertTrue(assigned)
                self.assertTrue(all(r['practice'] for r in assigned))
                self.assertGreater(t.env.goal_visits.sum(), 0)
                self.assertTrue((batch['learn'] & ~batch['choosing']).any())
            finally:
                t.env.close()


    def test_target_benchmark_uses_requested_dives_and_real_board_starts(self):
        policy = Policy(233, (16, 16), architecture='split')
        before = torch.get_rng_state().clone()
        report = evaluate_targets(policy, cases_per_target=1)
        self.assertEqual(len(report['episodes']), 10)
        self.assertFalse(report['autonomousSelection'])
        self.assertTrue(report['boardStartsOnly'])
        self.assertEqual(set(report['targets']), {r['declaration'] for r in report['episodes']})
        self.assertTrue(all(r['assignedEvaluationGoal'] and not r['practice'] for r in report['episodes']))
        torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)


class ContinuationTests(unittest.TestCase):
    def args(self, output, steps=160, direction=False):
        options = ['--output', str(output), '--envs', '4', '--threads', '1', '--widths', '16', '16',
                   '--horizon', '40', '--steps', str(steps), '--epochs', '1', '--batch', '80', '--evaluate-every', '0',
                   '--motor-curriculum', 'adaptive', '--architecture', 'split', '--reward-mode', 'phase-dense',
                   '--recovery-mode', 'progress', '--gae-lambda', '.99']
        if direction:
            options.extend(['--direction-practice', '--goal-practice'])
        return parser().parse_args(options)

    def test_continuation_preserves_network_optimizer_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            train(self.args(root/'old', 480))
            old = torch.load(root/'old/latest.pt', weights_only=False)
            a = self.args(root/'new', direction=True)
            a.continue_from = str(root/'old/latest.pt')
            t = Trainer(a)
            try:
                self.assertEqual(t.state['steps'], 480)
                self.assertEqual(t.state['updates'], old['training']['updates'])
                for key, value in old['model'].items():
                    torch.testing.assert_close(t.policy.state_dict()[key], value, rtol=0, atol=0)
                for key, values in old['optimizer']['state'].items():
                    for name, value in values.items():
                        torch.testing.assert_close(t.optimizer.state_dict()['state'][key][name], value, rtol=0, atol=0)
                self.assertTrue(t.env.choosing.all())
                before = copy.deepcopy(t.policy.state_dict())
                t.update()
                self.assertEqual(t.state['steps'], 640)
                for key, value in before.items():
                    if not key.startswith(('value', 'critic_trunk.')):
                        torch.testing.assert_close(t.policy.state_dict()[key], value, rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(t.policy.state_dict()[k], v) for k,v in before.items()
                                    if k.startswith('critic_trunk.')))
            finally:
                t.env.close()

    def test_new_phase_exact_resume_preserves_directional_curriculum(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            train(self.args(root/'whole', 640, direction=True))
            train(self.args(root/'split', 320, direction=True))
            a = self.args(root/'split', 320, direction=True)
            a.resume = str(root/'split/latest.pt')
            train(a)
            x,y = [torch.load(root/name/'latest.pt', weights_only=False) for name in ('whole','split')]
            for key in x['model']:
                torch.testing.assert_close(x['model'][key], y['model'][key], rtol=0, atol=0)
            for key in x['environment']['curriculum']:
                np.testing.assert_array_equal(x['environment']['curriculum'][key], y['environment']['curriculum'][key])


if __name__ == '__main__':
    unittest.main()
