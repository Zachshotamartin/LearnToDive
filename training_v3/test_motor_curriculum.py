"""Real-physics regressions for phase feedback, skill isolation and exact resume."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from checkpointing import hashes
from environment import Arena
from evaluation import evaluate, evaluate_motor_skills
from judge import judge
from motor_curriculum import MotorCurriculum
from motor_objective import GAMMA, ENTRY_WEIGHTS, terminal_components, shaped_reward, potential_components
from policy import Policy, FORMAT
from rules import IDS
from test_judge import measurements
from train import Trainer, parser, train


class MotorRewardTests(unittest.TestCase):
    def test_each_fault_has_continuous_feedback_even_for_failed_rotation(self):
        good = measurements()
        good['rotation'] = 0
        for key in ENTRY_WEIGHTS:
            previous = None
            for severity in (0., .1, 1., 10., 100.):
                m = copy.deepcopy(good)
                m['entryFaultLosses'][key] = severity
                result = terminal_components(judge(IDS['101C'], 'platform', 10, m), m)
                if previous is not None:
                    self.assertLess(result['total'], previous)
                previous = result['total']
        score = judge(IDS['101C'], 'platform', 10, good)
        baseline = terminal_components(score, good)
        # Reported judge caps and clipped points do not enter the motor objective.
        score.update(execution=0, points=100, scoreAdjustments=dict(positionCap=999))
        self.assertEqual(terminal_components(score, good), baseline)

    def test_potential_telescopes_through_pose_loops_and_terminal_failure(self):
        for potentials in ([0, -7, -2, -7, -2, 0], [0, 1, 1, 1, 1, 0], [-3, 1, -2, 7, 0]):
            total = 0.
            for i in range(len(potentials) - 1):
                total += GAMMA ** i * shaped_reward(potentials[i], potentials[i+1], i == len(potentials)-2)
            self.assertAlmostEqual(total, -potentials[0], places=12)

    def test_wrong_rotation_still_receives_flight_shape_feedback(self):
        e = Arena(1, 21, 1, training=False, reward_mode='phase-dense')
        try:
            e.group[0], e.height[0], e.apparatus[0] = 1, 10, 1
            e.declare(0, IDS['103C'])
            e.physics.released[0] = True
            e.physics.above_water[0] = 10
            e.physics.phase_theta[0] = -3  # Wrong direction, outside old measurement window.
            parts = potential_components(e)
            self.assertLess(parts['flightPosition'][0], 0.)
            e.physics._assess_flight(np.zeros(1, bool))
            self.assertEqual(e.physics.position_ticks[0], 0)
            self.assertGreater(e.physics.motor_position_weight[0], 0)
        finally:
            e.close()


class CurriculumTests(unittest.TestCase):
    def test_practice_is_finite_not_counted_and_resets_routine_history(self):
        e = MotorCurriculum(Arena(8, 15, 1, practice=1, reward_mode='phase-dense', recovery_mode='progress'))
        try:
            e.task[:] = [1, 2, 3, 4] * 2
            e.up_goal[e.task == 4] = [0, 0, -1]
            completed = set()
            # Disable assignments after this batch to isolate eight known skills.
            e.enabled = False
            for _ in range(155):
                choice = e.mask().argmax(1)
                _, reward, done, rows = e.step(choice, np.tile(e.initial_action, (8, 1)))
                self.assertTrue(np.isfinite(reward).all())
                self.assertFalse(any(r['practice'] for r in rows))
                for row in e.last_completed:
                    i = row['index']
                    completed.add(i)
                    self.assertFalse(e.base.used[i].any())
                    self.assertEqual(e.base.round[i], 0)
                    self.assertEqual(e.base.routine_points[i], 0)
            self.assertEqual(len(completed), 8)
            self.assertEqual(int(e.visits.sum()), 8)
        finally:
            e.close()

    def test_full_evaluation_never_uses_random_airborne_starts(self):
        e = MotorCurriculum(Arena(2, 1, 1, training=False), enabled=False)
        try:
            p = Policy(e.observation_size, (16, 16), initial_action=e.initial_action, architecture='split')
            before = torch.get_rng_state().clone()
            report = evaluate(p, cases=2, reward_mode='phase-dense')
            self.assertEqual(len(report['episodes']), 12)
            self.assertEqual(report['summary']['practiceEpisodes'], 0)
            skills = evaluate_motor_skills(p, cases=2)
            self.assertTrue(skills['practiceOnly'])
            self.assertEqual(len(skills['episodes']), 8)
            self.assertEqual(skills, evaluate_motor_skills(p, cases=2))
            torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)
        finally:
            e.close()

    def test_transfer_preserves_old_actor_and_selector_cannot_change_motor_trunk(self):
        with tempfile.TemporaryDirectory() as folder:
            old = Policy(234, (32, 32), .6)
            path = Path(folder) / 'parent.pt'
            torch.save(dict(model=old.state_dict(), training=dict(steps=123),
                            contract=dict(format=FORMAT, observationSize=234, sourceHashes=hashes())), path)
            args = parser().parse_args(['--output', str(Path(folder) / 'new'), '--envs', '2', '--threads', '1',
                                        '--widths', '32', '32', '--initialize-from', str(path),
                                        '--motor-curriculum', 'adaptive', '--architecture', 'split'])
            t = Trainer(args)
            try:
                x = torch.randn(6, 234)
                expanded = torch.cat([x[:, :-18], torch.zeros(6, 10), x[:, -18:]], 1)
                mask = torch.ones(6, old.choice.out_features, dtype=torch.bool)
                choose = torch.tensor([True, False] * 3)
                a, b = old(x, mask, choose, deterministic=True), t.policy(expanded, mask, choose, deterministic=True)
                torch.testing.assert_close(a['action'], b['action'])
                torch.testing.assert_close(a['choice'], b['choice'])
                self.assertEqual(len(t.optimizer.state), 0)
                t.policy.zero_grad()
                out = t.policy(expanded, mask, torch.ones(6, dtype=torch.bool))
                out['logp'].sum().backward()
                for parameter in t.policy.trunk.parameters():
                    self.assertTrue(parameter.grad is None or not parameter.grad.any())
                self.assertTrue(any(p.grad is not None and p.grad.any() for p in t.policy.selector_trunk.parameters()))
                t.policy.zero_grad()
                critic = t.policy(expanded, mask, choose)['normalizedValue']
                critic.square().mean().backward()
                for block in (t.policy.trunk, t.policy.selector_trunk, t.policy.motor, t.policy.choice):
                    for parameter in block.parameters():
                        self.assertTrue(parameter.grad is None or not parameter.grad.any())
                self.assertTrue(any(p.grad is not None and p.grad.any() for p in t.policy.critic_trunk.parameters()))
            finally:
                t.env.close()

    def test_adaptive_split_exact_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            def args(name, steps, resume=None):
                a = parser().parse_args(['--output', str(Path(folder) / name), '--envs', '4', '--threads', '1',
                    '--widths', '16', '16', '--horizon', '40', '--steps', str(steps), '--epochs', '1', '--batch', '80',
                    '--evaluate-every', '0', '--motor-curriculum', 'adaptive', '--architecture', 'split',
                    '--reward-mode', 'phase-dense', '--recovery-mode', 'progress', '--gae-lambda', '.99'])
                a.resume = resume
                return a
            with contextlib.redirect_stdout(io.StringIO()):
                train(args('whole', 960))
                train(args('resumed', 480))
                train(args('resumed', 480, str(Path(folder) / 'resumed/latest.pt')))
            a, b = [torch.load(Path(folder) / name / 'latest.pt', weights_only=False) for name in ('whole', 'resumed')]
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
            for key in a['environment']['curriculum']:
                np.testing.assert_array_equal(a['environment']['curriculum'][key], b['environment']['curriculum'][key])
            self.assertEqual(a['training']['steps'], 960)
            self.assertEqual(a['contract']['observationSize'], 244)

class QualificationTests(unittest.TestCase):
    def test_good_pooled_score_cannot_hide_bad_arms_or_missing_jumps(self):
        from model_selection import competence, comparison
        from test_assessment import report
        base = report()
        good = copy.deepcopy(base)
        good['summary']['categories'] = {str(i): dict(n=8, clean=.9, valid=1., entryArms=1.,
            motorPositionQuality=.8, completedRotation=1., jumped=1., rise=.3) for i in range(1, 7)}
        self.assertTrue(competence(good)['qualified'])
        bad = copy.deepcopy(good)
        bad['summary']['categories']['3']['entryArms'] = .2
        self.assertFalse(competence(bad)['qualified'])
        bad = copy.deepcopy(good)
        bad['summary']['categories']['1']['jumped'] = 0.
        self.assertFalse(competence(bad)['qualified'])
        candidate = report(points=50)
        for row in candidate['episodes']:
            if row['category'] == 3:
                row['measurements']['entryArmPositionValid'] = False
        self.assertFalse(comparison(candidate, base)['eligibleForReview'])

    def test_reference_digest_and_physics_are_enforced(self):
        from reference_policies import load_reference
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            p = Policy(223, (16, 16))
            path = Path(folder) / 'reference.pt'
            saved = dict(model=p.state_dict(), config=dict(widths=[16,16], rho=.6),
                contract=dict(format=FORMAT, observationSize=223, sourceHashes=hashes()), training=dict(steps=0))
            torch.save(saved, path)
            rng = torch.get_rng_state().clone()
            load_reference(path, hashlib.sha256(path.read_bytes()).hexdigest())
            torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_reference(path, 'incorrect')
            saved['contract']['sourceHashes']['diver.xml'] = 'different'
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, 'physical'):
                load_reference(path)

    def test_declaration_critic_uses_the_same_complete_return_as_selector(self):
        from train import estimate_advantages
        batch = dict(rewards=torch.tensor([[0.], [0.], [1.]]), values=torch.zeros(3, 1),
            dones=torch.tensor([[0.], [0.], [1.]]), choosing=torch.tensor([[True], [False], [False]]),
            bootstrap=torch.tensor([99.]))
        advantage, returns = estimate_advantages(batch, .95)
        self.assertAlmostEqual(float(advantage[0]), GAMMA ** 2, places=6)
        self.assertAlmostEqual(float(returns[0]), GAMMA ** 2, places=6)


if __name__ == '__main__':
    unittest.main()


class EntryTaskStartTests(unittest.TestCase):
    def test_entry_task_starts_near_the_pose_an_entry_needs(self):
        from motor_curriculum import MotorCurriculum, ENTRY_POSE
        from geometry import quat_up
        base = Arena(6, seed=12, threads=1, stage=0)
        try:
            e = MotorCurriculum(base, enabled=True)
            e.task[:] = 4
            e.up_goal[:] = [0, 0, -1]
            e.level[:] = 0.
            for i in range(6):
                e.initialize_task(i)
            physics = base.physics
            up = quat_up(physics.state[:, 5:9])
            self.assertTrue((up[:, 2] < -np.cos(np.radians(12))).all(), 'head first within twelve degrees of vertical')
            self.assertTrue((np.abs(physics.targets - ENTRY_POSE) <= .16).all())
            height = physics.sensors[:, 2] + physics.height
            self.assertTrue((height > 2.) .all() and (height < 4.).all())
        finally:
            base.close()
