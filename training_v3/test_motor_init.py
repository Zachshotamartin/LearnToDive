"""The motor head's posture prior: the untrained entry-pose policy already passes the entry sub-task."""
import unittest

import numpy as np
import torch

from evaluation import evaluate_motor_skills
from geometry import decoded_action, encoded_action
from motor_curriculum import ENTRY_POSE, EXTRA_OBSERVATIONS
from policy import Policy
from rotation_progress import NEW_SIZE
from train import motor_init_mode, motor_initial_action, parser


class MotorInitTests(unittest.TestCase):
    def test_modes_and_aliases(self):
        args = parser().parse_args(['--output', 'x'])
        self.assertEqual(motor_init_mode(args), 'stance')
        self.assertEqual(motor_init_mode(parser().parse_args(['--output', 'x', '--random-motor-init'])), 'random')
        self.assertIsNone(motor_initial_action(parser().parse_args(['--output', 'x', '--random-motor-init']), None))
        entry = parser().parse_args(['--output', 'x', '--motor-init', 'entry-pose'])
        np.testing.assert_allclose(motor_initial_action(entry, None), encoded_action(ENTRY_POSE), atol=1e-6)
        with self.assertRaises(SystemExit):
            parser().parse_args(['--output', 'x', '--motor-init', 'somersault'])

    def test_untrained_entry_pose_policy_holds_the_pose_and_passes_the_entry_task(self):
        torch.manual_seed(3)
        policy = Policy(NEW_SIZE, (32, 32), architecture='split', initial_action=encoded_action(ENTRY_POSE),
                        noise_rho=.9, normalize_inputs=True).eval()
        # The autoregressive steady state of the motor head is the pose itself.
        steady = torch.tanh(2 * torch.tanh(policy.motor.bias.detach() / 2)).numpy()
        pose = decoded_action(steady)
        np.testing.assert_allclose(pose[[0, 3, 4, 5, 6, 8]], ENTRY_POSE[[0, 3, 4, 5, 6, 8]], atol=.12)
        self.assertLess(pose[1], .1, 'knees straight')
        self.assertLess(pose[7], .1, 'elbows straight')
        report = evaluate_motor_skills(policy, cases=12, direction_practice=True)
        self.assertGreaterEqual(report['tasks']['entry']['success'], .5)
        self.assertLess(report['tasks']['entry']['meanError'], 1.)
        centred = Policy(NEW_SIZE, (32, 32), architecture='split', noise_rho=.9, normalize_inputs=True).eval()
        self.assertEqual(evaluate_motor_skills(centred, cases=12, direction_practice=True)['tasks']['entry']['success'], 0.)


if __name__ == '__main__':
    unittest.main()


class ExplorationScaleTests(unittest.TestCase):
    def test_small_coherent_noise_keeps_the_entry_task_winnable_during_training(self):
        """The suite's exploration scale must leave the held pose enough precision to enter cleanly."""
        from evaluation import matching_arena, DT
        args = parser().parse_args(['--output', 'x', '--motor-init', 'entry-pose', '--motor-logstd', '-2'])
        self.assertEqual(args.motor_logstd, -2.)
        torch.manual_seed(5)
        results = {}
        for logstd in (-.5, -2.):
            policy = Policy(NEW_SIZE, (32, 32), architecture='split', initial_action=encoded_action(ENTRY_POSE),
                            noise_rho=.9, normalize_inputs=True, motor_logstd=logstd).eval()
            e = matching_arena(policy, 24, 782100, practice=0)
            e.task[:] = 4
            e.level[:] = 0
            e.up_goal[:] = [0, 0, -1]
            records = []
            with torch.no_grad():
                for _ in range(int(3 / DT) + 3):
                    obs = e.observe()
                    out = policy(torch.tensor(obs), torch.tensor(e.mask()), torch.tensor(e.choosing), deterministic=False)
                    e.step(out['choice'].numpy(), out['action'].numpy(), out['noise'].numpy())
                    records.extend(e.last_completed)
                    if len(records) >= 24:
                        break
            e.close()
            results[logstd] = float(np.mean([r['success'] for r in records]))
        self.assertEqual(results[-.5], 0., 'the historical scale never enters cleanly while exploring')
        self.assertGreaterEqual(results[-2.], .15)
        with self.assertRaises(ValueError):
            Policy(NEW_SIZE, (32, 32), motor_logstd=-3.5)


class ExplorationCeilingTests(unittest.TestCase):
    """A run may explore less than it started with, never more."""

    def policy(self, logstd, ceiling=-1.9):
        model = Policy(NEW_SIZE, (32, 32), architecture='split', initial_action=encoded_action(ENTRY_POSE),
                       noise_rho=.9, normalize_inputs=True, motor_logstd=-2., motor_logstd_max=ceiling).eval()
        with torch.no_grad():
            model.logstd.fill_(logstd)
        return model

    def scale(self, model):
        features = torch.zeros(1, sum(model.widths[:1]))
        return model.motor_distribution(model.trunk(torch.zeros(1, NEW_SIZE)), torch.zeros(1, 9)).scale

    def test_the_ceiling_bounds_the_sampled_scale(self):
        correction = np.sqrt(1 - .9 ** 2)
        below = self.scale(self.policy(-2.5))
        np.testing.assert_allclose(below.detach().numpy(), np.exp(-2.5) * correction, rtol=1e-6)
        # Anything above the ceiling is clamped to it, however far the parameter drifts.
        for logstd in (-1.9, -1.5, 0., 5.):
            capped = self.scale(self.policy(logstd))
            np.testing.assert_allclose(capped.detach().numpy(), np.exp(-1.9) * correction, rtol=1e-6)

    def test_the_exported_scale_is_the_one_training_used(self):
        exported = self.policy(-1.2).export()['state']['logstd']
        np.testing.assert_allclose(exported, [-1.9] * 9, rtol=1e-6)

    def test_the_default_ceiling_keeps_the_historical_behaviour(self):
        model = Policy(NEW_SIZE, (32, 32), architecture='split')
        self.assertEqual(model.logstd_max, 0.)
        with self.assertRaises(ValueError):
            Policy(NEW_SIZE, (32, 32), motor_logstd=-1., motor_logstd_max=-1.5)
        with self.assertRaises(ValueError):
            Policy(NEW_SIZE, (32, 32), motor_logstd_max=.5)
