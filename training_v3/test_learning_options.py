"""Credit assignment, exploration density, and resumable recovery sampling."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from policy import Policy
from recovery_curriculum import RecoveryArchive
from rules import DIVES
from train import estimate_advantages, parser, train


class LearningOptions(unittest.TestCase):
    def test_credit_trace_and_terminal_boundary(self):
        batch = dict(rewards=torch.tensor([[0.], [0.], [1.]]), values=torch.zeros(3, 1),
                     dones=torch.tensor([[0.], [0.], [1.]]), choosing=torch.zeros(3, 1, dtype=torch.bool),
                     bootstrap=torch.tensor([99.]))
        for lam in [.95, .98, .99]:
            advantage, returns = estimate_advantages(batch, lam)
            torch.testing.assert_close(advantage[:, 0], torch.tensor([(.995 * lam) ** 2, .995 * lam, 1.]))
            torch.testing.assert_close(returns, advantage)

    def test_covariance_density_gradients_and_deterministic_mean(self):
        torch.manual_seed(43)
        policy = Policy(20, (32, 32), exploration='state-covariance')
        obs = torch.randn(6, 20).clamp(-.8, .8)
        mask = torch.ones(6, len(DIVES), dtype=torch.bool)
        choosing = torch.zeros(6, dtype=torch.bool)
        sample = policy(obs, mask, choosing)
        replay = policy(obs, mask, choosing, raw=sample['raw'], choice=sample['choice'])
        torch.testing.assert_close(sample['logp'], replay['logp'], atol=0, rtol=0)
        h = policy.trunk(obs)
        mean = (1 - policy.rho) * 2 * torch.tanh(policy.motor(h) / 2) + policy.rho * obs[:, -9:].atanh()
        distribution = policy.motor_distribution(h, mean)
        expected = distribution.log_prob(sample['raw']) - policy.log_jacobian(sample['raw']).sum(-1)
        torch.testing.assert_close(expected, sample['logp'])
        covariance = distribution.covariance_matrix
        off_diagonal = covariance - torch.diag_embed(covariance.diagonal(dim1=-2, dim2=-1))
        self.assertGreater(float(off_diagonal.detach().abs().sum()), 0)
        self.assertFalse(torch.equal(covariance[0], covariance[1]))
        (-sample['logp'].mean() - .01 * sample['entropy'].mean()).backward()
        self.assertTrue(torch.isfinite(policy.noise_factor.weight.grad).all())
        self.assertGreater(float(policy.noise_factor.weight.grad.abs().sum()), 0)
        torch.testing.assert_close(policy(obs, mask, choosing, deterministic=True)['action'], mean.tanh())

    def test_recovery_archive_diversity_and_restoration(self):
        archive = RecoveryArchive()
        for stage in range(3):
            for k in range(8):
                snapshot = dict(declaration=1, height=10., apparatus=1, physics=dict(x=np.array([k])))
                archive.add(0, snapshot, [stage, k / 4, 0, 0], stage)
        self.assertEqual(len(archive.groups[0]), 24)
        first = archive.groups[0][0]
        archive.observe(0, first['id'], .8)
        self.assertGreater(first['fast'], first['slow'])
        restored = RecoveryArchive()
        restored.load_state_dict(archive.state_dict())
        rng1, rng2 = np.random.default_rng(8), np.random.default_rng(8)
        selected = [archive.choose(0, 1, 10, 1, rng1)['id'] for _ in range(100)]
        self.assertEqual(selected, [restored.choose(0, 1, 10, 1, rng2)['id'] for _ in range(100)])
        self.assertEqual({r['stage'] for r in archive.groups[0] if r['id'] in selected}, {0, 1, 2})
        self.assertIsNone(archive.choose(0, 2, 10, 1, rng1))

    def test_all_options_exact_training_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            def options(name, steps, resume=None):
                a = parser().parse_args(['--output', str(Path(folder) / name), '--envs', '4', '--threads', '1',
                    '--horizon', '32', '--widths', '32', '32', '--batch', '64', '--epochs', '1',
                    '--steps', str(steps), '--evaluate-every', '0', '--archive-every', '1',
                    '--gae-lambda', '.98', '--recovery-mode', 'progress', '--exploration', 'state-covariance'])
                a.resume = resume
                return a
            with contextlib.redirect_stdout(io.StringIO()):
                train(options('whole', 512))
                train(options('split', 256))
                train(options('split', 256, str(Path(folder) / 'split/latest.pt')))
            whole = torch.load(Path(folder) / 'whole/latest.pt', weights_only=False)
            split = torch.load(Path(folder) / 'split/latest.pt', weights_only=False)
            for key in whole['model']:
                torch.testing.assert_close(whole['model'][key], split['model'][key], atol=0, rtol=0)
            torch.testing.assert_close(whole['torchRNG'], split['torchRNG'], atol=0, rtol=0)
            np.testing.assert_array_equal(whole['environment']['recoveryArchive']['readiness'],
                                          split['environment']['recoveryArchive']['readiness'])
