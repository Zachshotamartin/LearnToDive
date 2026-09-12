"""Exact resume, the PPO surrogate and the source contract."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from checkpointing import hashes
from engine import TRAINING_SOURCES
from losses import ppo_terms
from train import Trainer, parser, train

TOOLING = ['run_suite.py', 'audit_physics.py', 'check_browser_parity.py', 'extract_difficulty.py', 'checkpointing.py', 'probes.py']
ESSENTIAL = ['engine.py', 'geometry.py', 'stance.py', 'positions.py', 'water.py', 'judge.py', 'rules.py', 'policy.py', 'losses.py',
             'train.py', 'evaluation.py', 'environment.py', 'difficulty.json', 'diver.xml']


class TrainingTests(unittest.TestCase):
    def test_exact_resume_and_finite_gradients(self):
        with tempfile.TemporaryDirectory() as folder:
            def args(name, steps, resume=None):
                a = parser().parse_args(['--output', str(Path(folder) / name), '--envs', '4', '--threads', '1', '--horizon', '32',
                                         '--widths', '32', '32', '--batch', '64', '--epochs', '1', '--steps', str(steps),
                                         '--evaluate-every', '0', '--archive-every', '1'])
                a.resume = resume
                return a
            with contextlib.redirect_stdout(io.StringIO()):
                train(args('whole', 512))
                train(args('split', 256))
                train(args('split', 256, str(Path(folder) / 'split/latest.pt')))
            a = torch.load(Path(folder) / 'whole/latest.pt', weights_only=False)
            b = torch.load(Path(folder) / 'split/latest.pt', weights_only=False)
            for k in a['model']:
                torch.testing.assert_close(a['model'][k], b['model'][k], atol=0, rtol=0)
            self.assertEqual(a['training']['steps'], b['training']['steps'])

    def test_resume_refuses_a_changed_source_unless_explicitly_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            a = parser().parse_args(['--output', str(Path(folder) / 'run'), '--envs', '4', '--threads', '1', '--horizon', '32',
                                     '--widths', '32', '32', '--batch', '64', '--epochs', '1', '--steps', '128',
                                     '--evaluate-every', '0', '--archive-every', '1'])
            with contextlib.redirect_stdout(io.StringIO()):
                train(a)
            saved = torch.load(Path(folder) / 'run/latest.pt', weights_only=False)
            saved['contract']['sourceHashes']['evaluation.py'] = 'changed'
            torch.save(saved, Path(folder) / 'run/latest.pt')
            a.resume = str(Path(folder) / 'run/latest.pt')
            with self.assertRaises(ValueError):
                Trainer(a)
            a.accept_source_change = ['evaluation.py']
            trainer = Trainer(a)
            self.assertEqual([x['file'] for x in trainer.state['sourceAmendments']], ['evaluation.py'])
            trainer.env.close()
            a.accept_source_change = ['train.py']
            with self.assertRaises(ValueError):
                Trainer(a)

    def test_evaluation_budget_fits_six_maximum_length_dives(self):
        from engine import DT, ENTRY_WINDOW, TIMEOUT
        from evaluation import MAX_TICKS, ROUTINE_LENGTH
        self.assertGreaterEqual(MAX_TICKS, ROUTINE_LENGTH * ((TIMEOUT + ENTRY_WINDOW) / DT + 2))

    def test_training_sources_exclude_tooling_and_cover_the_learning_problem(self):
        names = set(hashes())
        self.assertEqual(names, set(TRAINING_SOURCES))
        for tooling in TOOLING:
            self.assertNotIn(tooling, names)
        for essential in ESSENTIAL:
            self.assertIn(essential, names)


class LossTests(unittest.TestCase):
    def test_ppo_terms_weight_every_decision_once(self):
        n = 100
        logp = torch.zeros(n)
        old = torch.zeros(n)
        adv = torch.ones(n)
        choosing = torch.zeros(n, dtype=torch.bool)
        choosing[:2] = True
        selected = torch.ones(n, dtype=torch.bool)
        entropy = torch.cat([torch.full((2,), 3.), torch.full((98,), -4.)])
        loss, kl = ppo_terms(logp, old, adv, choosing, selected, entropy, motor_entropy=.006, declaration_entropy=.01)
        # The surrogate mean of -1 minus the two entropy bonuses.
        self.assertAlmostEqual(float(loss), -1 - (.006 * (-4.) * 98 + .01 * 3. * 2) / n, places=6)
        self.assertEqual(float(kl), 0.)
        adv = torch.cat([torch.full((2,), 50.), torch.ones(98)])
        loss, _ = ppo_terms(logp, old, adv, choosing, selected, entropy, motor_entropy=0, declaration_entropy=0)
        self.assertAlmostEqual(float(loss), -(50. * 2 + 98) / n, places=5)  # per-sample, not one half per group
        loss, kl = ppo_terms(logp, old, adv, choosing, torch.zeros(n, dtype=torch.bool), entropy)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(kl), 0.)


if __name__ == '__main__':
    unittest.main()
