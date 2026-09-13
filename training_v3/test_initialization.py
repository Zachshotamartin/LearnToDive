"""Fresh actor transfer must preserve policy weights, not old training state."""
import tempfile
import unittest
from pathlib import Path

import torch

from checkpointing import hashes
from policy import Policy
from train import FORMAT, Trainer, parser


class InitializationTests(unittest.TestCase):
    def test_actor_transfer_and_physics_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            options = ['--output', str(root / 'fresh'), '--envs', '2', '--threads', '1',
                       '--widths', '32', '32', '--reward-mode', 'continuous-entry']
            baseline = Trainer(parser().parse_args(options))
            size = baseline.env.observation_size
            baseline.env.close()
            model = Policy(size, (32, 32), .6)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(.123)
            saved = dict(contract=dict(format=FORMAT, observationSize=size, sourceHashes=hashes()),
                         model=model.state_dict(), training=dict(steps=12345))
            checkpoint = root / 'source.pt'
            torch.save(saved, checkpoint)
            args = parser().parse_args(options + ['--initialize-from', str(checkpoint)])
            trainer = Trainer(args)
            try:
                for name, value in trainer.policy.state_dict().items():
                    if not name.startswith('value'):
                        torch.testing.assert_close(value, saved['model'][name], atol=0, rtol=0)
                self.assertTrue(any(not torch.equal(value, saved['model'][name])
                                    for name, value in trainer.policy.state_dict().items()
                                    if name.startswith('value')))
                self.assertEqual(trainer.state['steps'], 0)
                self.assertEqual(len(trainer.optimizer.state), 0)
            finally:
                trainer.env.close()
            saved['contract']['sourceHashes']['diver.xml'] = 'incompatible'
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, 'physical model'):
                Trainer(args)

    def test_conflicting_sources_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = parser().parse_args(['--output', folder, '--envs', '1', '--threads', '1',
                                        '--initialize-from', 'first.pt', '--warm-start', 'second.pt'])
            with self.assertRaisesRegex(ValueError, 'only one'):
                Trainer(args)
