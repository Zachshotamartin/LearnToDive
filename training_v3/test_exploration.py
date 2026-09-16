"""Coherent exploration noise with an exact likelihood, folded input normalisation, stance width."""
import unittest

import numpy as np
import torch

from engine import Arena as Physics
from environment import Arena, HISTORY
from geometry import time_to_surface
from policy import Policy


class NoiseTests(unittest.TestCase):
    def test_likelihood_conditions_on_the_observed_previous_noise(self):
        torch.manual_seed(3)
        e = Arena(3, seed=8, threads=1, stage=0)
        try:
            p = Policy(e.observation_size, (32, 32), noise_rho=.9, architecture='split')
            e.step(e.mask().argmax(1), np.zeros((3, 9)))   # declare, so the next step is a motor step
            obs = torch.tensor(e.observe())
            mask = torch.tensor(e.mask())
            choosing = torch.tensor(e.choosing)
            self.assertFalse(choosing.any())
            out = p(obs, mask, choosing)
            again = p(obs, mask, choosing, out['raw'], out['choice'])
            torch.testing.assert_close(out['logp'], again['logp'])
            shifted = obs.clone()
            shifted[:, -HISTORY:-9] += .5
            other = p(shifted, mask, choosing, out['raw'], out['choice'])
            self.assertFalse(torch.allclose(out['logp'], other['logp']), 'the noise history must change the likelihood')
            still = p(obs, mask, choosing, deterministic=True)
            torch.testing.assert_close(still['noise'], torch.zeros_like(still['noise']))
            # The environment stores the noise so the next observation carries it.
            _, _, _, _ = e.step(out['choice'].numpy(), out['action'].numpy(), out['noise'].numpy())
            np.testing.assert_allclose(e.previous_noise, out['noise'].numpy(), atol=1e-6)
        finally:
            e.close()

    def test_first_stage_declarations_do_not_move_the_body(self):
        e = Arena(2, seed=1, threads=1, stage=0)
        try:
            before = e.physics.state.copy()
            e.step(e.mask().argmax(1), np.zeros((2, 9)), np.ones((2, 9)))
            np.testing.assert_array_equal(e.physics.state[:, 0], 0)
            np.testing.assert_array_equal(e.previous_noise, 0, 'declaration steps carry no motor history')
        finally:
            e.close()


class NormalizationTests(unittest.TestCase):
    def test_statistics_skip_the_history_and_fold_exactly_into_the_export(self):
        torch.manual_seed(5)
        e = Arena(4, seed=2, threads=1, stage=0)
        try:
            p = Policy(e.observation_size, (32, 32), noise_rho=.9, architecture='split', normalize_inputs=True)
            samples = torch.tensor(np.stack([e.observe() for _ in range(3)])) + torch.randn(3, 4, e.observation_size) * .3
            p.update_input_statistics(samples)
            p.update_input_statistics(samples * 2)
            self.assertGreater(float(p.input_count), 0)
            torch.testing.assert_close(p.input_mean[-HISTORY:], torch.zeros(HISTORY, dtype=torch.float64))
            torch.testing.assert_close(p.input_var[-HISTORY:], torch.ones(HISTORY, dtype=torch.float64))
            self.assertTrue((p.input_var[:-HISTORY] > 1e-9).any())
            folded = Policy(e.observation_size, (32, 32), noise_rho=.9, architecture='split')
            folded.load_state_dict({k: torch.tensor(v) for k, v in p.export()['state'].items()}, strict=False)
            obs = torch.tensor(e.observe())
            mask = torch.tensor(e.mask())
            choosing = torch.tensor(e.choosing)
            with torch.no_grad():
                a = p(obs, mask, choosing, deterministic=True)
                b = folded(obs, mask, choosing, deterministic=True)
            torch.testing.assert_close(a['action'], b['action'], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(a['choice'], b['choice'])
            self.assertEqual(p.export()['format'], 'self-declared-diver-v13')
        finally:
            e.close()


class StanceWidthTests(unittest.TestCase):
    def test_zero_spread_reproduces_one_stance_and_full_spread_varies_it(self):
        p = Physics(4, seed=6, threads=1, skills=[0], heights=(10, 10), minimum_heights=[1] * 6, training=True)
        try:
            p.stance_spread = 0.
            p.reset(np.arange(4))
            fixed = p.state.copy()
            np.testing.assert_allclose(fixed[1:, 1:], np.repeat(fixed[:1, 1:], 3, axis=0), atol=1e-9)
            p.stance_spread = 1.
            p.reset(np.arange(4))
            self.assertGreater(np.abs(p.state[1:, 1:] - p.state[:1, 1:]).max(), 1e-3)
        finally:
            p.close()

    def test_time_to_surface_is_kinematic(self):
        self.assertAlmostEqual(float(time_to_surface(0., 0.)), 0.)
        self.assertAlmostEqual(float(time_to_surface(4.905, 0.)), 1., places=5)
        self.assertGreater(float(time_to_surface(2., 3.)), float(time_to_surface(2., -3.)))
        self.assertEqual(float(time_to_surface(100., 0.)), 3.)


if __name__ == '__main__':
    unittest.main()
