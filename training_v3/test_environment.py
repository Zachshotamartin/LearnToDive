"""Declaration, physics stepping, exact state restore and the untrained prior."""
import unittest

import numpy as np
import torch

from environment import Arena
from policy import Policy
from rules import IDS

SUPPORT_COLUMNS = slice(4 + 6 + 14 + 14 + 9 + 3 + 3 + 13, 4 + 6 + 14 + 14 + 9 + 3 + 3 + 13 + 4)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.e = Arena(4, seed=4, threads=1)

    def tearDown(self):
        self.e.close()

    def test_selection_physics_and_resume(self):
        e = self.e
        choices = e.mask().argmax(1)
        observation, _, _, _ = e.step(choices, np.zeros((4, 9)))
        np.testing.assert_array_equal(e.physics.state[:, 0], 0)
        self.assertFalse(e.choosing.any())
        self.assertTrue(np.isfinite(observation).all())
        state = e.state_dict()
        action = np.full((4, 9), .2)
        x = e.step(choices, action)
        e.load_state_dict(state)
        y = e.step(choices, action)
        for i in range(3):
            np.testing.assert_allclose(x[i], y[i], rtol=0, atol=0)

    def test_policy_likelihood_and_popart(self):
        e = self.e
        m = Policy(e.observation_size, (32, 32))
        o = torch.tensor(e.observe())
        mask = torch.tensor(e.mask())
        choose = torch.tensor(e.choosing)
        out = m(o, mask, choose)
        again = m(o, mask, choose, out['raw'], out['choice'])
        torch.testing.assert_close(out['logp'], again['logp'])
        v = out['value'].detach()
        m.update_value_scale(torch.tensor([10., 20., 40., 80.]))
        torch.testing.assert_close(v, m(o, mask, choose)['value'], atol=1e-5, rtol=1e-5)
        again = m(o, mask, choose)
        loss = -again['logp'].mean() + again['value'].square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_rigid_platform(self):
        e = self.e
        e.apparatus[:] = 1
        e.height[:] = 5
        e.group[:] = 1
        e.used[:] = False
        e.step(np.full(4, IDS['101C']), np.zeros((4, 9)))
        for _ in range(20):
            e.step(np.zeros(4, int), np.zeros((4, 9)))
        self.assertLess(np.abs(e.physics.state[:, 1]).max(), .002)


class PriorTests(unittest.TestCase):
    def test_untrained_policy_holds_the_reset_stance_and_observes_support(self):
        torch.manual_seed(1)
        e = Arena(2, seed=6, threads=1, training=False)
        self.assertEqual(e.observation_size, 234)
        o = e.observe()
        support = o[:, SUPPORT_COLUMNS]
        self.assertTrue((support[:, :2] == 1).all(), 'both feet start supported on springboard and platform alike')
        self.assertTrue(np.all(np.abs(support[:, 2] - np.where(e.apparatus == 1, 0, -.05123 * 4)) < 1e-3))
        m = Policy(e.observation_size, (32, 32), initial_action=e.initial_action)
        previous = np.zeros(9, np.float32)
        for _ in range(12):
            obs = torch.tensor(np.concatenate([o[:1, :-9], previous[None]], axis=1))
            out = m(obs, torch.tensor(e.mask()[:1]), torch.tensor(e.choosing[:1]), deterministic=True)
            previous = out['action'][0].detach().numpy()
        np.testing.assert_allclose(previous, np.clip(e.initial_action, -.95, .95), atol=.06)
        e.close()


if __name__ == '__main__':
    unittest.main()
