"""Progress direction, anti-farming, policy-preserving migration and native PPO."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from environment import Arena
from evaluation import evaluate_targets, evaluate_motor_skills
from motor_curriculum import MotorCurriculum
from motor_objective import GAMMA, potential_components, shaped_reward
from policy import Policy
from rotation_progress import OLD_SIZE, INSERT, ADDED, NEW_SIZE, remaining_turns, rotation_potentials
from rules import IDS, DIVES
from train import Trainer, parser


class RotationProgressTests(unittest.TestCase):
    def test_direction_and_under_over_rotation(self):
        for sign in (-1, 1):
            for actual, expected in [(1., .5), (1.5, 0.), (1.8, -.3)]:
                remain = remaining_turns(sign*1.5, 1., sign*actual, -.7)
                np.testing.assert_allclose(remain, [sign*expected, .3], atol=1e-12)
            potentials = [rotation_potentials(sign*1.5, 1., sign*x, .7)[0] for x in (0, 1, 1.3, 1.5, 1.8, 2.5)]
            self.assertTrue(potentials[0] < potentials[1] < potentials[2] < potentials[3])
            self.assertTrue(potentials[3] > potentials[4] > potentials[5])
            self.assertLess(rotation_potentials(sign*1.5, 1., -sign*.5, .7)[0], 0)

    def test_twist_handedness_and_no_twist_target(self):
        for twist in np.linspace(-2, 2, 25):
            np.testing.assert_allclose(rotation_potentials(1.5, 1, .7, twist),
                                       rotation_potentials(1.5, 1, .7, -twist))
        self.assertLess(rotation_potentials(.5, 0, .1, .05)[1], 0)
        self.assertGreater(rotation_potentials(.5, 1, .1, .75)[1],
                           rotation_potentials(.5, 1, .1, .5)[1])
        self.assertLess(rotation_potentials(.5, 1, .1, 1.5)[1],
                        rotation_potentials(.5, 1, .1, 1.25)[1])

    def test_discounted_loops_holds_and_every_terminal_cancel(self):
        # All paths start with zero potential. Reversing, waiting or falling
        # early cannot change the discounted total shaping reward.
        for path in ([0, .4, .8, .4, 0], [0, 1, 1, 1, 1], [0, .2], [0, 1.5, 3, -2, 1.5]):
            potentials = [rotation_potentials(1.5, 1., x, x/2) for x in path]
            for terminal_name in ('entry', 'failure', 'timeout'):
                total = np.zeros(2)
                for t in range(len(path)-1):
                    total += GAMMA**t * shaped_reward(potentials[t], potentials[t+1], t == len(path)-2)
                np.testing.assert_allclose(total, [0, 0], atol=1e-12, err_msg=terminal_name)

    def test_observations_agree_with_judge_measurements(self):
        old = MotorCurriculum(Arena(1, 3, 1, training=False), enabled=False)
        new = MotorCurriculum(Arena(1, 3, 1, training=False, rotation_progress=True), enabled=False)
        try:
            self.assertEqual(new.observation_size, NEW_SIZE)
            np.testing.assert_array_equal(new.observe()[:, INSERT:INSERT+2], 0)
            for code in ('103C', '203C', '303C', '403C', '5132D'):
                for env in (old, new):
                    env.group[:] = DIVES[IDS[code]]['group']
                    env.height[:], env.apparatus[:] = 10, 1
                    env.used[:] = False
                    env.base.declare(0, IDS[code])
                    env.physics.phase_theta[:] = -.75 * 2*np.pi
                    env.physics.air_twist[:] = -.3 * 2*np.pi
                    env.previous_actions[:] = np.linspace(-.8, .8, 9)
                obs = new.observe()
                np.testing.assert_array_equal(np.delete(obs, [INSERT, INSERT+1], axis=1), old.observe())
                d=DIVES[IDS[code]]
                np.testing.assert_allclose(obs[0, INSERT:INSERT+2]*5, [d['sign']*d['turns']+.75, d['twists']-.3], atol=1e-6)
                np.testing.assert_array_equal(obs[0, -9:], new.previous_actions[0])
                parts=potential_components(new.base)
                np.testing.assert_allclose([parts['somersaultProgress'][0], parts['twistProgress'][0]],
                                           rotation_potentials(d['sign']*d['turns'], d['twists'], -.75, -.3))
        finally:
            old.close(); new.close()

    def args(self, folder, rotation=False):
        args=parser().parse_args(['--output', str(folder), '--envs', '4', '--threads', '1', '--widths', '16', '16',
            '--horizon', '16', '--steps', '64', '--epochs', '1', '--batch', '32', '--evaluate-every', '0',
            '--motor-curriculum', 'adaptive', '--architecture', 'split', '--reward-mode', 'completion-first',
            '--direction-practice', '--goal-practice'])
        args.rotation_progress=rotation
        return args

    def test_migrate_adam_behavior_rollout_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            old=Trainer(self.args(root/'old'))
            try:
                old.update()
                saved=old.persist('paused')
            finally:
                old.env.close()
            args=self.args(root/'new', True); args.continue_from=str(root/'old/latest.pt')
            new=Trainer(args)
            try:
                self.assertEqual(new.state['steps'], saved['training']['steps'])
                state=new.policy.state_dict()
                for name, value in saved['model'].items():
                    actual=state[name]
                    if actual.shape != value.shape and actual.ndim == 1:
                        # Input statistics gain a neutral entry per inserted column.
                        fill = 1. if name == 'input_var' else 0.
                        self.assertEqual(actual.shape[0], NEW_SIZE)
                        torch.testing.assert_close(actual[INSERT:INSERT+2], torch.full((2,), fill, dtype=actual.dtype), atol=0, rtol=0)
                        actual=torch.cat([actual[:INSERT], actual[INSERT+2:]])
                    elif actual.shape != value.shape:
                        self.assertEqual(actual.shape[1], NEW_SIZE)
                        torch.testing.assert_close(actual[:, INSERT:INSERT+2], torch.zeros_like(actual[:, INSERT:INSERT+2]), atol=0, rtol=0)
                        actual=torch.cat([actual[:, :INSERT], actual[:, INSERT+2:]],1)
                    torch.testing.assert_close(actual,value,atol=0,rtol=0)
                current=new.optimizer.state_dict()
                for pid, fields in saved['optimizer']['state'].items():
                    for name,value in fields.items():
                        actual=current['state'][pid][name]
                        if actual.shape != value.shape:
                            torch.testing.assert_close(actual[:,INSERT:INSERT+2],torch.zeros_like(actual[:,INSERT:INSERT+2]),atol=0,rtol=0)
                            actual=torch.cat([actual[:,:INSERT],actual[:,INSERT+2:]],1)
                        torch.testing.assert_close(actual,value,atol=0,rtol=0)
                # No behavior change even when the new features are nonzero.
                previous=Policy(OLD_SIZE,(16,16),architecture='split',noise_rho=saved['config']['noise_rho'],normalize_inputs=bool(saved['config']['input_normalization'])); previous.load_state_dict(saved['model'])
                obs=torch.randn(32,NEW_SIZE)*.2
                old_obs=torch.cat([obs[:,:INSERT],obs[:,INSERT+2:]],1)
                mask=torch.ones(32,len(DIVES),dtype=torch.bool); choosing=torch.zeros(32,dtype=torch.bool)
                with torch.no_grad():
                    a=previous(old_obs,mask,choosing,deterministic=True)
                    b=new.policy(obs,mask,choosing,deterministic=True)
                for key in ('action','raw','value','logp'):
                    torch.testing.assert_close(a[key],b[key],atol=2e-6,rtol=2e-6)
                self.assertTrue(new.env.choosing.all())
                for name in ('goal_mastery','goal_visits','direction_visits','readiness','level'):
                    np.testing.assert_array_equal(getattr(new.env,name),saved['environment']['curriculum'][name])
                # Critic-only calibration then an actual actor update.
                for _ in range(5): new.update()
                self.assertEqual(new.state['steps'],saved['training']['steps']+5*64)
                self.assertTrue(torch.isfinite(new.policy.trunk[0].weight).all())
                self.assertTrue(torch.count_nonzero(new.policy.trunk[0].weight[:,INSERT:INSERT+2])>0)
                latest=new.persist('paused')
            finally:
                new.env.close()
            resumed_args=self.args(root/'new',True); resumed_args.resume=str(root/'new/latest.pt')
            resumed=Trainer(resumed_args)
            try:
                self.assertEqual(resumed.state['steps'],latest['training']['steps'])
                np.testing.assert_array_equal(resumed.env.observe(),np.nan_to_num(resumed.env.observe()))
                for name,value in latest['model'].items():
                    torch.testing.assert_close(resumed.policy.state_dict()[name],value,atol=0,rtol=0)
                report=evaluate_targets(resumed.policy,cases_per_target=1,reward_mode='completion-first')
                self.assertEqual(len(report['episodes']),10)
                self.assertTrue(all('somersaultProgress' in r['potentialShaping'] for r in report['episodes']))
                evaluate_motor_skills(resumed.policy,cases=1,direction_practice=True)
            finally:
                resumed.env.close()

if __name__=='__main__': unittest.main()
