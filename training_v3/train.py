"""Hybrid PPO: category-conditioned self-declarations and physical motors.

SIGTERM checkpoints at an update boundary. Resume preserves the simulator,
optimizer, RNG, recovery bank and routine state exactly. Evaluation never
publishes browser assets.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch

import mastery
from checkpointing import atomic_checkpoint, atomic_json, hashes
from environment import Arena
from evaluation import evaluate, summary, evaluate_motor_skills, evaluate_targets
from model_selection import update_selection, comparison, competence
from judge import VERSION
from losses import ppo_terms
from policy import FORMAT, MOTOR_FORMAT, FINAL_FORMAT, Policy
from motor_curriculum import MotorCurriculum, EXTRA_OBSERVATIONS
from motor_objective import DIRECTION_VERSION, COMPLETION_VERSION, CONJUNCTIVE_VERSION
from rules import DIVES, PRACTICE_SCOPE
from reference_policies import load_reference
from water import WATER_VERSION

__all__ = ['train', 'parser', 'Trainer', 'evaluate', 'summary', 'atomic_json', 'atomic_checkpoint', 'hashes']

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
GAMMA = .995
LAMBDA = .95
KL_LIMIT = .02              # mean approximate KL per epoch that ends the update early
GRADIENT_NORM = .5
REFINEMENT_CLEAN_RATE = .8  # held-out clean rate above which exploration is reduced
REFINEMENT_FACTOR = .25
PLATEAU_THRESHOLDS = [('points', .5), ('execution', .1), ('clean', .02), ('valid', .05)]
RESUME_KEYS = ['envs', 'widths', 'rho', 'horizon', 'seed', 'lr', 'epochs', 'batch', 'reward_mode', 'gae_lambda', 'recovery_mode', 'exploration', 'motor_curriculum', 'architecture', 'practice', 'reference_policy', 'eval_cases', 'final_cases', 'final_seed', 'direction_practice', 'goal_practice',
               'noise_rho', 'input_normalization', 'stage_curriculum', 'start_stage', 'gates']
BEST_KEY_SIZE = 4
# Gates: a run that has not produced these by the given step count stops for a
# change instead of waiting for a plateau or the budget.
GATES = (dict(name='entry-skill-and-first-clean-dive', steps=10_240_000, entrySuccess=.5, cleanDives=1),
         dict(name='clean-rate-on-current-scope', steps=30_720_000, cleanRate=.2))


def estimate_advantages(batch, gae_lambda=LAMBDA):
    """GAE for motor steps; a full n-step return to the episode boundary for declarations.

    Declaration is a sparse high-level decision, so it does not use GAE's
    exponentially attenuated motor credit.
    """
    rewards, values, dones, choosing, bootstrap = (batch[k] for k in ['rewards', 'values', 'dones', 'choosing', 'bootstrap'])
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N)
    for t in reversed(range(T)):
        future = bootstrap if t == T - 1 else values[t + 1]
        live = 1 - dones[t]
        delta = rewards[t] + GAMMA * future * live - values[t]
        gae = delta + GAMMA * gae_lambda * live * gae
        advantages[t] = gae
    returns = advantages + values
    decision_return = bootstrap.clone()
    for t in reversed(range(T)):
        decision_return = rewards[t] + GAMMA * decision_return * (1 - dones[t])
        advantages[t] = torch.where(choosing[t], decision_return - values[t], advantages[t])
        returns[t] = torch.where(choosing[t], decision_return, returns[t])
    return advantages, returns


def transfer_warm_start(policy, args, out):
    """Copy a v10 actor's physical-feature weights into the new trunk and motor head.

    Only explicitly mapped physical features and existing motor outputs
    transfer. Old prescribed-target columns and the critic are not transferred
    as new experience.
    """
    old = torch.load(args.warm_start, map_location='cpu', weights_only=False)['model']
    width = old['actor.0.weight'].shape[0]
    if width > min(args.widths) or len(args.widths) != 2:
        raise ValueError('Warm start requires two layers at least as wide as the source')
    mapping = {**{i: i for i in range(22)}, **{22 + i: 24 + i for i in range(12)}, **{34 + i: 38 + i for i in range(8)},
               **{42 + i: 47 + i for i in range(6)}}
    with torch.no_grad():
        first, second = policy.trunk[0], policy.trunk[2]
        first.weight[:width].zero_()
        first.bias[:width] = old['actor.0.bias']
        for previous, current in mapping.items():
            first.weight[:width, current] = old['actor.0.weight'][:, previous]
        second.weight[:width].zero_()
        second.weight[:width, :width] = old['actor.2.weight']
        second.bias[:width] = old['actor.2.bias']
        policy.motor.weight.zero_()
        policy.motor.weight[:, :width] = old['actor.4.weight']
        policy.motor.bias.copy_(old['actor.4.bias'])
    source = Path(args.warm_start).resolve()
    atomic_json(out / 'warm-start.json', dict(source=str(source), sourceSHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                              mapping=mapping, newObjectiveSteps=0, optimizer='fresh',
                                              declaration='newly initialized', critic='fresh', notEquivalentToOldPolicy=True))


class Trainer:
    """One resumable PPO run writing into one output folder."""

    def __init__(self, args):
        self.args = args
        if (args.direction_practice or args.goal_practice) and (args.motor_curriculum != 'adaptive' or args.reward_mode not in ('phase-dense', 'completion-first', 'conjunctive')):
            raise ValueError('Direction practice requires the adaptive phase-dense curriculum')
        if not 0 <= args.gae_lambda <= 1:
            raise ValueError('GAE lambda must be between zero and one')
        if args.final_seed == 771100 or args.final_cases < 1:
            raise ValueError('Final evaluation requires a disjoint seed and positive cohort size')
        self.out = Path(args.output).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / 'latest.pt').exists() and not args.resume:
            raise ValueError('Use --resume; never overwrite a saved run')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.progression = mastery.Progression(args.start_stage)
        stage = self.progression.index if args.stage_curriculum else None
        self.env = Arena(args.envs, args.seed, args.threads, practice=args.practice, reward_mode=args.reward_mode, recovery_mode=args.recovery_mode,
                         rotation_progress=getattr(args, 'rotation_progress', False), stage=stage)
        if args.motor_curriculum != 'off':
            self.env = MotorCurriculum(self.env, enabled=args.motor_curriculum == 'adaptive', direction_practice=args.direction_practice,
                                       goal_practice=args.goal_practice)
        try:
            self.policy = Policy(self.env.observation_size, tuple(args.widths), args.rho,
                                 initial_action=None if getattr(args, 'random_motor_init', False) else self.env.initial_action,
                                 exploration=args.exploration, architecture=args.architecture,
                                 noise_rho=args.noise_rho, normalize_inputs=args.input_normalization)
            if getattr(args, 'random_motor_init', False) and any((args.initialize_from, args.warm_start, args.continue_from)):
                raise ValueError('Random initialization cannot load a checkpoint')
            if sum(bool(path) for path in (args.initialize_from, args.resume, args.warm_start, args.continue_from)) > 1:
                raise ValueError('Choose only one initialization, warm-start, or exact resume source')
            if args.initialize_from and not args.resume:
                saved = torch.load(args.initialize_from, map_location='cpu', weights_only=False)
                old_size = saved['contract']['observationSize']
                added_context = self.env.observation_size - old_size
                if saved['contract']['format'] not in (FORMAT, MOTOR_FORMAT, FINAL_FORMAT) or added_context not in (0, EXTRA_OBSERVATIONS):
                    raise ValueError('Initialization requires the same physical observation and policy format')
                physical_files = ('diver.xml', 'geometry.py', 'stance.py', 'water.py')
                current_hashes = hashes()
                if any(saved['contract']['sourceHashes'].get(name) != current_hashes[name]
                       for name in physical_files):
                    raise ValueError('Initialization requires the same physical model and measurements')
                fresh = self.policy.state_dict()
                # Input statistics are not actor weights; a transferred actor starts fresh ones.
                transferred = {k: v for k, v in saved['model'].items() if not k.startswith(('value', 'critic_trunk.', 'input_'))}
                if added_context:
                    # Insert task goals before the final history columns.
                    # Zero new weights preserve the old motor/choice functions.
                    from policy import HISTORY
                    for key in ('trunk.0.weight', 'selector_trunk.0.weight'):
                        if key in transferred:
                            old = transferred[key]
                            new = torch.zeros(old.shape[0], self.env.observation_size)
                            new[:, :old_size - HISTORY] = old[:, :-HISTORY]
                            new[:, -HISTORY:] = old[:, -HISTORY:]
                            transferred[key] = new
                if args.architecture == 'split' and not any(k.startswith('selector_trunk.') for k in transferred):
                    transferred.update({k.replace('trunk.', 'selector_trunk.', 1): v.clone()
                                        for k, v in list(transferred.items()) if k.startswith('trunk.')})
                fresh.update(transferred)
                self.policy.load_state_dict(fresh)
                atomic_json(self.out / 'initialization.json', dict(
                    sourceSHA256=hashlib.sha256(Path(args.initialize_from).read_bytes()).hexdigest(),
                    sourceSteps=saved['training']['steps'], actorTransferred=True, optimizer='fresh',
                    valueHead='fresh', environment='fresh', newObjectiveSteps=0,
                    insertedMotorContextColumns=added_context, architecture=args.architecture))
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr, eps=1e-5)
            self.state = dict(steps=0, updates=0, elapsedSeconds=0, history=[], evaluations=[], bestValue=-1e30, stopReason=None)
            self.contract = dict(format=self.policy.format, judge=VERSION, water=WATER_VERSION, observationSize=self.env.observation_size,
                                 motorCurriculum=args.motor_curriculum, architecture=args.architecture,
                                 rewardMode=args.reward_mode, exploration=args.exploration, directionPractice=args.direction_practice, goalPractice=args.goal_practice,
                                 boundedPractice=PRACTICE_SCOPE, completionFeedback=COMPLETION_VERSION,
                                 conjunctiveCredit=CONJUNCTIVE_VERSION if args.reward_mode == 'conjunctive' else None,
                                 noiseRho=args.noise_rho, inputNormalization=args.input_normalization,
                                 stageCurriculum=[s['name'] for s in mastery.STAGES] if args.stage_curriculum else None,
                                 actions=9, declarations=[d['id'] for d in DIVES], sourceHashes=hashes())
            if getattr(args, 'rotation_progress', False):
                from rotation_progress import VERSION as ROTATION_VERSION
                self.contract['rotationProgress'] = ROTATION_VERSION
            if args.resume:
                self.resume(args.resume)
            elif args.continue_from:
                self.continue_phase(args.continue_from)
            elif args.warm_start:
                transfer_warm_start(self.policy, args, self.out)
        except BaseException:
            self.env.close()
            raise
        self.stop = False
        self.started = time.monotonic()
        self.prior_seconds = self.state['elapsedSeconds']
        self.block = args.envs * args.horizon
        self.target = self.state['steps'] + math.ceil(args.steps / self.block) * self.block

    # ------------------------------------------------------------ persistence
    def continue_phase(self, path):
        """Explicit objective migration, preserving learned weights and Adam state.

        This is a continuation with new experiences, not an exact resume of the
        old learning problem. It must write a separate directory. No prior
        archive is overwritten and cumulative training counters never reset.
        """
        raw = Path(path).read_bytes()
        import io
        saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        if Path(path).resolve().parent == self.out:
            raise ValueError('A new phase must preserve the previous output directory')
        if getattr(self.args, 'rotation_progress', False) and not saved['contract'].get('rotationProgress'):
            return self.continue_rotation_phase(saved, raw, path)
        if self.args.reward_mode == 'completion-first':
            return self.continue_completion_phase(saved, raw, path)
        if not self.args.direction_practice or saved['config'].get('direction_practice', False):
            raise ValueError('Only the v14 to directional-practice migration is supported')
        for key in RESUME_KEYS:
            if key not in ('direction_practice', 'goal_practice') and saved['config'][key] != getattr(self.args, key):
                raise ValueError('Continuation changed ' + key)
        for key in ('format', 'observationSize', 'architecture', 'actions', 'declarations', 'judge', 'water'):
            if saved['contract'][key] != self.contract[key]:
                raise ValueError('Incompatible continuation contract: ' + key)
        for name in ('diver.xml', 'geometry.py', 'stance.py', 'water.py', 'policy.py', 'rules.py', 'judge.py'):
            if saved['contract']['sourceHashes'][name] != self.contract['sourceHashes'][name]:
                raise ValueError('Continuation must preserve physics, policy and judging: ' + name)
        if self.args.architecture != 'split':
            raise ValueError('Critic recalibration requires independent actor and critic trunks')
        self.policy.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.state = copy.deepcopy(saved['training'])
        # Restore the episode generators and unaffected skill progress. Start
        # fresh episodes because old partial rewards/tasks have different goals.
        old = saved['environment']
        self.env.base.rng.bit_generator.state = copy.deepcopy(old['base']['rng'])
        self.env.physics.rng.bit_generator.state = copy.deepcopy(old['base']['physicsRng'])
        self.env.base.interactions = old['base']['interactions']
        for name in ('visits', 'successes', 'readiness', 'level', 'error_sum'):
            getattr(self.env, name)[:] = old['curriculum'][name]
            getattr(self.env, name)[1] = 0  # Old jump-only success is not directional mastery.
        self.env.base.reset(np.arange(self.env.n), new_routine=True)
        self.env.assign(np.arange(self.env.n))
        torch.set_rng_state(saved['torchRNG'])
        np.random.set_state(saved['numpyRNG'])
        random.setstate(saved['pythonRNG'])
        phase = dict(version=DIRECTION_VERSION, goalPractice=self.args.goal_practice,
                     startSteps=self.state['steps'], startUpdates=self.state['updates'],
                     parent=str(Path(path).resolve()), parentSHA256=hashlib.sha256(raw).hexdigest(),
                     actor='preserved', optimizer='preserved', critic='preserved; four critic-only recalibration updates',
                     episodeState='fresh; partial rollouts and recovery snapshots remain in parent checkpoint',
                     unaffectedMotorProgress='preserved', takeoffReadiness='reset for changed skill criterion',
                     sourceHashes=self.contract['sourceHashes'])
        self.state.setdefault('trainingPhases', []).append(phase)
        self.state['phaseStartSteps'] = self.state['steps']
        self.state['criticWarmupRemaining'] = 4
        atomic_json(self.out / 'continuation.json', phase)

    def continue_completion_phase(self, saved, raw, path):
        """Change the reward without resetting any learned network or optimizer.

        Fresh episode boundaries are necessary: unfinished old-reward rollouts
        and potential accumulators cannot become new-objective experience.
        Preserved parent checkpoints retain every original episode/archive.
        """
        if saved['config']['reward_mode'] != 'phase-dense' or not self.args.goal_practice:
            raise ValueError('Completion-first continuation requires existing specific-dive training')
        for key in RESUME_KEYS:
            if key != 'reward_mode' and saved['config'][key] != getattr(self.args, key):
                raise ValueError('Completion continuation changed ' + key)
        old_contract = {k: v for k, v in saved['contract'].items() if k not in ('sourceHashes', 'rewardMode')}
        new_contract = {k: v for k, v in self.contract.items() if k not in ('sourceHashes', 'rewardMode')}
        if old_contract != new_contract:
            raise ValueError('Completion continuation changed the model contract')
        allowed = {'train.py', 'training_reward.py', 'motor_objective.py', 'environment.py', 'motor_curriculum.py'}
        if set(saved['contract']['sourceHashes']) != set(self.contract['sourceHashes']):
            raise ValueError('Completion continuation changed the source inventory')
        for name, digest in self.contract['sourceHashes'].items():
            if name not in allowed and saved['contract']['sourceHashes'][name] != digest:
                raise ValueError('Completion continuation changed protected source: ' + name)
        if self.args.architecture != 'split':
            raise ValueError('Critic recalibration requires independent actor and critic trunks')
        self.policy.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.state = copy.deepcopy(saved['training'])
        self.env.load_state_dict(saved['environment'])
        # Retain all motor readiness and visit counts. Only the target success
        # averages change meaning under completion-first and must be relearned.
        self.env.goal_mastery[:] = 0
        self.env.goal_progress[:] = 0
        self.env.base.bank = [[] for _ in self.env.base.bank]
        self.env.base.recovery_archive = type(self.env.base.recovery_archive)()
        self.env.base.reset(np.arange(self.env.n), new_routine=True)
        self.env.assign(np.arange(self.env.n))
        torch.set_rng_state(saved['torchRNG'])
        np.random.set_state(saved['numpyRNG'])
        random.setstate(saved['pythonRNG'])
        phase = dict(version=COMPLETION_VERSION, startSteps=self.state['steps'],
                     startUpdates=self.state['updates'], parent=str(Path(path).resolve()),
                     parentSHA256=hashlib.sha256(raw).hexdigest(), actor='preserved', optimizer='preserved',
                     critic='preserved; four critic-only recalibration updates',
                     episodeState='fresh at reward boundary; old episodes retained in parent',
                     motorProgress='preserved', goalMastery='reset because its success criterion changed',
                     sourceHashes=self.contract['sourceHashes'])
        self.state.setdefault('trainingPhases', []).append(phase)
        self.state['phaseStartSteps'] = self.state['steps']
        self.state['criticWarmupRemaining'] = 4
        atomic_json(self.out / 'continuation.json', phase)

    def continue_rotation_phase(self, saved, raw, path):
        """Add two zero-weight inputs and dense feedback to the current learner.

        Every existing parameter and Adam moment survives. Fresh episode starts
        prevent mixing partial returns from the two reward definitions.
        """
        from rotation_progress import VERSION, OLD_SIZE, NEW_SIZE, INSERT, ADDED, migrated_states
        if (saved['contract']['observationSize'] != OLD_SIZE or self.env.observation_size != NEW_SIZE
                or self.args.architecture != 'split' or self.args.reward_mode != 'completion-first'):
            raise ValueError('Rotation continuation requires the existing completion-first split motor model')
        for key in RESUME_KEYS:
            if saved['config'][key] != getattr(self.args, key):
                raise ValueError('Rotation continuation changed ' + key)
        excluded = {'sourceHashes', 'observationSize', 'rotationProgress'}
        if ({k: v for k, v in saved['contract'].items() if k not in excluded}
                != {k: v for k, v in self.contract.items() if k not in excluded}):
            raise ValueError('Rotation continuation changed a protected contract field')
        old_hashes, new_hashes = saved['contract']['sourceHashes'], self.contract['sourceHashes']
        allowed = {'environment.py', 'motor_objective.py', 'engine.py', 'evaluation.py', 'train.py'}
        if set(new_hashes) != set(old_hashes) | {'rotation_progress.py'}:
            raise ValueError('Rotation continuation changed source inventory')
        for name in old_hashes:
            if name not in allowed and old_hashes[name] != new_hashes[name]:
                raise ValueError('Rotation continuation changed protected source: ' + name)
        model, optimizer = migrated_states(saved['model'], saved['optimizer'], self.policy.named_parameters())
        self.policy.load_state_dict(model)
        self.optimizer.load_state_dict(optimizer)
        self.state = copy.deepcopy(saved['training'])
        old_environment = copy.deepcopy(saved['environment'])
        for name in ('phase_previous', 'phase_totals'):
            old_environment['base']['arrays'][name] = np.zeros_like(getattr(self.env.base, name))
        self.env.load_state_dict(old_environment)
        # Physics, curriculum mastery, counters, and random generators survive;
        # old recovery snapshots and their return histories stay in the parent.
        self.env.base.bank = [[] for _ in self.env.base.bank]
        self.env.base.recovery_archive = type(self.env.base.recovery_archive)()
        self.env.base.reset(np.arange(self.env.n), new_routine=True)
        self.env.assign(np.arange(self.env.n))
        torch.set_rng_state(saved['torchRNG'])
        np.random.set_state(saved['numpyRNG'])
        random.setstate(saved['pythonRNG'])
        phase = dict(version=VERSION, startSteps=self.state['steps'], startUpdates=self.state['updates'],
                     parent=str(Path(path).resolve()), parentSHA256=hashlib.sha256(raw).hexdigest(),
                     actor='all old weights preserved; two zero-initialized inputs',
                     optimizer='all old Adam moments and steps preserved; new input moments zero',
                     critic='preserved; four critic-only calibration updates',
                     insertedObservationColumns=[INSERT, INSERT + ADDED],
                     episodeState='fresh boundaries; old partial rollouts remain in parent checkpoint',
                     motorProgress='preserved', sourceHashes=self.contract['sourceHashes'])
        self.state.setdefault('trainingPhases', []).append(phase)
        self.state['phaseStartSteps'] = self.state['steps']
        self.state['criticWarmupRemaining'] = 4
        atomic_json(self.out / 'continuation.json', phase)

    def resume(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        amendments = self.source_amendments(saved['contract'], saved['training']['steps'])
        if amendments is None:
            raise ValueError('Changed source or contract; exact resume refused')
        for key in RESUME_KEYS:
            if saved['config'][key] != getattr(self.args, key):
                raise ValueError('Resume changed ' + key)
        self.policy.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.env.load_state_dict(saved['environment'])
        self.state = copy.deepcopy(saved['training'])
        if 'stages' in self.state:
            self.progression.load_state_dict(self.state['stages'])
            if self.args.stage_curriculum:
                self.base_environment().set_stage(self.progression.index)
        if amendments:
            # The change is explicit on the command line and permanent in the checkpoint.
            self.state.setdefault('sourceAmendments', []).extend(amendments)
        torch.set_rng_state(saved['torchRNG'])
        np.random.set_state(saved['numpyRNG'])
        random.setstate(saved['pythonRNG'])

    def source_amendments(self, saved_contract, at_steps):
        """Differences between the saved and current contract, or None when they are not all accepted.

        Exact resume refuses any change to a learning file. A bug fix in a file
        that is hashed but does not change the learning problem (for example the
        evaluation loop's tick budget) can be accepted explicitly with
        --accept-source-change, which records the old and new hashes.
        """
        if saved_contract == self.contract:
            return []
        saved_hashes = saved_contract.get('sourceHashes', {})
        current_hashes = self.contract['sourceHashes']
        others = {k: v for k, v in saved_contract.items() if k != 'sourceHashes'}
        if others != {k: v for k, v in self.contract.items() if k != 'sourceHashes'} or set(saved_hashes) != set(current_hashes):
            return None
        changed = sorted(name for name in current_hashes if saved_hashes[name] != current_hashes[name])
        accepted = set(self.args.accept_source_change or [])
        if not changed or not set(changed) <= accepted:
            return None
        return [dict(file=name, before=saved_hashes[name], after=current_hashes[name], atSteps=at_steps)
                for name in changed]

    def base_environment(self):
        return self.env.base if isinstance(self.env, MotorCurriculum) else self.env

    def persist(self, reason):
        state = self.state
        state['elapsedSeconds'] = self.prior_seconds + time.monotonic() - self.started
        state['stopReason'] = reason
        state['stages'] = self.progression.state_dict()
        saved = dict(contract=self.contract, config=vars(self.args), model=self.policy.state_dict(),
                     optimizer=self.optimizer.state_dict(), environment=self.env.state_dict(), training=copy.deepcopy(state),
                     torchRNG=torch.get_rng_state(), numpyRNG=np.random.get_state(), pythonRNG=random.getstate())
        atomic_checkpoint(self.out / 'latest.pt', saved)
        atomic_json(self.out / 'STATUS.json', dict(pid=os.getpid(), phase=reason, steps=state['steps'],
                                                   elapsedSeconds=state['elapsedSeconds'], updates=state['updates'],
                                                   bestValue=state['bestValue'],
                                                   latestMetrics=state['history'][-1] if state['history'] else None,
                                                   stage=self.progression.state_dict(), gates=state.get('gates', []),
                                                   publication='Not qualified for browser publication'))
        return saved

    def save_sources(self):
        (self.out / 'source').mkdir(exist_ok=True)
        for name in self.contract['sourceHashes']:
            shutil.copy2(HERE / name, self.out / 'source' / name)

    def request_stop(self, sig, frame):
        self.stop = True

    # ------------------------------------------------------------- evaluation
    def initialize_references(self):
        if not self.args.reference_policy:
            return
        if 'referenceInputs' in self.state:
            for item in self.state['referenceInputs']:
                load_reference(item['path'], item['sha256'])
            return
        inputs, reports = [], []
        for path in self.args.reference_policy:
            model, identity = load_reference(path)
            report = evaluate(model, cases=self.args.eval_cases, reward_mode=self.args.reward_mode)
            report['reference'] = identity
            inputs.append(identity)
            reports.append(report)
        self.state['referenceInputs'], self.state['referenceReports'] = inputs, reports
        atomic_json(self.out / 'reference-evaluations.json', reports)

    def final_test(self):
        """One disjoint-cohort test per run; never automatically publish its model."""
        state = self.state
        if not state['selection']['eligibleForFinalTest'] or 'finalTest' in state:
            return
        candidate = evaluate(self.policy, seed=self.args.final_seed, cases=self.args.final_cases,
                             reward_mode=self.args.reward_mode)
        decisions = []
        for item in state['referenceInputs']:
            model, _ = load_reference(item['path'], item['sha256'])
            reference = evaluate(model, seed=self.args.final_seed, cases=self.args.final_cases,
                                 reward_mode=self.args.reward_mode)
            decisions.append(dict(reference=item, comparison=comparison(candidate, reference)))
        ability = competence(candidate)
        qualified = ability['qualified'] and all(d['comparison']['eligibleForReview'] for d in decisions)
        state['finalTest'] = dict(steps=state['steps'], candidate=candidate, competence=ability,
                                 comparisons=decisions, qualified=qualified, automaticPublication=False)
        atomic_json(self.out / 'final-test.json', state['finalTest'])
        if qualified:
            atomic_checkpoint(self.out / 'qualified-for-review.pt', self.persist('qualified-awaiting-review'))

    def check_gates(self, report):
        """Record every gate whose step count has been reached; returns the first failed one."""
        state = self.state
        results = state.setdefault('gates', [])
        checked = {row['name'] for row in results}
        clean = report['summary']['full']['clean'] or 0.
        clean_dives = sum(1 for row in report['episodes'] if row['clean'] and not row['practice'])
        entry_success = (report.get('motorSkills', {}).get('tasks', {}).get('entry', {}).get('success'))
        failed = None
        for gate in GATES:
            if gate['name'] in checked or state['steps'] < gate['steps']:
                continue
            reasons = []
            if 'entrySuccess' in gate and (entry_success is None or entry_success < gate['entrySuccess']):
                reasons.append(f"entry sub-task success {entry_success} below {gate['entrySuccess']}")
            if 'cleanDives' in gate and clean_dives < gate['cleanDives']:
                reasons.append(f"{clean_dives} clean dives, at least {gate['cleanDives']} required")
            if 'cleanRate' in gate and clean < gate['cleanRate']:
                reasons.append(f"clean rate {clean:.3f} below {gate['cleanRate']}")
            row = dict(gate, checkedAtSteps=state['steps'], passed=not reasons, reasons=reasons,
                       cleanRate=clean, cleanDives=clean_dives, entrySuccess=entry_success)
            results.append(row)
            if reasons and failed is None:
                failed = row
        return failed if self.args.gates else None

    def advance_stage(self, report):
        """Widen the declaration scope once the current stage's clean rate has held."""
        if not self.args.stage_curriculum:
            return
        clean = report['summary']['full']['clean']
        if self.progression.observe(clean, self.state['steps']):
            self.base_environment().set_stage(self.progression.index)
            print(json.dumps(dict(stageAdvanced=self.progression.history[-1], stage=mastery.stage(self.progression.index)['name'])), flush=True)

    def run_evaluation(self):
        """Evaluate on held-out routines and keep the best; True once training should stop."""
        # Evaluation uses a separate RNG and environment; preserve learner randomness.
        self.initialize_references()
        rng = torch.get_rng_state()
        stage = self.progression.index if self.args.stage_curriculum else None
        report = evaluate(self.policy, cases=self.args.eval_cases, reward_mode=self.args.reward_mode, stage=stage)
        torch.set_rng_state(rng)
        state = self.state
        report['steps'] = state['steps']
        if isinstance(self.env, MotorCurriculum):
            report['motorSkills'] = evaluate_motor_skills(self.policy, cases=self.args.eval_cases, direction_practice=self.args.direction_practice)
        if self.args.goal_practice:
            targets = evaluate_targets(self.policy, cases_per_target=max(2, self.args.eval_cases // 6), reward_mode=self.args.reward_mode)
            targets['steps'] = state['steps']
            atomic_json(self.out / 'target-evaluations' / f"{state['steps']}.json", targets)
        atomic_json(self.out / 'evaluations' / f"{state['steps']}.json", report)
        state['evaluations'].append(dict(steps=state['steps'], stage=stage, summary=report['summary']))
        labels = update_selection(state, report)
        # The checkpoint that ships is the one with the most clean dives, once any exist.
        if 'best-clean' in labels and (report['summary']['full']['clean'] or 0) > 0 and 'best' not in labels:
            labels.append('best')
        failed_gate = self.check_gates(report)
        self.advance_stage(report)
        self.final_test()
        saved = self.persist('evaluated')
        for label in labels:
            atomic_checkpoint(self.out / (label + '.pt'), saved)
        atomic_json(self.out / 'selection.json', state['selection'])
        if 'best' in labels:
            atomic_json(self.out / 'best-policy.json', dict(**self.policy.export(), contract=self.contract,
                                                            steps=state['steps'], qualified=False))
        if failed_gate is not None:
            print(json.dumps(dict(gateFailed=failed_gate)), flush=True)
            return 'gate'
        return self.plateaued()

    def plateaued(self):
        """No category improved on any metric over the last ``patience`` evaluations."""
        state, args = self.state, self.args
        evaluations = [row for row in state['evaluations'] if row['steps'] >= state.get('phaseStartSteps', 0)]
        if state['steps'] < args.minimum_steps or len(evaluations) <= args.patience:
            return False
        old = evaluations[:-args.patience]
        new = evaluations[-args.patience:]

        def best(rows, g, key):
            return max((r['summary']['categories'][str(g)].get(key) or 0) for r in rows)

        improving = any(best(new, g, key) > best(old, g, key) + threshold
                        for g in range(1, 7) for key, threshold in PLATEAU_THRESHOLDS)
        return not improving

    # --------------------------------------------------------------- learning
    def collect(self):
        """Run one horizon of every environment under the current policy."""
        args, env, policy = self.args, self.env, self.policy
        T, N = args.horizon, args.envs
        batch = dict(observations=torch.zeros(T, N, env.observation_size), masks=torch.zeros(T, N, len(DIVES), dtype=torch.bool),
                     choosing=torch.zeros(T, N, dtype=torch.bool), raw=torch.zeros(T, N, 9),
                     choices=torch.zeros(T, N, dtype=torch.long), oldlog=torch.zeros(T, N), values=torch.zeros(T, N),
                     rewards=torch.zeros(T, N), dones=torch.zeros(T, N), learn=torch.ones(T, N, dtype=torch.bool), episodes=[])
        for t in range(T):
            obs = torch.tensor(env.observe())
            mask = torch.tensor(env.mask())
            select = torch.tensor(env.choosing)
            with torch.no_grad():
                pred = policy(obs, mask, select)
            batch['observations'][t] = obs
            batch['masks'][t] = mask
            batch['choosing'][t] = select
            batch['raw'][t] = pred['raw']
            batch['choices'][t] = pred['choice']
            batch['oldlog'][t] = pred['logp']
            batch['values'][t] = pred['value']
            _, reward, done, finished = env.step(pred['choice'].numpy(), pred['action'].numpy(), pred['noise'].numpy())
            # Recovery snapshots assist motor learning, not declaration selection.
            batch['learn'][t] = ~(select & torch.tensor(env.practice))
            batch['rewards'][t] = torch.tensor(reward)
            batch['dones'][t] = torch.tensor(done.astype(np.float32))
            batch['episodes'].extend(finished)
        with torch.no_grad():
            batch['bootstrap'] = policy(torch.tensor(env.observe()), torch.tensor(env.mask()), torch.tensor(env.choosing))['value']
        return batch

    def rescale_value_head(self, returns):
        """Change the PopArt scale before constructing gradient-bearing forward passes."""
        rescale = self.policy.update_value_scale(returns.flatten())
        for parameter in self.policy.value.parameters():
            moments = self.optimizer.state.get(parameter, {})
            if 'exp_avg' in moments:
                moments['exp_avg'].mul_(rescale)
                moments['exp_avg_sq'].mul_(rescale ** 2)

    def entropy_weights(self):
        """Refinement is explicit and driven by the held-out clean rate, not training reward."""
        evaluations = self.state['evaluations']
        if evaluations and evaluations[-1]['summary']['full']['clean'] > REFINEMENT_CLEAN_RATE:
            return self.args.entropy * REFINEMENT_FACTOR, self.args.declaration_entropy * REFINEMENT_FACTOR
        return self.args.entropy, self.args.declaration_entropy

    def optimize(self, batch, advantages, returns):
        """Clipped PPO epochs over the flattened batch; returns per-minibatch losses and KLs."""
        args, policy, optimizer = self.args, self.policy, self.optimizer
        self.rescale_value_head(returns)
        flat = lambda x: x.flatten(0, 1)
        adv = flat(advantages)
        active = flat(batch['learn'])
        choosing = flat(batch['choosing'])
        for kind in (False, True):
            subset = active & (choosing == kind)
            if subset.any():
                adv[subset] = (adv[subset] - adv[subset].mean()) / (adv[subset].std(unbiased=False) + 1e-8)
        ret = flat(returns)
        losses, kls = [], []
        motor_entropy, declaration_entropy = self.entropy_weights()
        T, N = args.horizon, args.envs
        minibatches = math.ceil(T * N / args.batch)
        for epoch in range(args.epochs):
            for ids in torch.randperm(T * N).split(args.batch):
                pred = policy(flat(batch['observations'])[ids], flat(batch['masks'])[ids], choosing[ids],
                              flat(batch['raw'])[ids], flat(batch['choices'])[ids])
                selected = active[ids]
                pg, kl = ppo_terms(pred['logp'], flat(batch['oldlog'])[ids], adv[ids], choosing[ids], selected, pred['entropy'],
                                   motor_entropy=motor_entropy, declaration_entropy=declaration_entropy)
                # Value targets from recovery-practice declaration steps mix practice-assisted
                # returns into V(choosing); they are excluded like their policy gradient.
                value_rows = selected if selected.any() else torch.ones_like(selected)
                vf = .5 * (pred['normalizedValue'] - (ret[ids] - policy.value_mean) / policy.value_std).square()[value_rows].mean()
                loss = vf if self.state.get('criticWarmupRemaining', 0) > 0 else pg + vf
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite PPO loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), GRADIENT_NORM)
                optimizer.step()
                with torch.no_grad():
                    policy.logstd.clamp_(-2.8, 0)
                losses.append(float(loss.detach()))
                kls.append(float(kl))
            if np.mean(kls[-minibatches:]) > KL_LIMIT:
                break
        return losses, kls

    def update(self):
        """One rollout, one PPO update and the bookkeeping around them; truthy when training should stop."""
        batch = self.collect()
        advantages, returns = estimate_advantages(batch, self.args.gae_lambda)
        losses, kls = self.optimize(batch, advantages, returns)
        # Input statistics move only after an update, so every observation in one
        # rollout is normalised the same way when it is collected and when it is
        # learned from, and a resumed checkpoint continues identically.
        self.policy.update_input_statistics(batch['observations'])
        state, args = self.state, self.args
        state['criticWarmupRemaining'] = max(0, state.get('criticWarmupRemaining', 0) - 1)
        state['steps'] += self.block
        state['updates'] += 1
        row = dict(steps=state['steps'], **summary(batch['episodes']), loss=float(np.mean(losses)), kl=float(np.mean(kls)),
                   valueScale=float(self.policy.value_std))
        if isinstance(self.env, MotorCurriculum):
            row['motorSkills'] = self.env.metrics()
        state['history'].append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        saved = self.persist('training')
        finished = state['steps'] >= self.target
        if state['updates'] % args.archive_every == 0 or self.stop or finished:
            folder = self.out / 'checkpoints'
            folder.mkdir(exist_ok=True)
            atomic_checkpoint(folder / f"{state['steps']}.pt", saved)
        if not self.stop and args.evaluate_every and (state['updates'] % args.evaluate_every == 0 or finished):
            return self.run_evaluation()
        if isinstance(self.env, MotorCurriculum) or args.stage_curriculum:
            row['stage'] = self.progression.index if args.stage_curriculum else None
        return False

    def train(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.request_stop)
        self.save_sources()
        atomic_json(self.out / 'contract.json', self.contract)
        self.persist('starting')
        try:
            self.initialize_references()
            while self.state['steps'] < self.target and not self.stop:
                outcome = self.update()
                if outcome:
                    self.persist('gate-failed-awaiting-review' if outcome == 'gate' else 'plateau-awaiting-review')
                    return self.state
            # A run killed between its final update and its final evaluation resumes here
            # with the budget complete; produce the missing report instead of skipping it.
            final_report = self.out / 'evaluations' / f"{self.state['steps']}.json"
            if not self.stop and self.args.evaluate_every and self.state['steps'] >= self.target and not final_report.exists():
                self.run_evaluation()
            self.persist('paused' if self.stop else 'budget-complete-awaiting-review')
            return self.state
        except BaseException:
            self.persist('failed')
            raise
        finally:
            self.env.close()


def train(args):
    return Trainer(args).train()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--resume')
    p.add_argument('--random-motor-init', action='store_true',
                   help='Fresh random networks without the neutral-pose motor bias; an exact resume remains supported')
    p.add_argument('--continue-from', help='Explicit new curriculum phase preserving actor, optimizer and cumulative counters')
    p.add_argument('--rotation-progress', action='store_true', help='Explicit remaining turns and potential-based rotation progress')
    p.add_argument('--direction-practice', action='store_true', help='Balance and teach signed takeoff momentum and extend into flight')
    p.add_argument('--goal-practice', action='store_true', help='Train shared motors on assigned legal dive targets alongside autonomous routines')
    p.add_argument('--warm-start')
    p.add_argument('--initialize-from', help='Same-schema actor only; fresh optimizer/value head/worlds for comparisons')
    p.add_argument('--accept-source-change', nargs='*', default=[],
                   help='Files whose hash may differ from the resumed checkpoint; recorded as an amendment')
    p.add_argument('--widths', type=int, nargs='+', default=[256, 256])
    p.add_argument('--rho', type=float, default=.6)
    integers = [('envs', 64), ('threads', 4), ('horizon', 160), ('epochs', 4), ('batch', 1024), ('steps', 2048000),
                ('seed', 109310), ('archive-every', 100), ('evaluate-every', 200), ('eval-cases', 24),
                ('minimum-steps', 102400000), ('patience', 10)]
    for name, default in integers:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--entropy', type=float, default=.006)
    p.add_argument('--declaration-entropy', type=float, default=.01)
    p.add_argument('--reward-mode', choices=['v12', 'continuous-entry', 'phase-dense', 'completion-first', 'conjunctive'], default='v12')
    p.add_argument('--noise-rho', type=float, default=.9, help='AR(1) correlation of the observed exploration noise (0 = white)')
    p.add_argument('--input-normalization', type=int, choices=[0, 1], default=1, help='Running input normalisation, folded into the export')
    p.add_argument('--stage-curriculum', type=int, choices=[0, 1], default=1, help='Mastery-gated declaration scope and stance randomisation')
    p.add_argument('--start-stage', type=int, default=0)
    p.add_argument('--gates', type=int, choices=[0, 1], default=1, help='Stop for review when a milestone gate fails')
    p.add_argument('--motor-curriculum', choices=['off', 'context', 'adaptive'], default='off')
    p.add_argument('--architecture', choices=['shared', 'split'], default='shared')
    p.add_argument('--reference-policy', nargs='*', default=[], help='Immutable baseline checkpoints for paired full-dive qualification')
    p.add_argument('--final-seed', type=int, default=883100)
    p.add_argument('--final-cases', type=int, default=48)
    p.add_argument('--practice', type=float, default=.35)
    p.add_argument('--gae-lambda', type=float, default=.95)
    p.add_argument('--recovery-mode', choices=['recent', 'progress'], default='recent')
    p.add_argument('--exploration', choices=['diagonal', 'state-covariance'], default='diagonal')
    return p


if __name__ == '__main__':
    train(parser().parse_args())
