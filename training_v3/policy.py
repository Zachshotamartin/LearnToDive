"""Hybrid self-declaration and motor outputs with optional independent trunks.

There are no per-dive specialists. Motor exploration is a conditional
autoregressive Gaussian with an explicit, correct likelihood; no noise is
filtered after sampling.
"""
import math

import torch
from torch import nn
from torch.distributions import Categorical, Normal, MultivariateNormal

from rules import DIVES

FORMAT = 'self-declared-diver-v11'
MOTOR_FORMAT = 'self-declared-motor-curriculum-v14'
ACTIONS = 9
LOG_STD_MIN = -2.8
LOG_STD_MAX = 0.
POPART_RATE = .01


class Policy(nn.Module):
    def __init__(self, obs, widths=(256, 256), rho=.6, initial_action=None, exploration="diagonal", architecture='shared'):
        super().__init__()
        self.widths = tuple(widths)
        self.obs = obs
        self.rho = rho
        if architecture not in ('shared', 'split'):
            raise ValueError('Unknown actor architecture')
        self.architecture = architecture
        if exploration not in ("diagonal", "state-covariance"):
            raise ValueError("Unknown exploration distribution")
        self.exploration = exploration
        layers = []
        size = obs
        for width in widths:
            layers.extend([nn.Linear(size, width), nn.Tanh()])
            size = width
        self.trunk = nn.Sequential(*layers)
        if architecture == 'split':
            import copy
            self.selector_trunk = copy.deepcopy(self.trunk)
            self.critic_trunk = copy.deepcopy(self.trunk)
        self.motor = nn.Linear(size, ACTIONS)
        self.choice = nn.Linear(size, len(DIVES))
        self.value = nn.Linear(size, 1)
        self.logstd = nn.Parameter(torch.full((ACTIONS,), -.5))
        self.register_buffer('value_mean', torch.zeros(()))
        self.register_buffer('value_std', torch.ones(()))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        for layer in [self.motor, self.choice]:
            nn.init.orthogonal_(layer.weight, .01)
        if exploration == "state-covariance":
            # Keep common actor/value initialization and rollout RNG identical
            # across comparison arms; only this extra distribution head differs.
            with torch.random.fork_rng():
                self.noise_factor = nn.Linear(size, ACTIONS * 3)
                nn.init.normal_(self.noise_factor.weight, std=.015)
                nn.init.normal_(self.noise_factor.bias, std=.03)
        if initial_action is not None:
            self.hold(initial_action)

    @torch.no_grad()
    def hold(self, initial_action):
        """Bias the motor mean so the untrained policy holds the reset stance.

        At the autoregressive steady state the mean is 2 tanh(motor / 2) when
        the previous action equals the raw output.
        """
        raw = torch.as_tensor(initial_action, dtype=torch.float32).clamp(-.99, .99).atanh()
        self.motor.bias.copy_(2 * torch.atanh((raw / 2).clamp(-.9, .9)))

    @staticmethod
    def log_jacobian(z):
        return 2 * (math.log(2) - z - torch.nn.functional.softplus(-2 * z))

    def forward(self, obs, mask, choosing, raw=None, choice=None, deterministic=False):
        h = self.trunk(obs)
        selector_h = self.selector_trunk(obs) if self.architecture == 'split' else h
        previous = obs[:, -ACTIONS:].clamp(-.999, .999).atanh()
        mean = (1 - self.rho) * 2 * torch.tanh(self.motor(h) / 2) + self.rho * previous
        dist = self.motor_distribution(h, mean)
        if not mask.any(-1).all():
            raise ValueError('Empty legal-declaration set')
        selection = Categorical(logits=self.choice(selector_h).masked_fill(~mask, -torch.inf))
        if choice is None:
            choice = selection.logits.argmax(-1) if deterministic else selection.sample()
        if raw is None:
            raw = mean if deterministic else dist.sample()
        base_logp = dist.log_prob(raw)
        if self.exploration == "diagonal":
            base_logp = base_logp.sum(-1)
        motor_logp = base_logp - self.log_jacobian(raw).sum(-1)
        logp = torch.where(choosing, selection.log_prob(choice), motor_logp)
        z = dist.rsample()
        base_entropy = dist.entropy()
        if self.exploration == "diagonal":
            base_entropy = base_entropy.sum(-1)
        motor_entropy = base_entropy + self.log_jacobian(z).sum(-1)
        entropy = torch.where(choosing, selection.entropy(), motor_entropy)
        critic_h = self.critic_trunk(obs) if self.architecture == 'split' else h
        normalized = self.value(critic_h).squeeze(-1)
        return dict(action=raw.tanh(), choice=choice, raw=raw, logp=logp, entropy=entropy,
                    value=normalized * self.value_std + self.value_mean, normalizedValue=normalized)

    def motor_distribution(self, features, mean):
        std = self.logstd.clamp(LOG_STD_MIN, LOG_STD_MAX).exp() * math.sqrt(1 - self.rho ** 2)
        if self.exploration == 'diagonal':
            return Normal(mean, std)
        # Conditional full covariance, sampled anew at each decision. The existing
        # autoregression provides temporal smoothness; PPO uses this exact density.
        # This is not gSDE's held latent-noise matrix and does not claim to be.
        factor = .5 * self.noise_factor(features).tanh().reshape(-1, ACTIONS, 3)
        factor = std[None, :, None] * factor
        covariance = torch.diag_embed(std.square().expand(len(mean), -1))
        covariance = covariance + factor @ factor.transpose(-1, -2)
        return MultivariateNormal(mean, covariance_matrix=covariance)

    @torch.no_grad()
    def update_value_scale(self, returns):
        """PopArt affine correction; every unnormalized prediction is preserved exactly."""
        mean = (1 - POPART_RATE) * self.value_mean + POPART_RATE * returns.mean()
        second = ((1 - POPART_RATE) * (self.value_std.square() + self.value_mean.square())
                  + POPART_RATE * returns.square().mean())
        std = (second - mean.square()).clamp(min=.01).sqrt()
        ratio = self.value_std / std
        self.value.weight.mul_(ratio)
        self.value.bias.copy_((self.value_std * self.value.bias + self.value_mean - mean) / std)
        self.value_mean.copy_(mean)
        self.value_std.copy_(std)
        return float(ratio)

    def export(self):
        return dict(format=self.format,
                    architecture=self.architecture, exploration=self.exploration, observationSize=self.obs, actionSize=ACTIONS, choiceSize=len(DIVES),
                    widths=self.widths, rho=self.rho,
                    state={k: v.detach().cpu().tolist() for k, v in self.state_dict().items()})

    @property
    def format(self):
        if self.architecture == 'split' or self.obs == 233:
            return MOTOR_FORMAT
        return FORMAT if self.exploration == 'diagonal' else 'self-declared-diver-state-covariance-v1'
