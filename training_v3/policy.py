"""Shared actor trunk with hybrid self-declaration and motor outputs.

There are no per-dive specialists. Motor exploration is a conditional
autoregressive Gaussian with an explicit, correct likelihood; no noise is
filtered after sampling.
"""
import math

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from rules import DIVES

FORMAT = 'self-declared-diver-v11'
ACTIONS = 9
LOG_STD_MIN = -2.8
LOG_STD_MAX = 0.
POPART_RATE = .01


class Policy(nn.Module):
    def __init__(self, obs, widths=(256, 256), rho=.6, initial_action=None):
        super().__init__()
        self.widths = tuple(widths)
        self.obs = obs
        self.rho = rho
        layers = []
        size = obs
        for width in widths:
            layers.extend([nn.Linear(size, width), nn.Tanh()])
            size = width
        self.trunk = nn.Sequential(*layers)
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
        previous = obs[:, -ACTIONS:].clamp(-.999, .999).atanh()
        mean = (1 - self.rho) * 2 * torch.tanh(self.motor(h) / 2) + self.rho * previous
        dist = Normal(mean, self.logstd.clamp(LOG_STD_MIN, LOG_STD_MAX).exp() * math.sqrt(1 - self.rho ** 2))
        if not mask.any(-1).all():
            raise ValueError('Empty legal-declaration set')
        selection = Categorical(logits=self.choice(h).masked_fill(~mask, -torch.inf))
        if choice is None:
            choice = selection.logits.argmax(-1) if deterministic else selection.sample()
        if raw is None:
            raw = mean if deterministic else dist.sample()
        motor_logp = (dist.log_prob(raw) - self.log_jacobian(raw)).sum(-1)
        logp = torch.where(choosing, selection.log_prob(choice), motor_logp)
        z = dist.rsample()
        motor_entropy = (dist.entropy() + self.log_jacobian(z)).sum(-1)
        entropy = torch.where(choosing, selection.entropy(), motor_entropy)
        normalized = self.value(h).squeeze(-1)
        return dict(action=raw.tanh(), choice=choice, raw=raw, logp=logp, entropy=entropy,
                    value=normalized * self.value_std + self.value_mean, normalizedValue=normalized)

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
        return dict(format=FORMAT, observationSize=self.obs, actionSize=ACTIONS, choiceSize=len(DIVES),
                    widths=self.widths, rho=self.rho,
                    state={k: v.detach().cpu().tolist() for k, v in self.state_dict().items()})
