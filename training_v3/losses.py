"""PPO surrogate for the hybrid declaration/motor policy.

Every decision step is weighted once, whether it was a declaration or a motor
command. Averaging the two groups separately gave the ~1% of declaration steps
half of the policy gradient while their entropy bonus was diluted ~100x, which
drove the categorical head to collapse. The declaration keeps its own entropy
coefficient because a categorical over legal dives and a nine-dimensional
squashed Gaussian have different scales.
"""
import torch


def ppo_terms(logp, old_logp, advantages, choosing, selected, entropy, clip=.2, motor_entropy=.006, declaration_entropy=.01):
    ratio = (logp - old_logp).exp()
    surrogate = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
    if not selected.any():
        zero = ratio.sum() * 0
        return zero, zero.detach()
    count = int(selected.sum())
    policy_loss = surrogate[selected].sum() / count
    bonus = (motor_entropy * entropy[selected & ~choosing].sum() + declaration_entropy * entropy[selected & choosing].sum()) / count
    kl = (((ratio - 1) - (logp - old_logp))[selected].sum() / count).detach()
    return policy_loss - bonus, kl
