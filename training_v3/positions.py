"""Flight position quality: how close the body is to straight, pike or tuck.

Quality is 1 at the reference shape and falls off as 1 / (1 + distance), so a
badly bent body still carries a usable slope toward the shape. The Gaussian
kernels used before were numerically zero (and so was their gradient) once
the hips were a radian off, which left the learner unable to feel the
position component of the judge at all.
"""
import numpy as np

STRAIGHT = dict(hip=0., knee=0., hip_scale=.35, knee_scale=.35)
PIKE = dict(hip=1.5, knee=0., hip_scale=.8, knee_scale=.35)
TUCK = dict(hip=1.4, knee=2., hip_scale=.8, knee_scale=.8)
TUCK_HAND_SCALE = .45   # metres from hand to shin at which the tuck quality halves


def kernel(distance):
    """Heavy-tailed quality in (0, 1]; its slope never vanishes."""
    return 1 / (1 + distance)


def shape_quality(hip, knee, target):
    distance = np.sqrt(((hip - target['hip']) / target['hip_scale']) ** 2 + ((knee - target['knee']) / target['knee_scale']) ** 2)
    return kernel(distance)


def position_qualities(hip, knee, hand_distance):
    """Qualities for positions A (straight), B (pike), C (tuck) and D (free = best of the three)."""
    straight = shape_quality(hip, knee, STRAIGHT)
    pike = shape_quality(hip, knee, PIKE)
    tuck = shape_quality(hip, knee, TUCK) * kernel(hand_distance / TUCK_HAND_SCALE)
    return np.stack([straight, pike, tuck, np.maximum.reduce([straight, pike, tuck])], axis=-1)
