"""Single-athlete physics probes shared by the measurement and stance tests."""
from engine import Arena as Physics
from geometry import encoded_action


def single_athlete(seed):
    """One forward-facing athlete with every height allowed and no training randomization."""
    return Physics(1, seed=seed, threads=1, skills=[0], heights=(1, 10), minimum_heights=[1] * 6, training=False)


def action_for(targets):
    """The (1, 9) normalized action that commands the given servo targets."""
    return encoded_action(targets)[None]
