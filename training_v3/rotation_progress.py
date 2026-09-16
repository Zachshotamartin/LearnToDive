"""Measured remaining turns and policy-invariant rotation progress shaping.

This does not predict a trajectory, command joints, or change dive judging.
Features are inserted before the base environment's nine action-history values.
"""
import copy
import numpy as np
import torch

VERSION = 'remaining-rotation-potential-v1'
# Layout of the motor-curriculum observation: 216 physical and routine columns,
# then the two remaining-rotation columns, ten task-context columns and the
# eighteen history columns (previous exploration noise, previous action).
INSERT = 216
ADDED = 2
OLD_SIZE = 244
NEW_SIZE = OLD_SIZE + ADDED
INPUT_WEIGHTS = ('trunk.0.weight', 'selector_trunk.0.weight', 'critic_trunk.0.weight')
WEIGHT = 2.


def remaining_turns(signed_goal, twist_goal, rotation, twist):
    """World-signed flip remainder; twist magnitude remainder (either hand legal).

    Negative twist remainder means overshoot, not left-handed twisting. Flip
    remainder changes sign across its target even for reverse/inward dives.
    Measurements are unwrapped turns, exactly as used by the terminal judge.
    """
    return np.stack([np.asarray(signed_goal) - rotation,
                     np.asarray(twist_goal) - np.abs(twist)], axis=-1)


def rotation_potentials(signed_goal, twist_goal, rotation, twist):
    """Two bounded potentials, maximal at exact counts and worse on either side.

    Zero initial rotation has zero potential for every declaration. Both
    terminal and time-limit transitions must cancel the potential. Discounted
    closed loops then earn no surplus, including reversals and waiting.
    """
    error = np.abs(remaining_turns(signed_goal, twist_goal, rotation, twist))
    initial = np.abs(np.stack([signed_goal, twist_goal], axis=-1))
    return WEIGHT * (initial / (1 + initial) - error / (1 + error))


def insert_columns(tensor):
    """Zero new feature weights/moments, retaining every old column exactly."""
    if tensor.ndim != 2 or tensor.shape[1] != OLD_SIZE:
        raise ValueError(f'Rotation migration requires the {OLD_SIZE}-feature motor policy')
    return torch.cat([tensor[:, :INSERT], tensor.new_zeros((tensor.shape[0], ADDED)),
                      tensor[:, INSERT:]], dim=1)


def migrated_states(model, optimizer, named_parameters):
    """Keep all learned weights, PopArt, Adam steps and old first/second moments."""
    model = copy.deepcopy(model)
    optimizer = copy.deepcopy(optimizer)
    for name in INPUT_WEIGHTS:
        model[name] = insert_columns(model[name])
    # Input statistics, when present, gain a neutral (zero mean, unit variance) entry per new column.
    for name, fill in (('input_mean', 0.), ('input_var', 1.)):
        if name in model and model[name].ndim == 1 and model[name].shape[0] == OLD_SIZE:
            vector = model[name]
            model[name] = torch.cat([vector[:INSERT], vector.new_full((ADDED,), fill), vector[INSERT:]])
    ids = [i for group in optimizer['param_groups'] for i in group['params']]
    names = [name for name, _ in named_parameters]
    if len(ids) != len(names):
        raise ValueError('Optimizer parameter inventory changed')
    for pid, name in zip(ids, names):
        if name in INPUT_WEIGHTS:
            for key, value in optimizer['state'].get(pid, {}).items():
                if torch.is_tensor(value) and value.ndim == 2:
                    optimizer['state'][pid][key] = insert_columns(value)
    return model, optimizer
