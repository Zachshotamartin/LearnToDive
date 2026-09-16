"""Independent entry faults and explicit maximum execution deductions.

These are automated geometric proxies, not official per-fault judging tables.
The total five-point budget replaces entryForm + feet + legs + hands; those
aggregate costs must not also be charged. Each monotone kernel retains a slope
when other faults have already exhausted the displayed execution score.
"""
import math

ENTRY_BUDGETS = {
    'entryHipBend': .5,
    'entryKneeBend': .5,
    'entryShoulderPitch': 1.,
    'entryShoulderRoll': .5,
    'entryElbowBend': .5,
    'entryFootPoint': .5,
    'entryHandAlignment': .25,
    'entryHandSeparation': .25,
    'entryHandHeightDifference': .25,
    'entryKneeSeparation': .125,
    'entryAnkleSeparation': .25,
    'entryToeSeparation': .125,
    'entryCrossedLegs': .25,
}


def entry_deductions(losses):
    if set(losses) != set(ENTRY_BUDGETS):
        raise ValueError('Entry scoring requires every separately measured fault')
    result = {}
    for name, budget in ENTRY_BUDGETS.items():
        loss = float(losses[name])
        if not math.isfinite(loss) or loss < 0:
            raise ValueError(f'Invalid entry fault: {name}')
        result[name] = budget * loss / (1 + loss)
    return result
