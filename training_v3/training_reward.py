"""Versioned reward experiments. The competition judge is never modified here."""
import math

from judge import training_deductions

MODES = ('v12', 'continuous-entry', 'phase-dense', 'completion-first', 'conjunctive')
VERSION = 'diving-independent-reward-ledger-v2'
ALIGNMENT_BUDGET = 1.5


def alignment_cost(angle):
    """Same maximum cost as v12, but a slope across the entire physical range.

    Keep the original slope for the first 30 degrees, then use the remaining
    budget through 180 degrees. A better angle always helps a failed dive;
    the penalty never outweighs the separate takeoff costs by growing unbounded.
    """
    if not math.isfinite(angle) or not 0 <= angle <= 180:
        raise ValueError('Expected a finite entry angle between zero and 180 degrees')
    return .025 * min(angle, 30) + .005 * max(0, angle - 30)


def reward_components(score, measurements, mode='v12'):
    if mode not in MODES:
        raise ValueError('Unknown reward comparison mode')
    if mode == 'conjunctive':
        from motor_objective import conjunctive_components
        return conjunctive_components(score, measurements, armstand=score['category'] == 6)
    if mode in ('phase-dense', 'completion-first'):
        from motor_objective import terminal_components
        return terminal_components(score, measurements, armstand=score['category'] == 6,
                                   completion_first=mode == 'completion-first')
    costs = training_deductions(score, measurements)
    if mode == 'continuous-entry':
        costs['entryAlignment'] = alignment_cost(score['entryAngle'])
    credits = dict(execution=score['trainingValue'], completedDive=float(score['valid']))
    return dict(total=float(sum(credits.values()) - sum(costs.values())),
                deductions=costs, credits=credits, deductionTotal=float(sum(costs.values())),
                alignmentCost=costs['entryAlignment'],
                takeoffCost=sum(costs[name] for name in ['takeoffHeight', 'takeoffLean', 'preparationBounces', 'takeoffSpeed']),
                rotationCost=costs['somersaultCount'], twistCost=costs['twistCount'],
                positionCost=costs['position'])
