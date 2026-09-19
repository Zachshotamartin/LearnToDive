"""Continuous motor feedback, independent of the competition score's caps.

Potential shaping uses gamma * Phi(next) - Phi(previous), with zero terminal
potential. It telescopes in discounted returns: holding a pose or delaying
entry cannot repeatedly collect its potential. These are training units,
not additional competition deductions.
"""
import numpy as np

from geometry import entry_geometry, quat_up, tuck_geometry, entry_arm_errors
from positions import position_qualities

GAMMA = .995
VERSION = 'continuous-motor-phases-v14'
DIRECTION_VERSION = 'signed-takeoff-practice-v1'
COMPLETION_VERSION = 'continuous-count-completion-v3'
CONJUNCTIVE_VERSION = 'conjunctive-rotation-times-entry-v1'
ENTRY_QUALITY_ANGLE = 20.     # degrees of entry error at which the entry quality factor halves
SAFETY_COST = 16.             # a physical failure costs the whole maximum credit
MINIMUM_ILLEGAL_TAKEOFF = .25   # an invalid takeoff this view cannot grade still costs a quarter of the safety charge
LEGALITY_NEAR_WEIGHT = .5       # share of the severity that rises linearly between the limit and twice the limit
LEGALITY_SPAN = 5.              # orders of magnitude over the limit at which a takeoff is as illegal as it gets
END_COST = 4.                 # wrong end first or the wrong rotation plane
COUNT_COST = 8.               # log1p weight of the somersault and twist count errors
VALID_BONUS = 2.              # a completed declaration within the judge's tolerances
TAKEOFF_RISE_WEIGHT = 2.      # training cost of the missing centre-of-mass rise (the judge charges 1.5 more)
TAKEOFF_SPEED_WEIGHT = 1.     # training cost of the missing upward departure speed
TAKEOFF_FLOOR = .4            # credit multiplier of a dive with no rise at all; 1 at the full rise target
COMPLETION_BONUS = 16.
EXECUTION_BUDGET = 4.
ENTRY_WEIGHTS = dict(entryHipBend=.5, entryKneeBend=.75, entryShoulderPitch=1.5,
                    entryShoulderRoll=.5, entryElbowBend=.75, entryFootPoint=.75,
                    entryHandAlignment=.5, entryHandSeparation=.5,
                    entryHandHeightDifference=.25, entryKneeSeparation=.25,
                    entryAnkleSeparation=.25, entryToeSeparation=.25, entryCrossedLegs=.5)


def entry_weight(height, vertical_speed):
    """Smooth preparation window: 0.8 s before ballistic surface arrival.

    Used by the assessor, not to command the actor or set a maneuver's timing.
    Actual collisions and water contact still come from MuJoCo.
    """
    height = np.maximum(height, 0)
    arrival = (vertical_speed + np.sqrt(vertical_speed ** 2 + 2 * 9.81 * height)) / 9.81
    x = np.clip((.9 - arrival) / .6, 0, 1)
    return x * x * (3 - 2 * x)


def live_errors(e):
    q = e.state[:, 1 + e.qadr]
    geo = entry_geometry(e.sensors)
    pitch, roll, elbow = entry_arm_errors(q, e.headfirst)
    up = quat_up(e.state[:, 5:9])
    cosine = up[:, 2] * np.where(e.headfirst, -1, 1)
    return dict(
        alignment=np.arccos(np.clip(cosine, -1, 1)) / (np.pi / 4),
        hip=np.mean(np.abs(q[:, [0, 3]]), axis=1),
        knee=np.mean(np.abs(q[:, [1, 4]]), axis=1),
        shoulderPitch=np.mean(np.abs(pitch), axis=1),
        shoulderRoll=np.mean(np.abs(roll), axis=1),
        elbow=np.mean(np.abs(elbow), axis=1),
        toes=np.mean(geo['footLineAngles'], axis=1),
        hands=np.maximum(geo['handSeparation'] - .1, 0) * 3 * e.headfirst,
        legs=np.maximum(geo['ankleGap'] - .13, 0) * 3,
    )


def rotation_direction_cost(momentum_y, signed_turns, released=True):
    """Bounded momentum-direction error; no prescribed angular speed or sequence.

    World Y momentum and the judge's unwrapped swing share a sign. Beyond a
    modest momentum floor there is no extra credit. Competition still judges
    the actual rotation count and entry; rocking before release is not success.
    """
    direction = np.sign(signed_turns)
    error = np.clip(1 - direction * np.asarray(momentum_y) / 12., 0, 3)
    return np.where(direction != 0, np.where(released, error, np.maximum(1., error)), 0.)


def potential_components(env):
    e = env.physics
    q = e.state[:, 1 + e.qadr]
    errors = live_errors(e)
    quality = position_qualities(q[:, 0], q[:, 1], np.max(tuck_geometry(e.sensors)[0], axis=1), q[:, 3], q[:, 4])
    from rules import DIVES, POSITIONS
    positions = np.array([POSITIONS.index(DIVES[d]['position']) for d in env.declaration])
    pose = quality[np.arange(e.n), positions]
    preparing = entry_weight(e.above_water, e.sensors[:, 5])
    airborne = e.released & ~np.isfinite(e.entry_time)
    active = ~env.choosing
    takeoff = np.clip(e.sensors[:, 5] / 2.8, -1, 1)
    components = dict(
        takeoff=1.5 * takeoff * ~e.released * ~e.armstand,
        flightPosition=-2 * (1 - pose) * airborne * (1 - preparing),
    )
    # Contacts can generate momentum before departure. Terminal cancellation
    # prevents repeatedly collecting points by rocking or delaying takeoff.
    components['takeoffDirection'] = (
        -.75 * rotation_direction_cost(e.sensors[:, 7], e.goals[:, 0], e.released)
        * ~e.released * bool(getattr(env, 'direction_practice', False)))
    weights = dict(alignment=1., hip=.5, knee=.75, shoulderPitch=1.5,
                   shoulderRoll=.5, elbow=.75, toes=.75, hands=.5, legs=.5)
    for name, value in errors.items():
        components['entry' + name[0].upper() + name[1:]] = -weights[name] * value * airborne * preparing
    if getattr(env, 'rotation_progress', False):
        from rotation_progress import rotation_potentials
        progress = rotation_potentials(e.goals[:, 0], e.goals[:, 1],
                                       e.phase_theta / (2 * np.pi), e.air_twist / (2 * np.pi))
        components['somersaultProgress'] = progress[:, 0]
        components['twistProgress'] = progress[:, 1]
    return {k: v * active for k, v in components.items()}


def terminal_components(score, m, armstand=False, completion_first=False):
    """Continuous named errors remain effective even for an invalid dive."""
    faults = m['entryFaultLosses']
    if set(faults) != set(ENTRY_WEIGHTS) or any(not np.isfinite(v) or v < 0 for v in faults.values()):
        raise ValueError('All finite independent entry measurements are required')
    costs = {name: float(weight * np.log1p(faults[name])) for name, weight in ENTRY_WEIGHTS.items()}
    costs.update(
        takeoffHeight=0. if armstand else float(max(0, 1 - m['ascent'] / .45)),
        takeoffSpeed=0. if armstand else float(.5 * max(0, 1 - m['takeoffVerticalSpeed'] / 2.8)),
        takeoffLean=float(max(0, m['departureLean'] - 15) / 90),
        preparationBounces=float(m['preparationBounces']),
        position=2 * (1 - float(np.clip(m.get('motorPositionQuality', m['positionQuality']), 0, 1))),
        entryAlignment=float(score['entryAngle'] / 45),
        somersaultCount=2 * float(score['rotationError']),
        twistCount=2 * float(score['twistError']),
        incompleteEntry=2 * float(not m['fullEntryComplete']),
        boardContact=2 * float(m['boardInvalid']),
        missedWater=2 * float(not m['water']),
        wrongEntryEnd=float(any(reason in score['failures'] for reason in
                               ('feet entered before head or hands', 'feet-first dive did not enter feet first'))),
        wrongRotationPlane=float('wrong rotation plane' in score['failures']),
        lateralEntry=float(m['surfaceLateralSpeed'] / 3),
        distance=float(max(0, .6 - m['x']) * 2),
    )
    # Difficulty earns credit only for an actually completed legal declaration.
    # Motor learning never uses positionCap/armPositionCap/clipped execution.
    credits = dict(completedDive=float(score['valid']),
                   difficulty=.5 * score['difficulty'] * float(score['valid']))
    if completion_first:
        return completion_components(score, costs)
    return dict(total=float(sum(credits.values()) - sum(costs.values())), deductions=costs,
                credits=credits, deductionTotal=float(sum(costs.values())), version=VERSION)


def completion_components(score, raw_costs):
    """Continuous progress toward measured counts, independent of pose recognition.

    No completion-threshold jump or difficulty bonus changes the motor objective.
    Each requested axis earns progress from its own zero-rotation baseline and
    loses credit on overshoot; a stationary attempt earns zero. Unrequested
    rotations still incur count error. Official DD/validity are report-only.
    Physical failure deductions exceed the maximum possible progress credit.
    """
    valid = bool(score.get('completedDeclaration', score['valid']))
    safety = dict(incompleteEntry=32., boardContact=32., missedWater=32.,
                  wrongEntryEnd=4., wrongRotationPlane=4., preparationBounces=2.)
    primary = {'somersaultCount', 'twistCount', *safety}
    form = {k: v for k, v in raw_costs.items() if k not in primary}
    importance = {k: {'entryAlignment': 2., 'position': 2., 'takeoffHeight': 1.5}.get(k, 1.) for k in form}
    total_weight = sum(importance.values())
    budgets = {k: EXECUTION_BUDGET * weight / total_weight for k, weight in importance.items()}
    # Individual saturation preserves an incentive for every fault, including
    # already failed dives, without a large fault changing another's deduction.
    costs = {k: budgets[k] * value / (1 + value) for k, value in form.items()}
    # Original safety charges use 2 for binary faults and 1 for end/plane.
    costs.update({k: weight * min(1., raw_costs[k] / (2. if k in
                  ('incompleteEntry', 'boardContact', 'missedWater') else 1.)) for k, weight in safety.items()})
    costs['somersaultCount'] = float(8. * np.log1p(score['rotationError']))
    costs['twistCount'] = float(8. * np.log1p(score['twistError']))
    from rules import DIVES, IDS
    intent = DIVES[IDS[score['declaration']]]
    goals = np.array([intent['turns'], intent['twists']])
    errors = np.array([score['rotationError'], score['twistError']])
    requested = goals > 0
    baseline = goals[requested] / (.5 + goals[requested])
    progress = (baseline - errors[requested] / (.5 + errors[requested])) / baseline
    credits = dict(rotationProgress=COMPLETION_BONUS * float(progress.mean()) if requested.any() else 0.)
    execution_cost = sum(costs[k] for k in form)
    return dict(total=float(sum(credits.values()) - sum(costs.values())), deductions=costs,
                credits=credits, deductionTotal=float(sum(costs.values())),
                executionDeductionTotal=float(execution_cost), executionBudget=EXECUTION_BUDGET,
                executionBudgets=budgets, rawDeductions=raw_costs,
                stage='execution-refinement' if valid else 'complete-declared-dive',
                version=COMPLETION_VERSION)


def rotation_progress(score):
    """Progress from zero rotation toward the declared counts, 1 at exact counts.

    Overshoot and the wrong direction lower it; declarations without a
    requested axis contribute nothing.
    """
    from rules import DIVES, IDS
    intent = DIVES[IDS[score['declaration']]]
    goals = np.array([intent['turns'], intent['twists']])
    errors = np.array([score['rotationError'], score['twistError']])
    requested = goals > 0
    if not requested.any():
        return 0.
    baseline = goals[requested] / (.5 + goals[requested])
    progress = (baseline - errors[requested] / (.5 + errors[requested])) / baseline
    return float(np.clip(progress.mean(), -1., 1.))


def entry_quality(angle):
    """1 at a vertical entry, 1/2 at ENTRY_QUALITY_ANGLE, never zero; it multiplies the credit."""
    return 1. / (1. + float(angle) / ENTRY_QUALITY_ANGLE)


def takeoff_quality(rise, armstand=False):
    """TAKEOFF_FLOOR for a dive that only falls off the edge, 1 at the full rise target."""
    from judge import RISE_TARGET
    if armstand:
        return 1.
    return TAKEOFF_FLOOR + (1 - TAKEOFF_FLOOR) * float(np.clip(rise / RISE_TARGET, 0., 1.))


def takeoff_legality(m):
    """How illegal a takeoff is, from 0 (comfortably legal) to 1 (clearly illegal).

    The judge's rule stays binary, as the sport's does. Training charges this
    graded view instead, because a binary sixteen-point fault gives a learner
    that has never had a legal takeoff nothing to improve: every attempt costs
    the same whether it was close or wild. Each term is the amount by which one
    measured cause exceeds its own limit.
    """
    from engine import ASSIST_FORCE, ASSIST_IMPULSE, FOOT_GAP_LIMIT, RECONTACT_TILT_LIMIT
    ratio = max(m.get('boardAssistImpulse', 0.) / ASSIST_IMPULSE, m.get('boardAssistForce', 0.) / ASSIST_FORCE,
                m.get('footDepartureGap', 0.) / FOOT_GAP_LIMIT, m.get('recontactTilt', 0.) / RECONTACT_TILT_LIMIT,
                m.get('departureLean', 0.) / 90.)
    severity = graded_severity(ratio)
    if m.get('boardInvalid'):
        # A cause this view cannot grade still costs something definite.
        severity = max(severity, MINIMUM_ILLEGAL_TAKEOFF)
    return float(np.clip(severity, 0., 1.))


def graded_severity(ratio):
    """Severity from the worst cause's ratio to its limit: 0 at the limit, 1 at LEGALITY_SPAN decades over it.

    A linear term keeps the slope where the takeoff is nearly legal. The log
    term keeps a slope where the measured causes are thousands of times over
    their limits: the v13.6 diver's forward takeoffs all crashed onto the
    platform edge at 10,000 times the assist limit, and a clip of the linear
    excess graded every one of them exactly 1.0, so nothing paid for a softer
    crash and the forward takeoff never changed in 100M steps.
    """
    ratio = np.maximum(np.asarray(ratio, dtype=float), 1.)
    near = np.clip(ratio - 1., 0., 1.)
    far = np.clip(np.log10(ratio) / LEGALITY_SPAN, 0., 1.)
    return LEGALITY_NEAR_WEIGHT * near + (1. - LEGALITY_NEAR_WEIGHT) * far


def live_takeoff_legality(e, lean_degrees):
    """The same graded view during an episode, from the engine's live arrays."""
    from engine import ASSIST_FORCE, ASSIST_IMPULSE, FOOT_GAP_LIMIT, RECONTACT_TILT_LIMIT
    ratio = np.maximum.reduce([np.maximum(e.board_impulse, e.stand_impulse) / ASSIST_IMPULSE,
                               np.maximum(e.board_peak, e.stand_peak) / ASSIST_FORCE,
                               e.foot_departure_gap / FOOT_GAP_LIMIT,
                               e.recontact_tilt / RECONTACT_TILT_LIMIT,
                               np.asarray(lean_degrees, dtype=float) / 90.])
    severity = graded_severity(ratio)
    return np.clip(np.where(e.board_invalid, np.maximum(severity, MINIMUM_ILLEGAL_TAKEOFF), severity), 0., 1.)


def conjunctive_components(score, m, armstand=False):
    """Credit that only pays when rotation AND entry are right; costs at the judge's weights.

    Three additive rewards each let the learner max the cheapest phase: entry
    alone produced a faller, takeoff terms a flat-landing jumper, rotation
    credit a flat-landing spinner. Here the rotation credit is multiplied by
    the entry quality and by the takeoff quality, so the maximum needs all
    three phases, while every judge deduction keeps its competition weight and
    the takeoff outcomes (rise, departure speed, lean) are charged separately.
    """
    from judge import RISE_TARGET, TAKEOFF_SPEED_TARGET
    costs = dict(score['deductions'])
    # Alignment keeps its slope across the whole range in training; the judge caps at six.
    costs['entryAlignment'] = float(score['entryAngle']) / 10.
    costs['takeoffRise'] = 0. if armstand else TAKEOFF_RISE_WEIGHT * max(0., 1 - m['ascent'] / RISE_TARGET)
    costs['takeoffSpeed'] = 0. if armstand else TAKEOFF_SPEED_WEIGHT * min(1.5, max(0., 1 - m['takeoffVerticalSpeed'] / TAKEOFF_SPEED_TARGET))
    costs['somersaultCount'] = COUNT_COST * float(np.log1p(score['rotationError']))
    costs['twistCount'] = COUNT_COST * float(np.log1p(score['twistError']))
    costs['incompleteEntry'] = SAFETY_COST * float(not m['fullEntryComplete'])
    costs['boardContact'] = SAFETY_COST * takeoff_legality(m)
    costs['missedWater'] = SAFETY_COST * float(not m['water'])
    costs['wrongEntryEnd'] = END_COST * float(any(reason in score['failures'] for reason in
                                                   ('feet entered before head or hands', 'feet-first dive did not enter feet first')))
    costs['wrongRotationPlane'] = END_COST * float('wrong rotation plane' in score['failures'])
    costs = {k: float(v) for k, v in costs.items()}
    progress = rotation_progress(score)
    quality = entry_quality(score['entryAngle'])
    takeoff = takeoff_quality(m['ascent'], armstand)
    credits = dict(rotationThroughEntry=COMPLETION_BONUS * progress * quality * takeoff,
                   completedDive=VALID_BONUS * float(score['valid']))
    return dict(total=float(sum(credits.values()) - sum(costs.values())), deductions=costs, credits=credits,
                deductionTotal=float(sum(costs.values())), rotationProgress=progress, entryQuality=quality,
                takeoffQuality=takeoff, version=CONJUNCTIVE_VERSION)


def shaped_reward(previous, current, terminal):
    """The terminal correction also applies to failed and truncated episodes."""
    return GAMMA * np.where(terminal, 0., current) - previous
