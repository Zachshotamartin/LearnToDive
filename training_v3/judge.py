"""Independent geometric judge.

Execution is an automated approximation, not a claim to replace human judging,
and no reward coefficients enter this module.

Rule mapping (World Aquatics Competition Regulations, February 2026, Part
Four): failed dives follow 8.6.5 (double bounce on a springboard, twist off by
90 degrees or more, wrong end first); the 4.5 cap for wrong arm placement
follows 8.6.7; whole-body submersion completes the dive (10.6.7); position
faults are 0.5 to 2 points (10.5.5); an unsafe dive close to the board is
capped at 2 points (10.5.4) and distance from the board is a deduction, never a
failure. A somersault count off by a quarter turn or more is treated as a dive
other than the announced number (10.1.7). Distance from the board and the entry
are judged 'according to opinion'; the geometric proxies below are documented
approximations.
"""
import numpy as np

from rules import DIVES, difficulty
from entry_faults import entry_deductions

VERSION = 'declared-position-entry-arms-v18-clean-form-0.7'
FEET_FIRST_GEOMS = (12, 15)   # one-based sensor geom ids of the two feet
ROTATION_TOLERANCE = .25      # somersaults; a quarter turn short is another dive
TWIST_TOLERANCE = .25
SIDEWAYS_TILT = .85           # |lateral up component| that means tumbling sideways
CLEAN_ANGLE = 15
CLEAN_FORM = .7               # the best scripted dives reach 0.67 to 0.79 on this rigid rig; 0.8 was unreachable
UNSAFE_DISTANCE = .2          # metres from the board that cap the award at 2
POINTS_MULTIPLIER = 3         # three judges' awards times the degree of difficulty
RISE_TARGET = .3              # metres of centre-of-mass rise for a full-credit takeoff (rule 10.4.3, control and height)
FREE_LEAN = 15                # degrees of body lean at departure before the takeoff loses control credit
LEAN_RANGE = 45               # degrees beyond FREE_LEAN at which the lean deduction reaches one point
TAKEOFF_SPEED_TARGET = 2.     # m/s of upward departure speed that earns full training credit
RISE_WEIGHT = 1.              # training-reward weight of the missing rise
SPEED_WEIGHT = .5             # training-reward weight of the missing departure speed


def failures(d, apparatus, m, rotation_error, twist_error):
    """Reasons a dive is failed outright. Failures never become an easier dive."""
    reasons = []
    if m['boardInvalid']:
        reasons.append('invalid takeoff or platform contact')
    if not m['water']:
        reasons.append('no water entry')
    if rotation_error > ROTATION_TOLERANCE:
        reasons.append('declared somersault count not completed')
    if twist_error >= TWIST_TOLERANCE:
        reasons.append('declared twist count not completed')
    if not m['fullEntryComplete']:
        reasons.append('entry did not complete')
    if d['headfirst'] and m['firstGeometry'] in FEET_FIRST_GEOMS:
        reasons.append('feet entered before head or hands')
    if not d['headfirst'] and m['firstGeometry'] not in FEET_FIRST_GEOMS:
        reasons.append('feet-first dive did not enter feet first')
    if apparatus == 'springboard' and m['preparationBounces'] > 0:
        reasons.append('double bounce')
    if m['maxLateral'] > SIDEWAYS_TILT and d['twists'] == 0:
        reasons.append('wrong rotation plane')
    return reasons


def deductions(d, m, angle):
    """Execution deductions in points; the insertion order is the summation order."""
    return {
        'takeoffHeight': min(1.5, 1.5 * max(0, 1 - m['ascent'] / RISE_TARGET)),
        'takeoffLean': min(1, max(0, m['departureLean'] - FREE_LEAN) / LEAN_RANGE),
        'preparationBounces': min(2, m['preparationBounces']),
        'position': 2 * (1 - np.clip(m['positionQuality'], 0, 1)),
        'entryAlignment': min(6, angle / 10),
        **entry_deductions(m['entryFaultLosses']),
        'lateralEntry': min(1, m['surfaceLateralSpeed'] / 3),
        'distance': min(2, max(0, .6 - m['x']) * 4),
    }


def judge(declaration, apparatus, height, m):
    """Score the measurements ``m`` of one dive against its declaration."""
    d = DIVES[int(declaration)]
    measured = m['rotation'] * d['sign']
    rotation_error = abs(measured - d['turns'])
    twist_error = abs(abs(m['twist']) - d['twists'])
    reasons = failures(d, apparatus, m, rotation_error, twist_error)
    angle = max(m['firstContactAngle'], m['entryAngle'])
    faults = deductions(d, m, angle)
    raw_execution = float(10 - sum(faults.values()))
    execution = float(np.clip(raw_execution, 0, 10))
    # Score caps are separately reported rule adjustments, not a second charge
    # in the dense training ledger for faults already measured above.
    adjustments = {}
    recognized = m.get("positionRecognition", {})
    fractions = recognized.get("fractions", [0., 0., 0.])
    dominant = int(np.argmax(fractions))
    # A malformed pike is not automatically a clearly different tuck. Only a
    # sustained recognizable alternative shape triggers the wrong-position cap.
    wrong_position = (d["position"] != "D" and recognized.get("samples", 0) >= 5
                      and fractions[dominant] >= .75 and "ABC"[dominant] != d["position"])
    # Numeric validity and matching the complete declaration are different:
    # a clearly different body position is capped, not a failed dive. It must
    # nevertheless not collect the learner's completion/difficulty bonus.
    matched_position = d['position'] == 'D'
    if not matched_position:
        matched_position = ((recognized.get('samples', 0) >= 5 and fractions['ABC'.index(d['position'])] >= .5)
                            if recognized else m['positionQuality'] >= .5)
    for name, applies, cap in [
        ('positionCap', wrong_position, 2.),
        ('unsafeDistanceCap', m['x'] < UNSAFE_DISTANCE, 2.),
        ('armPositionCap', m.get('entryArmPositionCap', not m['entryArmPositionValid']), 4.5),
        ('failedDive', bool(reasons), 0.),
    ]:
        adjustment = max(0., execution - cap) if applies else 0.
        adjustments[name] = adjustment
        execution -= adjustment
    dd = difficulty(declaration, apparatus, height)
    clean = (not reasons and angle <= CLEAN_ANGLE and m['form'] >= CLEAN_FORM and m['entryGeometryValid']
             and m['entryLimbsValid'] and m['entryArmPositionValid'])
    return dict(declaration=d['id'], category=d['group'], difficulty=dd, execution=execution,
                points=POINTS_MULTIPLIER * dd * execution, trainingValue=dd * execution, valid=not reasons,
                rawExecution=raw_execution, scoreAdjustments=adjustments,
                clean=bool(clean and matched_position and not wrong_position), wrongBodyPosition=bool(wrong_position),
                declaredPositionMatched=bool(matched_position), completedDeclaration=bool(not reasons and matched_position),
                deductions={k: float(v) for k, v in faults.items()}, failures=reasons,
                rotationError=rotation_error, twistError=twist_error, entryAngle=angle,
                recognition=dict(somersaults=round(measured * 2) / 2, twists=round(abs(m['twist']) * 2) / 2),
                automatedJudge=True, splashIsProxy=True, judgeVersion=VERSION)


def training_deductions(score, m):
    """One additive training cost per named fault; no aggregate form charge.

    Extra takeoff emphasis is expressed as a weight on the same height fault,
    rather than charging height twice under two names. Rule caps remain display
    adjustments. The ledger stays active for invalid and zero-score dives.
    """
    costs = {name: .25 * value for name, value in score['deductions'].items()}
    costs['takeoffHeight'] *= 1 + RISE_WEIGHT / (.25 * 1.5)
    costs.update(
        takeoffSpeed=SPEED_WEIGHT * min(1.5, max(0, 1 - m['takeoffVerticalSpeed'] / TAKEOFF_SPEED_TARGET)),
        somersaultCount=.8 * float(np.log1p(score['rotationError'])),
        twistCount=.8 * float(np.log1p(score['twistError'])),
        incompleteEntry=1.5 * float(not m['fullEntryComplete']),
        boardContact=2 * float(m['boardInvalid']),
        missedWater=1.5 * float(not m['water']),
        wrongEntryEnd=1.5 * float(any(reason in score['failures'] for reason in
                                    ['feet entered before head or hands', 'feet-first dive did not enter feet first'])),
        wrongRotationPlane=1.5 * float('wrong rotation plane' in score['failures']),
    )
    return costs


def terminal_reward(score, m):
    """Negative fault costs remain visible even when execution is clipped to zero."""
    return float(score['trainingValue'] + float(score['valid']) - sum(training_deductions(score, m).values()))
