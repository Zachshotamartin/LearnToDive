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

VERSION = 'self-declared-whole-entry-v11'
FEET_FIRST_GEOMS = (12, 15)   # one-based sensor geom ids of the two feet
ROTATION_TOLERANCE = .25      # somersaults; a quarter turn short is another dive
TWIST_TOLERANCE = .25
SIDEWAYS_TILT = .85           # |lateral up component| that means tumbling sideways
CLEAN_ANGLE = 15
CLEAN_FORM = .8
UNSAFE_DISTANCE = .2          # metres from the board that cap the award at 2
POINTS_MULTIPLIER = 3         # three judges' awards times the degree of difficulty


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
    g = m['entryGeometryWorst']
    hands = (min(1, max(0, g['handSeparation'] - .08) * 5 + max(0, g['handHeightGap'] - .02) * 10)
             if d['headfirst'] else 0)
    return {
        'takeoff': min(1.5, 1.5 * max(0, 1 - m['ascent'] / .3)) + min(2, m['preparationBounces']),
        'position': 2 * (1 - np.clip(m['positionQuality'], 0, 1)),
        'entryAlignment': min(6, angle / 10),
        'entryForm': min(2, 2 * (1 - np.clip(m['form'], 0, 1))),
        'feet': min(1, max(g['footLineAngles']) / 45),
        'legs': min(1, max(0, g['ankleGap'] - .13) * 5 + float(g['crossedLegs'])),
        'hands': hands,
        # Sideways (world y) speed of the parts crossing the surface, m/s.
        'lateralEntry': min(1, m['surfaceLateralSpeed'] / 3),
        # Too close to the board (rules 10.4.6 and 10.5.3, 'according to opinion').
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
    execution = float(np.clip(10 - sum(faults.values()), 0, 10))
    if m['positionQuality'] < .5:
        execution = min(execution, 2.)
    if m['x'] < UNSAFE_DISTANCE:
        execution = min(execution, 2.)
    if not m.get('entryArmPositionValid', True):
        execution = min(execution, 4.5)
    if reasons:
        execution = 0.
    dd = difficulty(declaration, apparatus, height)
    clean = (not reasons and angle <= CLEAN_ANGLE and m['form'] >= CLEAN_FORM and m['entryGeometryValid']
             and m['entryLimbsValid'])
    return dict(declaration=d['id'], category=d['group'], difficulty=dd, execution=execution,
                points=POINTS_MULTIPLIER * dd * execution, trainingValue=dd * execution, valid=not reasons,
                clean=bool(clean), deductions={k: float(v) for k, v in faults.items()}, failures=reasons,
                rotationError=rotation_error, twistError=twist_error, entryAngle=angle,
                recognition=dict(somersaults=round(measured * 2) / 2, twists=round(abs(m['twist']) * 2) / 2),
                automatedJudge=True, splashIsProxy=True, judgeVersion=VERSION)


def terminal_reward(score, m):
    """Training signal, not the competition score.

    Every judged component enters as a continuous cost, so a dive whose
    official execution is already clipped to zero still receives gradient
    toward each fault; a completed declaration earns a bounded bonus. The
    reward is always <= 0 on a failed dive and extra spins never buy points.
    """
    faults = score['deductions']
    costs = (.25 * sum(faults.values()) + .03 * score['entryAngle'] + .8 * np.log1p(score['rotationError'])
             + .8 * np.log1p(score['twistError']))
    costs += 1.5 * float(not m['fullEntryComplete']) + 2 * float(m['boardInvalid']) + 1.5 * float(not m['water'])
    return float(score['trainingValue'] + float(score['valid']) - costs)
