"""Continuous motor feedback, independent of the competition score's caps.

Potential shaping uses gamma * Phi(next) - Phi(previous), with zero terminal
potential. It telescopes in discounted returns: holding a pose or delaying
entry cannot repeatedly collect its potential. These are training units,
not additional competition deductions.
"""
import numpy as np

from geometry import entry_geometry, quat_up, tuck_geometry
from positions import position_qualities

GAMMA = .995
VERSION = 'continuous-motor-phases-v14'
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
    arm = np.where(e.headfirst, 3.05, 0)
    pitch = q[:, [6, 9]] - arm[:, None]
    up = quat_up(e.state[:, 5:9])
    cosine = up[:, 2] * np.where(e.headfirst, -1, 1)
    return dict(
        alignment=np.arccos(np.clip(cosine, -1, 1)) / (np.pi / 4),
        hip=np.mean(np.abs(q[:, [0, 3]]), axis=1),
        knee=np.mean(np.abs(q[:, [1, 4]]), axis=1),
        shoulderPitch=np.mean(np.abs(pitch), axis=1),
        shoulderRoll=np.mean(np.abs(q[:, [7, 10]]), axis=1),
        elbow=np.mean(np.abs(q[:, [8, 11]]), axis=1),
        toes=np.mean(geo['footLineAngles'], axis=1),
        hands=np.maximum(geo['handSeparation'] - .1, 0) * 3 * e.headfirst,
        legs=np.maximum(geo['ankleGap'] - .13, 0) * 3,
    )


def potential_components(env):
    e = env.physics
    q = e.state[:, 1 + e.qadr]
    errors = live_errors(e)
    quality = position_qualities(q[:, 0], q[:, 1], np.max(tuck_geometry(e.sensors)[0], axis=1))
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
    weights = dict(alignment=1., hip=.5, knee=.75, shoulderPitch=1.5,
                   shoulderRoll=.5, elbow=.75, toes=.75, hands=.5, legs=.5)
    for name, value in errors.items():
        components['entry' + name[0].upper() + name[1:]] = -weights[name] * value * airborne * preparing
    return {k: v * active for k, v in components.items()}


def terminal_components(score, m, armstand=False):
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
    return dict(total=float(sum(credits.values()) - sum(costs.values())), deductions=costs,
                credits=credits, deductionTotal=float(sum(costs.values())), version=VERSION)


def shaped_reward(previous, current, terminal):
    """The terminal correction also applies to failed and truncated episodes."""
    return GAMMA * np.where(terminal, 0., current) - previous
