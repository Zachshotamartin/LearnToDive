"""Physical starting poses: a flat-footed stand at the edge and a balanced armstand.

Nothing here supports the athlete afterwards. Balance, the press, the jump and
the flight are the controller's job; these functions only place a body on the
board in a pose a diver could actually hold for a moment.
"""
import mujoco
import numpy as np

from geometry import ACTUATOR_MAP

STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
BOARD_TOP = .055          # board box half height; the tip is at x = 0
CONTACT_CLEARANCE = 1e-4  # metres of initial penetration so contact is live
SKILLS = [
    {'id': '101C', 'code': 101, 'name': 'Forward dive · tuck', 'turns': .5, 'twists': 0, 'back': 0, 'shape': 0, 'height': 3},
    {'id': '103C', 'code': 103, 'name': 'Forward 1½ somersaults · tuck', 'turns': 1.5, 'twists': 0, 'back': 0, 'shape': 0, 'height': 5},
    {'id': '103B', 'code': 103, 'name': 'Forward 1½ somersaults · pike', 'turns': 1.5, 'twists': 0, 'back': 0, 'shape': 1, 'height': 7.5},
    {'id': '201C', 'code': 201, 'name': 'Back dive · tuck', 'turns': .5, 'twists': 0, 'back': 1, 'shape': 0, 'height': 3},
    {'id': '203C', 'code': 203, 'name': 'Back 1½ somersaults · tuck', 'turns': 1.5, 'twists': 0, 'back': 1, 'shape': 0, 'height': 5},
    {'id': '5132D', 'code': 5132, 'name': 'Forward 1½ somersaults · 1 twist', 'turns': 1.5, 'twists': 1, 'back': 0, 'shape': 2, 'height': 7.5},
]


def orientation(pitch, yaw):
    """Quaternion (w, x, y, z) for a pitch about world Y after a yaw about Z."""
    return [np.cos(pitch / 2) * np.cos(yaw / 2), np.sin(pitch / 2) * np.sin(yaw / 2),
            np.sin(pitch / 2) * np.cos(yaw / 2), np.cos(pitch / 2) * np.sin(yaw / 2)]


def foot_geoms(model):
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ['foot_L', 'foot_R']]
    return [int(model.body_geomadr[body]) for body in bodies]


def lowest_point(model, data, geoms):
    """Lowest world z over the given box/ellipsoid geoms, using their orientation."""
    return min(data.geom_xpos[g, 2] - np.dot(np.abs(data.geom_xmat[g].reshape(3, 3)[2]), model.geom_size[g])
               for g in geoms)


def leading_edge(model, data, geoms):
    """Largest world x reached by the given box geoms."""
    return max(data.geom_xpos[g, 0] + np.abs(data.geom_xmat[g].reshape(3, 3)[0]) @ model.geom_size[g]
               for g in geoms)


def bisect(evaluate, low, high, iterations=40):
    """Root of a monotone function by bisection; returns the midpoint of the final bracket."""
    value_low = evaluate(low)
    for _ in range(iterations):
        mid = (low + high) / 2
        value = evaluate(mid)
        if np.sign(value) == np.sign(value_low):
            low, value_low = mid, value
        else:
            high = mid
    return (low + high) / 2


def capture(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state, data.sensordata.copy()


def initial_state(model, data, height, skill, lean=.02, preload=-.18, hip=.12, knee=.2, toe_over=.03, platform=False):
    """Flat-footed standing start at the edge.

    The ankle angle is solved so both feet lie flat on the board, the leading
    edge of the feet (toes when facing the water, heels when facing away)
    overhangs the tip by ``toe_over``, and the trunk leans toward the water by
    ``lean``. The old start stood on the toe corner with the centre of mass
    already past the tip, which made every dive a fall.
    """
    yaw = np.pi * SKILLS[skill]['back']
    actuated = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    feet = foot_geoms(model)

    def place(ankle):
        mujoco.mj_resetData(model, data)
        data.mocap_pos[0] = [0, 0, -height]
        data.mocap_quat[0] = [1, 0, 0, 0]
        data.eq_active[0] = platform
        data.qpos[4:8] = orientation(lean, yaw)
        data.qpos[0] = preload
        targets = np.array([hip, knee, ankle, .3, .3, 0, 0, 0, 0.])
        data.qpos[actuated] = targets[ACTUATOR_MAP]
        data.ctrl[:] = targets[ACTUATOR_MAP]
        mujoco.mj_forward(model, data)
        return targets, float(data.geom_xmat[feet[0]].reshape(3, 3)[2, 0])

    ankle = bisect(lambda angle: place(angle)[1], -.6, 1.4)
    targets, slope = place(ankle)
    if abs(slope) > 1e-3:
        raise ValueError('No flat-footed stance for this hip/knee/lean combination')
    data.qpos[3] += preload + BOARD_TOP - lowest_point(model, data, feet) - CONTACT_CLEARANCE
    data.qpos[1] += toe_over - leading_edge(model, data, feet)
    mujoco.mj_forward(model, data)
    state, sensors = capture(model, data)
    return state, sensors, targets


ARMSTAND_TARGETS = np.array([0, 0, 1.2, 3.14, 3.14, 0, 0, 0, .06])
ARMSTAND_HAND_X = -.2
HAND_GEOM_IDS = [6, 9]


def balanced_armstand(model, data, state, back):
    """Rewrite ``state`` in place into a handstand whose centre of mass sits over the hands.

    Hands are shoulder-width on the platform, arms straight, and the trunk pitch
    is solved by bisection. Holding it and pressing off are learned; the
    shoulders only carry a bounded servo torque.
    """
    yaw = np.pi * back
    actuated = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    mujoco.mj_setState(model, data, state, STATE_SPEC)

    def place(pitch):
        data.qpos[4:8] = orientation(pitch, yaw)
        data.qpos[actuated] = ARMSTAND_TARGETS[ACTUATOR_MAP]
        data.ctrl[:] = ARMSTAND_TARGETS[ACTUATOR_MAP]
        data.qpos[0] = 0
        data.qpos[1] = 0
        data.qpos[3] = 0
        mujoco.mj_forward(model, data)
        bottom = min(data.geom_xpos[g, 2] - np.linalg.norm(data.geom_xmat[g].reshape(3, 3)[2] * model.geom_size[g])
                     for g in HAND_GEOM_IDS)
        data.qpos[3] += BOARD_TOP - bottom - CONTACT_CLEARANCE
        data.qpos[1] += ARMSTAND_HAND_X - np.mean(data.geom_xpos[HAND_GEOM_IDS, 0])
        mujoco.mj_forward(model, data)
        return float(data.subtree_com[2][0] - np.mean(data.geom_xpos[HAND_GEOM_IDS, 0]))

    pitch = bisect(place, np.pi - .35, np.pi + .35)
    if abs(place(pitch)) > 2e-3:
        raise ValueError('Armstand start could not be balanced over the hands')
    captured, sensors = capture(model, data)
    state[:] = captured
    return sensors, ARMSTAND_TARGETS.copy()
