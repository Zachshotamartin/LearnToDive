"""Sensor layout, quaternion helpers and body-geometry measurements.

Everything here is a pure function of a MuJoCo sensor row (or a batch of rows)
from ``diver.xml``. The sensor order is fixed by the XML: centre of mass and its
velocity, angular momentum, pelvis orientation and angular velocity, site
positions, foot touch sensors, one position/orientation pair per athlete geom,
board and stand contact forces per geom, and one linear/angular velocity pair
per geom.
"""
import numpy as np

# Neural actions: common hip, knee and ankle, left/right shoulder pitch,
# left/right shoulder roll, common elbow and common hip adduction. Left and
# right legs stay synchronized; nothing is animated.
ACTION_LOW = np.array([-.45, 0, -.55, -.5, -.5, -1.1, -1.1, 0, -.15])
ACTION_HIGH = np.array([2.2, 2.6, 1.35, 3.14, 3.14, 1.1, 1.1, 2.3, .18])
# Actuator order in diver.xml mapped onto the nine neural actions.
ACTUATOR_MAP = np.array([0, 1, 2, 0, 1, 2, 3, 5, 7, 4, 6, 7, 8, 8])

GEOM_COUNT = 15
COM = slice(0, 3)
COM_VELOCITY = slice(3, 6)
ANGULAR_MOMENTUM = slice(6, 9)
ORIENTATION = slice(9, 13)
ANGULAR_VELOCITY = slice(13, 16)
TOES = slice(16, 22)
HANDS = slice(22, 28)
KNEES = slice(37, 43)
GEOM_POSES = slice(45, 150)          # 15 x (position 3, quaternion 4)
BOARD_CONTACTS = slice(150, 210)     # 15 x (found, force 3)
STAND_CONTACTS = slice(210, 270)
GEOM_VELOCITIES = slice(270, 360)    # 15 x (linear 3, angular 3)
SHIN_LEFT = slice(115, 118)
SHIN_RIGHT = slice(136, 139)
HAND_LEFT_QUAT = slice(83, 87)
HAND_RIGHT_QUAT = slice(104, 108)
# Zero-based geom indices in sensor order.
TRUNK_GEOMS = [0, 1, 2]
SHOULDER_GEOMS = [3, 6]
FOREARM_GEOMS = [4, 7]
HAND_GEOMS = [5, 8]
THIGH_GEOMS = [9, 12]
SHIN_GEOMS = [10, 13]
FOOT_GEOMS = [11, 14]
LEG_GEOMS = [9, 10, 11, 12, 13, 14]
FOOT_CONTACT_FOUND = [194, 206]      # board contact 'found' flags of the feet
HAND_CONTACT_FOUND = [170, 182]      # board contact 'found' flags of the hands


GRAVITY = 9.81


def time_to_surface(height_above_water, vertical_speed, limit=3.):
    """Ballistic seconds until the body reaches the surface; what a diver sees coming.

    Purely kinematic from the current height and vertical speed; it predicts
    nothing about the body's own motion and commands nothing.
    """
    height = np.maximum(np.asarray(height_above_water, dtype=float), 0.)
    speed = np.asarray(vertical_speed, dtype=float)
    arrival = (speed + np.sqrt(speed ** 2 + 2 * GRAVITY * height)) / GRAVITY
    return np.clip(arrival, 0., limit)


def quat_up(q):
    """World up-axis of a body with quaternion rows ``q`` (w, x, y, z)."""
    w, x, y, z = q.T
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)


def quat_forward(q):
    """World forward-axis of a body with quaternion rows ``q``."""
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)


def framed_angles(q):
    """Forward-plane swing, axial twist and lateral tilt of pelvis quaternions.

    The decomposition is invalid near the lateral-axis singularity; sideways
    tumbling is rejected separately by the judge and cannot earn a code.
    """
    up = quat_up(q)
    forward = quat_forward(q)
    side = np.tile([0., 1., 0.], (len(q), 1)) - up * up[:, 1, None]
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-8)
    reference = np.cross(side, up)
    swing = np.arctan2(up[:, 0], up[:, 2])
    twist = np.arctan2(np.sum(forward * side, axis=1), np.sum(forward * reference, axis=1))
    return swing, twist, np.abs(up[:, 1])


def wrap(angle):
    """Wrap an angle into (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def decoded_action(action):
    """Map a normalized action in [-1, 1] onto joint servo targets in radians."""
    return ACTION_LOW + (np.clip(action, -1, 1) + 1) * .5 * (ACTION_HIGH - ACTION_LOW)


def encoded_action(targets):
    """Inverse of :func:`decoded_action` for targets inside the action range."""
    return 2 * (np.asarray(targets) - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW) - 1


def contact_forces(sensors, block):
    """Per-geom contact force magnitudes from a 15 x (found, fx, fy, fz) block."""
    forces = sensors[..., block].reshape(*sensors.shape[:-1], GEOM_COUNT, 4)
    return np.linalg.norm(forces[..., 1:], axis=-1)


def board_forces(sensors):
    return contact_forces(sensors, BOARD_CONTACTS)


def stand_forces(sensors):
    return contact_forces(sensors, STAND_CONTACTS)


def _pairs(sensors, block):
    return sensors[..., block].reshape(*sensors.shape[:-1], 2, 3)


def _ankles(sensors):
    """Ankle positions: the far end of each shin capsule, mirrored about its centre."""
    knees = _pairs(sensors, KNEES)
    shins = np.stack([sensors[..., SHIN_LEFT], sensors[..., SHIN_RIGHT]], axis=-2)
    return knees, 2 * shins - knees


def tuck_geometry(sensors):
    """Hand-to-shin distance and knee/ankle separation, for the tuck quality."""
    hands = _pairs(sensors, HANDS)
    knees, ankles = _ankles(sensors)
    axis = ankles - knees
    fraction = np.clip(np.sum((hands - knees) * axis, axis=-1) / np.maximum(np.sum(axis * axis, axis=-1), 1e-9), 0, 1)
    distance = np.linalg.norm(hands - knees - fraction[..., None] * axis, axis=-1)
    separation = np.maximum(np.linalg.norm(knees[..., 0, :] - knees[..., 1, :], axis=-1),
                            np.linalg.norm(ankles[..., 0, :] - ankles[..., 1, :], axis=-1))
    return distance, separation


def leg_geometry(sensors):
    """Knee, ankle and toe gaps plus whether the legs cross the pelvis side axis."""
    knees, ankles = _ankles(sensors)
    toes = _pairs(sensors, TOES)
    w, x, y, z = np.moveaxis(sensors[..., ORIENTATION], -1, 0)
    side = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=-1)
    differences = np.stack([part[..., 0, :] - part[..., 1, :] for part in [knees, ankles, toes]], axis=-2)
    gaps = np.linalg.norm(differences, axis=-1)
    signed = np.sum(differences * side[..., None, :], axis=-1)
    return dict(kneeGap=gaps[..., 0], ankleGap=gaps[..., 1], toeGap=gaps[..., 2],
                crossedLegs=np.any(signed <= .02, axis=-1), signedLegGaps=signed)


def entry_geometry(sensors):
    """Foot-line angles, hand-axis angles and hand placement; radians and metres."""
    hands = _pairs(sensors, HANDS)
    knees, ankles = _ankles(sensors)
    toes = _pairs(sensors, TOES)
    shin = ankles - knees
    toe = toes - ankles
    length = np.linalg.norm(shin, axis=-1) * np.linalg.norm(toe, axis=-1)
    cosine = np.sum(shin * toe, axis=-1) / np.maximum(length, 1e-9)
    foot = np.where(length > 1e-9, np.arccos(np.clip(cosine, -1, 1)), np.pi)
    hand_quaternions = np.stack([sensors[..., HAND_LEFT_QUAT], sensors[..., HAND_RIGHT_QUAT]], axis=-2)
    hand_up_z = 1 - 2 * (hand_quaternions[..., 1] ** 2 + hand_quaternions[..., 2] ** 2)
    unit = np.abs(np.linalg.norm(hand_quaternions, axis=-1) - 1) < 1e-3
    hand = np.where(unit, np.arccos(np.clip(hand_up_z, -1, 1)), np.pi)
    return dict(**leg_geometry(sensors), footLineAngles=foot, handAxisAngles=hand,
                handSeparation=np.linalg.norm(hands[..., 0, :] - hands[..., 1, :], axis=-1),
                handHeightGap=np.abs(hands[..., 0, 2] - hands[..., 1, 2]))


def entry_arm_errors(joints, headfirst):
    """Matching entry references for continuous feedback and final judging."""
    head = np.asarray(headfirst, dtype=bool)
    pitch = joints[..., [6, 9]] - np.where(head, 3.05, 0)[..., None]
    roll = joints[..., [7, 10]] - head[..., None] * np.array([-.3, .3])
    elbow = joints[..., [8, 11]]
    return pitch, roll, elbow


def entry_arm_assessment(joints, sensors, headfirst, arm_mask=True, elbow_mask=True, hand_mask=True):
    """Separate imperfect arm form from the severe wrong-end arm rule.

    Compare each hand with the head along the body's own axis, so inversion
    does not reverse 'above the head'. The cap uses actual hand geometry;
    shoulder roll and bent elbows remain independent form faults.
    """
    head = np.asarray(headfirst, dtype=bool)
    pitch, roll, elbow = entry_arm_errors(joints, head)
    valid = np.all((~np.asarray(arm_mask) | ((np.abs(pitch) < .6) & (np.abs(roll) < .5)))
                   & (~np.asarray(elbow_mask) | (np.abs(elbow) < .5)), axis=-1)
    head_center = sensors[..., 59:62]
    up = quat_up(sensors[..., ORIENTATION])
    along = np.sum((_pairs(sensors, HANDS) - head_center[..., None, :]) * up[..., None, :], axis=-1)
    cap = np.any(np.where(head[..., None], along < -.02, along > .02) & hand_mask, axis=-1)
    return valid & ~cap, cap
