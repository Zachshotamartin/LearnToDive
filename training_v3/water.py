"""Stateless water approximation for the v11 native trainer.

The browser keeps its own v8 mirror in ``src/core/water.js``. SI units; native
world Z up. One submerged-envelope centre per physical body. Directional
projected areas use collider dimensions, never bounding-sphere radius.
Quadratic drag is passive; buoyancy is external work and excluded from
``dragPower``.

Flow along a slender part's axis uses a streamlined coefficient: summing the
bluff coefficient over fifteen overlapping parts decelerated a straight diver
at about nine g and stopped the centre of mass 1.2 m below the surface,
whereas real head-first entries from 10 m reach roughly 3 m. Broadside (bluff)
flow keeps the original coefficient, so a flat entry is still punished by the
water. No added mass, free surface history, shielding, lift or CFD.
"""
import numpy as np

WATER_VERSION = 'directional-submerged-body-v2-streamlined-axial'
POOL_DEPTH = 5.
DENSITY = 1000.
DRAG_COEFFICIENT = .8          # bluff (broadside) flow
AXIAL_DRAG_COEFFICIENT = .15   # flow along a capsule/ellipsoid axis or a pointed foot
SPHERE_DRAG_COEFFICIENT = .35  # head, mostly in the wake of the arms or torso
ANGULAR_COEFFICIENT = 1.
BUOYANCY_MASS_RATIO = 1.015
GRAVITY = 9.81
# MuJoCo geom type codes.
SPHERE = 2
CAPSULE = 3
ELLIPSOID = 4
BOX = 6
# Horizontal pool extent in native coordinates (the board tip is at x = 0).
POOL_X = (-.44, 11.34)
POOL_HALF_WIDTH = 3.56


def rotation(q):
    """Rotation matrices (..., 3, 3) of quaternion rows (w, x, y, z)."""
    w, x, y, z = np.moveaxis(np.asarray(q), -1, 0)
    rows = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    return np.stack(rows, axis=-1).reshape(*np.shape(w), 3, 3)


def shape_parameters(types, sizes):
    """Per-geom local half extents and the projected area along each local axis."""
    types = np.asarray(types)
    sizes = np.asarray(sizes)
    r = sizes.copy()
    r = np.where((types == SPHERE)[..., None], sizes[..., :1], r)
    r = np.where((types == CAPSULE)[..., None],
                 np.stack([sizes[..., 0], sizes[..., 0], sizes[..., 0] + sizes[..., 1]], axis=-1), r)
    area = np.stack([np.pi * r[..., 1] * r[..., 2], np.pi * r[..., 0] * r[..., 2], np.pi * r[..., 0] * r[..., 1]],
                    axis=-1)
    area = np.where((types == BOX)[..., None], area * (4 / np.pi), area)
    capsule = np.stack([4 * sizes[..., 0] * sizes[..., 1] + np.pi * sizes[..., 0] ** 2] * 2
                       + [np.pi * sizes[..., 0] ** 2], axis=-1)
    area = np.where((types == CAPSULE)[..., None], capsule, area)
    return r, area


def drag_coefficients(types):
    """Per-geom local-frame coefficients: bluff broadside, streamlined along the axis."""
    types = np.asarray(types)
    c = np.full((*types.shape, 3), DRAG_COEFFICIENT)
    c = np.where((types == SPHERE)[..., None], SPHERE_DRAG_COEFFICIENT, c)
    axial = np.stack([types == BOX, np.zeros(types.shape, bool), (types == CAPSULE) | (types == ELLIPSOID)], axis=-1)
    return np.where(axial, AXIAL_DRAG_COEFFICIENT, c)


def vertical_extent(types, sizes, row):
    """Half height of each geom's vertical envelope from the world-z row of its rotation."""
    extent = np.sqrt(np.sum((row * sizes) ** 2, axis=-1))
    extent = np.where(types == SPHERE, sizes[..., 0], extent)
    extent = np.where(types == CAPSULE, sizes[..., 0] + sizes[..., 1] * np.abs(row[..., 2]), extent)
    return np.where(types == BOX, np.sum(np.abs(row) * sizes, axis=-1), extent)


def immersion(types, sizes, position, quaternion, water_z):
    """Submerged fraction of each geom's vertical envelope, the envelope half height and rotations."""
    R = rotation(quaternion)
    types = np.asarray(types)
    sizes = np.asarray(sizes)
    extent = vertical_extent(types, sizes, R[..., 2, :])
    fraction = np.clip((np.asarray(water_z) - position[..., 2] + extent) / np.maximum(2 * extent, 1e-9), 0, 1)
    inside = ((position[..., 0] >= POOL_X[0]) & (position[..., 0] <= POOL_X[1])
              & (np.abs(position[..., 1]) <= POOL_HALF_WIDTH))
    return fraction * inside, extent, R


def body_water(types, sizes, position, quaternion, com, linear, angular, mass, water_z):
    """Water force and torque at each body's centre of mass.

    Arrays are vectorizable with trailing dimensions 3 or 4. ``linear`` is the
    velocity at the centre of mass, not at the geom origin. A vertical envelope
    approximates the wet volume and its centroid; it is deliberately not an
    exact polygon or ellipsoid slice integration.
    """
    fraction, extent, R = immersion(types, sizes, position, quaternion, water_z)
    r, area = shape_parameters(types, sizes)
    center = np.array(position, copy=True)
    center[..., 2] -= extent * (1 - fraction)
    offset = center - com
    velocity = linear + np.cross(angular, offset)
    local_v = np.einsum('...ji,...j->...i', R, velocity)
    local_w = np.einsum('...ji,...j->...i', R, angular)
    local_f = (-.5 * DENSITY * drag_coefficients(types) * area * fraction[..., None] * local_v
               * np.linalg.norm(local_v, axis=-1, keepdims=True))
    k = .5 * DENSITY * ANGULAR_COEFFICIENT * r * (np.roll(r, 1, axis=-1) ** 4 + np.roll(r, 2, axis=-1) ** 4)
    local_t = -k * fraction[..., None] * local_w * np.linalg.norm(local_w, axis=-1, keepdims=True)
    drag = np.einsum('...ij,...j->...i', R, local_f)
    angular_drag = np.einsum('...ij,...j->...i', R, local_t)
    force = drag.copy()
    force[..., 2] += GRAVITY * np.asarray(mass) * BUOYANCY_MASS_RATIO * fraction
    torque = angular_drag + np.cross(offset, force)
    drag_power = np.sum(local_f * local_v + local_t * local_w, axis=-1)
    return dict(force=force, torque=torque, dragForce=drag, dragTorque=angular_drag + np.cross(offset, drag),
                dragPower=drag_power, fraction=fraction, extent=extent, area=area, center=center)


def apply_native(model, data, height):
    """Reference bridge used by exact prototype and parity tests; no owning handles."""
    import mujoco
    ids = np.arange(1, 16)
    bodies = model.geom_bodyid[ids]
    velocity = np.empty((15, 6))
    for i, body in enumerate(bodies):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(body), velocity[i], 0)
    quaternions = data.geom_xquat[ids] if hasattr(data, 'geom_xquat') else data.sensordata[45:150].reshape(15, 7)[:, 3:]
    result = body_water(model.geom_type[ids], model.geom_size[ids], data.geom_xpos[ids], quaternions,
                        data.xipos[bodies], velocity[:, 3:], velocity[:, :3], model.body_mass[bodies], -height)
    data.xfrc_applied[:] = 0
    data.xfrc_applied[bodies, :3] = result['force']
    data.xfrc_applied[bodies, 3:] = result['torque']
    return result
