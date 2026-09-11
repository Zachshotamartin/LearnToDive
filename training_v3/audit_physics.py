"""Actuator exerciser and free-flight conservation checks; never training data."""
import json

import mujoco
import numpy as np

from engine import ACTUATOR_MAP, STATE_SPEC
from environment import Arena
from rules import IDS

MOMENTUM_DRIFT_LIMIT = .15   # kg m^2/s of angular momentum change allowed over the free-flight exercise
EXERCISE_TICKS = 12
ATHLETES = 4


def free_flight(e):
    """Replace every athlete's state with a disposable free-flight pose away from the board."""
    m = e.physics.model
    for i in range(ATHLETES):
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, e.physics.state[i], STATE_SPEC)
        d.qpos[1] = 3
        d.qpos[3] = 4
        d.qvel[:] = 0
        d.qvel[5] = 2
        mujoco.mj_forward(m, d)
        mujoco.mj_getState(m, d, e.physics.state[i], STATE_SPEC)
        e.physics.sensors[i] = d.sensordata


def peak_actuator_forces(e, observed):
    m = e.physics.model
    for i in range(ATHLETES):
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, e.physics.state[i], STATE_SPEC)
        d.ctrl[:] = e.physics.targets[i, ACTUATOR_MAP]
        mujoco.mj_forward(m, d)
        observed = np.maximum(observed, np.abs(d.actuator_force))
    return observed


def audit():
    e = Arena(ATHLETES, seed=55, threads=1, training=False)
    m = e.physics.model
    try:
        e.group[:] = 1
        e.apparatus[:] = 1
        e.height[:] = 10
        e.used[:] = False
        e.step(np.full(ATHLETES, IDS['101C']), np.zeros((ATHLETES, 9)))
        # A disposable free-flight state isolates joint motion from launch contacts.
        free_flight(e)
        before = e.physics.sensors[:, 6:9].copy()
        observed = np.zeros(m.nu)
        rng = np.random.default_rng(315)
        for _ in range(EXERCISE_TICKS):
            e.step(np.zeros(ATHLETES, int), rng.uniform(-1, 1, (ATHLETES, 9)))
            observed = peak_actuator_forces(e, observed)
        after = e.physics.sensors[:, 6:9]
        error = float(np.max(np.linalg.norm(after - before, axis=1)))
        caps = np.max(np.abs(m.actuator_forcerange), axis=1)
        if np.any(observed > caps + 1e-7):
            raise AssertionError('Actuator torque exceeded its physical cap')
        if error > MOMENTUM_DRIFT_LIMIT:
            raise AssertionError(f'Free-flight angular momentum drift: {error}')
        actuators = [dict(name=m.actuator(i).name, maxTorqueNm=float(caps[i]), observedMaxTorqueNm=float(observed[i]))
                     for i in range(m.nu)]
        return dict(angularMomentumMaxDrift=error, actuators=actuators, rootActions=False,
                    trainingUsesTheseTrajectories=False)
    finally:
        e.close()


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
