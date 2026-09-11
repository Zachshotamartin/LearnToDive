"""Actuator exerciser and free-flight conservation tests, never training data."""
import json
import numpy as np
import mujoco
from environment import Arena
from engine import ACTION_LOW,ACTION_HIGH,ACTUATOR_MAP,STATE_SPEC
from rules import IDS

def audit():
 e=Arena(4,seed=55,threads=1,training=False);m=e.physics.model
 try:
  e.group[:]=1;e.apparatus[:]=1;e.height[:]=10;e.used[:]=False
  e.step(np.full(4,IDS['101C']),np.zeros((4,9)))
  # A disposable free-flight state isolates joint motion from launch contacts.
  for i in range(4):
   d=mujoco.MjData(m);mujoco.mj_setState(m,d,e.physics.state[i],STATE_SPEC)
   d.qpos[1]=3;d.qpos[3]=4;d.qvel[:]=0;d.qvel[5]=2
   mujoco.mj_forward(m,d);mujoco.mj_getState(m,d,e.physics.state[i],STATE_SPEC);e.physics.sensors[i]=d.sensordata
  before=e.physics.sensors[:,6:9].copy();observed=np.zeros(m.nu)
  rng=np.random.default_rng(315)
  for k in range(12):
   a=rng.uniform(-1,1,(4,9));e.step(np.zeros(4,int),a)
   for i in range(4):
    d=mujoco.MjData(m);mujoco.mj_setState(m,d,e.physics.state[i],STATE_SPEC);d.ctrl[:]=e.physics.targets[i,ACTUATOR_MAP];mujoco.mj_forward(m,d)
    observed=np.maximum(observed,np.abs(d.actuator_force))
  after=e.physics.sensors[:,6:9];error=float(np.max(np.linalg.norm(after-before,axis=1)))
  caps=np.max(np.abs(m.actuator_forcerange),axis=1)
  if np.any(observed>caps+1e-7):raise AssertionError('Actuator torque exceeded its physical cap')
  if error>.15:raise AssertionError(f'Free-flight angular momentum drift: {error}')
  return dict(angularMomentumMaxDrift=error,actuators=[dict(name=m.actuator(i).name,maxTorqueNm=float(caps[i]),observedMaxTorqueNm=float(observed[i])) for i in range(m.nu)],rootActions=False,trainingUsesTheseTrajectories=False)
 finally:e.close()
if __name__=='__main__':print(json.dumps(audit(),indent=2))
