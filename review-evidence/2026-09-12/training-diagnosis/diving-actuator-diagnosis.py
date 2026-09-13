import sys,json,pathlib,numpy as np,mujoco
root=pathlib.Path('/Users/zacharymartin/Desktop/GraphicsExperiments/LearnToDive');sys.path.insert(0,str(root/'training_v3'))
from environment import Arena
from engine import STATE_SPEC
from geometry import encoded_action,entry_geometry
from rules import IDS
from audit_takeoffs import probe
e=Arena(3,seed=13,threads=1,training=False)
targets=np.array([[0,0,1.35,3.05,3.05,-.23,.23,0,.07],[1.4,2,1.3,1.,1.,0,0,1.2,0],[1.5,0,1.35,.5,.5,0,0,0,0]])
try:
 e.group[:]=1;e.apparatus[:]=1;e.height[:]=10;e.used[:]=False
 for i in range(3):
  e.declare(i,IDS['101C']);d=mujoco.MjData(e.physics.model)
  mujoco.mj_setState(e.physics.model,d,e.physics.state[i],STATE_SPEC)
  d.qpos[1]=3;d.qpos[3]=8;d.qvel[:]=0;mujoco.mj_forward(e.physics.model,d)
  mujoco.mj_getState(e.physics.model,d,e.physics.state[i],STATE_SPEC);e.physics.sensors[i]=d.sensordata
 for _ in range(50):e.physics.step(encoded_action(targets),auto_reset=False)
 qs=e.physics.state[:,1+e.physics.qadr];g=entry_geometry(e.physics.sensors)
 rows=[]
 for i,name in enumerate(['overhead-entry','tuck','pike']):
  rows.append(dict(name=name,targets=targets[i].tolist(),hip=float(qs[i,0]),knee=float(qs[i,1]),shoulderPitch=qs[i,[6,9]].tolist(),elbow=qs[i,[8,11]].tolist(),handSeparation=float(g['handSeparation'][i]),footLineDegrees=np.degrees(g['footLineAngles'][i]).tolist()))
finally:e.close()
result=dict(trainingData=False,meaning='Isolated joint capability only; airborne reset, no valid dive claimed',poses=rows,takeoff=[probe('platform',7.5,g,16) for g in [1,2,6]])
out=root/'review-evidence/2026-09-12/training-diagnosis/actuator-probes.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
