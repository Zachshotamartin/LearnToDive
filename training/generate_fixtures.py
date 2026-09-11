"""Generate native-engine golden states for independent browser-WASM parity tests."""
import json,numpy as np
from sim import Arena,ROOT
from evaluate import baseline,score
cases=[]
for context in [dict(height=7.5,skill=1,tilt=.16,preload=-.18,x=-.13,disturbance=0,disturbanceTime=.9),dict(height=5.3,skill=3,tilt=.21,preload=-.195,x=-.12,disturbance=100,disturbanceTime=.7)]:
 env=Arena(1,threads=1,training=False);env.reset([0],[context]);steps=[]
 for i in range(231):
  action=[float(.7*np.sin(i*.09+j*.7)) for j in range(9)]
  _,_,done,info=env.step(np.array([action]),False)
  steps.append({'action':action,'state':env.state[0].tolist(),'observation':env.observe()[0].tolist(),'theta':float(env.theta[0]),'twist':float(env.twist[0]),'entryTime':float(env.entry_time[0]) if np.isfinite(env.entry_time[0]) else None,'fullEntryTime':float(env.full_entry_time[0]) if np.isfinite(env.full_entry_time[0]) else None,'released':bool(env.released[0]),'done':bool(done[0]),'baselineAction':baseline(env)[0].tolist(),'shapeAngle':float(env.shape_angle[0]),'score':score(info[0])if info else None})
  if done[0]:break
 cases.append({'parameters':context,'steps':steps});env.close()
(ROOT/'tests/fixtures/mujoco-parity.json').write_text(json.dumps({'engine':'MuJoCo3.13.0','cases':cases},separators=(',',':')))
