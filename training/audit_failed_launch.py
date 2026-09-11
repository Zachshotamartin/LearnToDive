"""Preserve failed diagnostic trajectories with their exact earlier physics source.
This audit intentionally uses the archived, superseded environment. It is evidence
of a defect found during development, never a current qualification benchmark.
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np
import mujoco
from evaluate import actor
from sim import ROOT

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 source=ROOT/'training/ppo-qualified-heights/source';policy=ROOT/'training/ppo-qualified-heights/policy.json'
 code=(source/'sim.py').read_text().replace('ROOT=Path(__file__).resolve().parents[1]',f'ROOT=Path({str(ROOT)!r})').replace("str(ROOT/'public/physics/diver.xml')",repr(str(source/'diver.xml')))
 ns={'__file__':str(source/'sim.py')};exec(compile(code,str(source/'sim.py'),'exec'),ns)
 # This explicit legacy bypass is safe only for this archived-source audit:
 # the environment is loaded from the exact old XML/source above, never v6.
 infer,model=actor(policy,allow_cross_physics=True);cases=[]
 for skill in range(6):
  env=ns['Arena'](1,training=False);context=dict(skill=skill,height=10 if skill==5 else 7.5,tilt=.16,preload=-.18,x=-.13,disturbance=0,disturbanceTime=.9);env.reset([0],[context])
  data=mujoco.MjData(env.model);bad=[];foot=[];wrench=np.zeros(6);last_time=-1
  for i in range(181):
   _,_,done,infos=env.step(infer(env.observe()),False)
   for state in env.roll_state[0]:
    if state[0]<=last_time:continue
    last_time=float(state[0]);mujoco.mj_setState(env.model,data,state,ns['STATE_SPEC']);mujoco.mj_forward(env.model,data)
    for ci,contact in enumerate(data.contact):
     pair=list(contact.geom)
     if 0 not in pair:continue
     geom=pair[1]if pair[0]==0 else pair[0]
     mujoco.mj_contactForce(env.model,data,ci,wrench)
     force=float(np.linalg.norm(wrench[:3]))
     if force<15:continue
     row=dict(time=float(data.time),athleteGeometry=int(geom),force=force,penetration=float(contact.dist))
     (foot if geom in (12,15) else bad).append(row)
   if done[0]:break
  cases.append(dict(context=context,oldJudge=infos[0]if infos else None,nonfootContactSamples=len(bad),nonfootPeak=max([x['force']for x in bad],default=0),firstNonfootContact=bad[0]if bad else None,nonfootContacts=bad,footContacts=foot))
  env.close()
 report=dict(status='FAILED DEVELOPMENT EVIDENCE — collision assistance; not release qualification',modelSteps=model['steps'],policySHA256=sha(policy),sourceSHA256={p.name:sha(p)for p in [source/'sim.py',source/'diver.xml']},cases=cases)
 out=ROOT/'training/diagnostics/invalid-launch';out.mkdir(parents=True,exist_ok=True)
 (out/'contact-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'policy.json').write_bytes(policy.read_bytes())
 for p in [source/'sim.py',source/'diver.xml']:(out/p.name).write_bytes(p.read_bytes())
 print(json.dumps([{k:c[k]for k in ['context','nonfootPeak','firstNonfootContact']}for c in cases],indent=2))
if __name__=='__main__':main()
