"""Evaluate frozen original PPO actors, including every failed episode.
No training and no selection on the evaluation seed. MuJoCo native uses the same
MJCF, 2 ms integration and 20 ms observations as the browser runtime.
"""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
from sim import Arena, ROOT, SKILLS, ACTION_LOW, ACTION_HIGH, quat_up, tuck_geometry

from contract import JUDGE_VERSION, runtime_contract, validate_contract
DD=[1.4,1.5,1.6,1.7,1.9,2.1]

def actor(path,allow_cross_physics=False):
 payload=Path(path).read_bytes();data=json.loads(payload)
 try:validate_contract(data.get('contract'))
 except ValueError:
  if not allow_cross_physics:raise
  data['crossPhysicsBaseline']=True
 data['loadedSHA256']=hashlib.sha256(payload).hexdigest();data['_sourcePayload']=payload;layers=[(np.asarray(l['weight'],np.float64),np.asarray(l['bias'],np.float64)) for l in data['layers']]
 def infer(obs):
  x=obs
  for i,(w,b) in enumerate(layers):
   # Match the exported browser arithmetic: JSON coefficients and each dot
   # product use float64; each completed layer is stored as float32. PPO itself
   # uses Torch float32. Evaluations target the actual frozen runtime actor.
   x=x.astype(np.float64)@w.T+b
   if i<len(layers)-1:x=np.tanh(x)
   elif data.get("outputActivation")=="squashed-mean-v1":x=np.tanh(2*np.tanh(x/2))
   x=x.astype(np.float32)
  return x
 return infer,data

def baseline(env):
 """Explicit joint-feedback reference, no privileged future state or root impulses.
 Tighten when remaining rotation exceeds current speed times estimated airtime;
 open when ahead. An asymmetric arm pair attempts axial twist when requested.
 """
 s=env.sensors;g=env.goal();h=np.maximum(s[:,2]+env.height-.7,0)
 tgo=np.clip((s[:,5]+np.sqrt(s[:,5]**2+19.62*h))/9.81,.08,3)
 remaining=g[:,0]*2*np.pi-env.phase_theta
 open_speed=np.maximum(0,s[:,7])/18
 tight=(remaining>open_speed*tgo+np.maximum(0,s[:,14]-open_speed)*.14+.05)&(tgo>.3)&env.released
 target=np.tile([0.,0.,1.35,2.8,2.8,-.24,.24,0.],(env.n,1))
 board=np.stack([np.where(g[:,2]>0,np.where(g[:,0]<1,1.5,.5),np.where(g[:,3]==1,1.5,.5)),np.where((g[:,0]<1)|(g[:,2]>0),2,0),np.full(env.n,-.5),np.full(env.n,3.05),np.full(env.n,3.05),np.zeros(env.n),np.zeros(env.n),np.zeros(env.n)],axis=1)
 target[~env.released]=board[~env.released]
 target[tight]=np.stack([np.full(env.n,1.7),np.where(g[:,3]==1,.05,2.3),np.full(env.n,.3),np.full(env.n,.4),np.full(env.n,.4),np.zeros(env.n),np.zeros(env.n),np.full(env.n,.8)],axis=1)[tight]
 # Body-axis angular speed is directly observable. This is an explicitly authored
 # controller, supplied as a transparent reference rather than an optimal planner.
 axial=np.sum(s[:,13:16]*quat_up(env.state[:,5:9]),axis=1)
 twist=(g[:,1]*2*np.pi-env.air_twist>np.maximum(axial,0)*tgo+.2)&(tgo>.4)&env.released&(g[:,1]>0)
 target[twist,3]=2.8;target[twist,4]=.3;target[twist,5]=-.8;target[twist,6]=.8
 target=np.concatenate([target,np.where(env.released,.06,0)[:,None]],axis=1)
 return 2*(target-ACTION_LOW)/(ACTION_HIGH-ACTION_LOW)-1

def score(info):
 sid=info['skill'];angle=info['entryAngle'];form=info['form']
 numeric=abs(info['error'])<.45 and abs(info['twistError'])<.55 and abs(info['airTwistError'])<.55 and info['maxLateral']<.85
 position=SKILLS[sid]['shape']==2 or info['shapeFraction']>=.65
 entry=info['water'] and info.get('fullEntryComplete',False) and info.get('entryGeometryValid',False) and info.get('entryLimbsValid',False) and angle<35 and form>.55 and info['firstGeometry'] in (6,9) and info['x']>.55
 splash=float(np.clip(.55*np.sin(np.radians(angle))**2+.25*(1-form)+.2*info.get('surfaceLateralSpeed',info['horizontalSpeed'])/3,0,1))
 deductions={'height':float(1.5*(1-np.clip(info['ascent']/.3,0,1))),'preparation':float(min(2,info['preparationBounces'])),'form':float(np.clip((1-form)*3,0,3)),'entry':float(np.clip(angle/10,0,4)),'splash':1.5*splash,'takeoff':float(np.clip(info['boardTwist'],.5,2)) if info['boardTwist']>.26 else 0.}
 execution=float(np.clip(10-sum(deductions.values()),0,10))
 if not position or not entry:execution=min(execution,2)
 if not numeric or not info['water'] or info['boardInvalid']:execution=0
 execution=math.floor(execution*2+.5)/2
 return dict(cleanEntry=bool(entry and angle<=15 and form>=.8),splashIsProxy=True,valid=bool(not info['boardInvalid'] and numeric and position and entry),launchValid=not info['boardInvalid'],numericCorrect=bool(numeric),shapeCorrect=bool(position),validEntry=bool(entry),execution=execution,points=3*execution*DD[sid],difficulty=DD[sid],splash=splash,deductions=deductions)

def run(contexts,infer=None,*,mode='pretrained',nominal_actions=None,record=False):
 """Bounded vectorized evaluation. Capture only the first terminal state per case."""
 results=[];traces=[]
 for start in range(0,len(contexts),128):
  cs=contexts[start:start+128];env=Arena(len(cs),seed=888,training=False);env.reset(np.arange(len(cs)),cs)
  complete=np.zeros(len(cs),bool);rows=[None]*len(cs);commands=[]
  hand_sum=np.zeros(len(cs));hand_ticks=np.zeros(len(cs));hand_peak=np.zeros(len(cs))
  for t in range(231):
   if nominal_actions is not None:
    a=np.asarray([nominal_actions[start+i][min(t,len(nominal_actions[start+i])-1)] for i in range(len(cs))],np.float32)
   else:a=baseline(env) if mode=='baseline' else infer(env.observe())
   commands.append(np.clip(a,-1,1).copy())
   _,_,done,info=env.step(a,auto_reset=False)
   distances,_=tuck_geometry(env.sensors);maximum=distances.max(axis=1);goal=env.goal()[:,0]*2*np.pi
   central=(env.air_theta>.15*goal)&(env.air_theta<.7*goal)&env.released&(~complete)
   hand_sum+=maximum*central;hand_ticks+=central;hand_peak=np.maximum(hand_peak,maximum*central)
   for r in info:
    i=r['index']
    if not complete[i]:
     complete[i]=True;rows[i]={**cs[i],**r,**score(r),'parameters':dict(cs[i]),'actionSteps':t+1};rows[i].pop('index',None);rows[i].update(centralTuckHandDistanceMean=float(hand_sum[i]/max(hand_ticks[i],1)),centralTuckHandDistanceMax=float(hand_peak[i]),centralTuckSamples=int(hand_ticks[i]))
   if complete.all():break
  if not complete.all():raise RuntimeError('A bounded episode did not terminate')
  if record:
   commands=np.asarray(commands)
   traces.extend([commands[:r['actionSteps'],i].tolist() for i,r in enumerate(rows)])
  results.extend(rows);env.close()
 return results,traces

def summary(rows):
 # Scalar case fields remain first-contact telemetry. Summary geometry explicitly
 # reports extrema over the full judged water interval whenever sampled.
 rows=[{**r,**(r.get('entryGeometryWorst')or {}),'geometryLimits':r.get('entryGeometryValid',r['geometryLimits'])}for r in rows]
 n=len(rows);p=float(np.mean([r['valid'] for r in rows]));z=1.95996398454;den=1+z*z/n
 mid=(p+z*z/(2*n))/den;radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 execution=np.asarray([r['execution'] for r in rows]);clean=np.asarray([r['valid'] and r['execution']>=6 for r in rows]);cp=float(clean.mean());cmid=(cp+z*z/(2*n))/den;cradius=z*np.sqrt(cp*(1-cp)/n+z*z/(4*n*n))/den
 return dict(geometrySummaryScope='Worst physical measurements over the scored entry interval; individual case scalar fields remain first-contact values',fullEntryCompleted=sum(bool(r.get('fullEntryComplete'))for r in rows),failureCounts={'countsMayOverlap':True,'launch':sum(r['boardInvalid'] for r in rows),'numericRotation':sum(not r['numericCorrect'] for r in rows),'declaredPosition':sum(not r['shapeCorrect'] for r in rows),'entry':sum(not r['validEntry'] for r in rows),'footLine':sum(max(r['footLineAngles'])>=20 for r in rows),'handAxis':sum(max(r['handAxisAngles'])>=30 for r in rows),'handSeparation':sum(r['handSeparation']>=.12 for r in rows),'handHeight':sum(r['handHeightGap']>=.04 for r in rows),'legGap':sum(r['kneeGap']>=.19 or r['ankleGap']>=.16 or r['toeGap']>=.15 for r in rows),'crossedLegs':sum(r['crossedLegs'] for r in rows),'executionBelowSix':sum(r['execution']<6 for r in rows)},entryGeometry={key:{'mean':float(np.mean([np.max(r[key]) for r in rows])),'quantiles':{str(q):float(np.quantile([np.max(r[key]) for r in rows],q)) for q in [0,.5,.9,.99,1]}} for key in ['footLineAngles','handAxisAngles','handSeparation','handHeightGap','kneeGap','ankleGap','toeGap','ascent','takeoffVerticalSpeed']},geometryPassRate=float(np.mean([r['geometryLimits'] for r in rows])),meanCentralTuckHandDistance=float(np.mean([r['centralTuckHandDistanceMean'] for r in rows if r.get('centralTuckSamples',0)>0])) if any(r.get('centralTuckSamples',0)>0 for r in rows) else None,launchInvalid=sum(r['boardInvalid'] for r in rows),launchClearanceRate=float(np.mean([not r['boardInvalid'] for r in rows])),boardNonfootPeakMax=max(r['boardNonfootPeak'] for r in rows),standPeakMax=max(r['standPeak'] for r in rows),cleanWithInvalidLaunch=sum(r['valid'] and r['boardInvalid'] for r in rows),clean95Wilson=[cmid-cradius,cmid+cradius],cleanEntryRate=float(np.mean([r['cleanEntry'] for r in rows])),cleanCompleted=int(clean.sum()),cleanCompletionRate=float(clean.mean()),executionQuantiles={str(q):float(np.quantile(execution,q)) for q in [0,.1,.25,.5,.75,.9,1]},executionHistogram={str(x/2):int(np.sum(execution==x/2)) for x in range(21)},episodes=n,completed=sum(r['valid'] for r in rows),completionRate=p,completion95Wilson=[mid-radius,mid+radius],meanExecution=float(np.mean([r['execution'] for r in rows])),meanPoints=float(np.mean([r['points'] for r in rows])),meanEntryAngle=float(np.mean([r['entryAngle'] for r in rows])),meanShapeFraction=float(np.mean([r['shapeFraction'] for r in rows])),numericRate=float(np.mean([r['numericCorrect'] for r in rows])),positionRate=float(np.mean([r['shapeCorrect'] for r in rows])))

def contexts(seed,n,height=None,stress=False,minimum_heights=None):
 rng=np.random.default_rng(seed);rows=[]
 for s,skill in enumerate(SKILLS):
  for i in range(n):
   lo=minimum_heights[s] if minimum_heights else skill['height'];h=height if height is not None else rng.uniform(lo,10)
   rows.append(dict(skill=s,height=float(h),tilt=float(rng.uniform(.10,.22)),preload=float(rng.uniform(-.20,-.16)),x=float(rng.uniform(-.16,-.10)),disturbance=float(rng.choice([-120,120])) if stress else float(rng.uniform(-80,80)) if rng.random()<.3 else 0.,disturbanceTime=float(rng.uniform(.65,1.05))))
 return rows

def main(args):
 started=time.time();infer,data=actor(args.policy,allow_cross_physics=args.cross_physics_baseline);initial,initial_data=actor(args.initial,allow_cross_physics=True)
 if data.get('crossPhysicsBaseline') and not args.source_xml and not data.get('contract',{}).get('xmlSHA256'):raise ValueError('An explicit cross-physics baseline needs --source-xml to identify the original physical model.')
 source_xml_hash=hashlib.sha256(Path(args.source_xml).read_bytes()).hexdigest() if args.source_xml else data.get('contract',{}).get('xmlSHA256')
 heldout=contexts(args.seed,args.per_skill,minimum_heights=args.minimum_heights)
 report={'sourcePhysicsSHA256':source_xml_hash,'crossPhysicsBaseline':bool(data.get('crossPhysicsBaseline')),'policyContract':data.get('contract'),'initialContract':initial_data.get('contract'),'initialInterpretation':'Old initial weights evaluated under the new judge; not an old-judge score','crossPhysicsInterpretation':'Explicit cross-physics diagnostic; source and runtime contracts differ. Architectural changes require a separately recorded weight migration.' if data.get('crossPhysicsBaseline') else None,'contract':runtime_contract(),'schemaVersion':2,'seed':args.seed,'modelSteps':data['steps'],'minimumHeights':args.minimum_heights or [s['height']for s in SKILLS],'policySHA256':data['loadedSHA256'],'initialSHA256':initial_data['loadedSHA256'],'physicsSHA256':hashlib.sha256((ROOT/'public/physics/diver.xml').read_bytes()).hexdigest(),'judge':JUDGE_VERSION,'sourceSHA256':{name:hashlib.sha256((ROOT/'training_v2'/name).read_bytes()).hexdigest() for name in ['sim.py','evaluate.py']},'policies':{},'perSkill':{},'heightGrid':{},'outOfDistribution':{},'feedback':{}}
 all_cases={}
 for name,fn in [('pretrained',infer),('initial',initial),('baseline',None)]:
  rows,_=run(heldout,fn,mode=name);all_cases[name]=rows;report['policies'][name]=summary(rows)
  report['perSkill'][name]={SKILLS[s]['id']:summary([r for r in rows if r['skill']==s]) for s in range(6)}
  print(name,json.dumps(report['policies'][name]),flush=True)
 if not args.quick:
  grid=[]
  for h in [3,4,5,6,7.5,9,10]:
   grid.extend(contexts(args.seed+int(h*10),args.grid_per_skill,height=h))
  rows,_=run(grid,infer);all_cases['heightGrid']=rows
  report['heightGrid']={str(h):{SKILLS[s]['id']:summary(group) for s in range(6) if (group:=[r for r in rows if r['skill']==s and r['height']==h])} for h in [3,4,5,6,7.5,9,10]}
  stress=contexts(args.seed+3,args.grid_per_skill,height=11,stress=True)
  rows,_=run(stress,infer);all_cases['outOfDistribution']=rows;report['outOfDistribution']={'description':'11 m, outside the 3–10 m training range; ±120 N pushes outside the ±80 N training range. Excluded from eligibility.','overall':summary(rows),'perSkill':{SKILLS[s]['id']:summary([r for r in rows if r['skill']==s]) for s in range(6)}}
  disturbed=contexts(args.seed+4,args.grid_per_skill,stress=True,minimum_heights=args.minimum_heights)
  for c in disturbed:c['disturbanceTime']=.85
  nominal=[{**c,'disturbance':0} for c in disturbed]
  nominal_rows,commands=run(nominal,infer,record=True)
  closed,closed_commands=run(disturbed,infer,record=True)
  replay,_=run(disturbed,infer,nominal_actions=commands)
  before=[];after=[]
  for a,b in zip(commands,closed_commands):
   n=min(len(a),len(b));delta=np.mean(np.abs(np.asarray(a[:n])-np.asarray(b[:n])),axis=1)
   before.extend(delta[:43]);after.extend(delta[45:])
  report['feedback']={'description':'Paired identical starts; ±120 N torso force for 0.12 s from t=0.85 s. Compare fresh feedback actions with replay of the undisturbed actor commands.','nominal':summary(nominal_rows),'closedLoop':summary(closed),'nominalCommandReplay':summary(replay),'appliedPush':{'episodes':len(closed),'fullDurationCount':sum(abs(r['pushDuration']-.12)<1e-6 for r in closed),'minDuration':min(r['pushDuration'] for r in closed),'maxDuration':max(r['pushDuration'] for r in closed),'meanAbsoluteImpulse':float(np.mean([abs(r['appliedImpulse']) for r in closed]))},'meanAbsoluteActionDifferenceBeforeForce':float(np.mean(before)),'meanAbsoluteActionDifferenceAfterForce':float(np.mean(after))}
  all_cases.update(feedbackNominal=nominal_rows,feedbackClosedLoop=closed,feedbackOpenLoop=replay)
 report['seconds']=time.time()-started
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'policy.json').write_bytes(data['_sourcePayload']);(out/'initial.json').write_bytes(initial_data['_sourcePayload']);(out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');(out/'cases.json').write_text(json.dumps(all_cases,separators=(',',':'))+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--policy',default=str(ROOT/'training/ppo-height-curriculum/policy.json'));p.add_argument('--initial',default=str(ROOT/'training/ppo-probe/initial.json'));p.add_argument('--seed',type=int,default=990517);p.add_argument('--per-skill',type=int,default=128);p.add_argument('--grid-per-skill',type=int,default=48);p.add_argument('--quick',action='store_true');p.add_argument('--cross-physics-baseline',action='store_true');p.add_argument('--source-xml');p.add_argument('--minimum-heights',nargs=6,type=float);p.add_argument('--output',default=str(ROOT/'training/evaluation-current'));main(p.parse_args())
