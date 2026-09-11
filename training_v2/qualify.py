"""Check frozen release evidence before creating browser-facing metric claims.
This does not train, choose a checkpoint, alter eligibility, or publish anything.
"""
import argparse,hashlib,json
from pathlib import Path
from sim import ROOT,SKILLS
from contract import validate_contract,runtime_contract

def catalog_errors(catalog,minimum_heights):
 """A measured range cannot qualify a UI that offers untested lower heights."""
 by_id={item['id']:item for item in catalog};errors=[]
 for index,skill in enumerate(SKILLS):
  item=by_id.get(skill['id'])
  if item is None:errors.append(f"Missing runtime catalog entry: {skill['id']}")
  elif item.get('minHeight')!=minimum_heights[index]:errors.append(f"Runtime height range differs from measured eligibility: {skill['id']}")
 return errors

def check(report,runtime,manifest,protocol):
 errors=[]
 def require(condition,message):
  if not condition:errors.append(message)
 files={f['key']:f for f in manifest['files']}
 for name,contract in [('report',report.get('contract')),('policy',report.get('policyContract')),('manifest',manifest.get('contract')),('manifest policy',manifest.get('policyContract')),('WASM',runtime.get('contract'))]:
  try:validate_contract(contract)
  except ValueError as error:errors.append(f'{name}: {error}')
 require(not report.get('crossPhysicsBaseline') and not manifest.get('crossPhysicsBaseline'),'A cross-physics baseline cannot qualify as a release')
 errors.extend(catalog_errors(runtime.get('runtimeCatalog',[]),report['minimumHeights']))
 require((report.get('policyContract') or {}).get('stepsOnContract',0)>0,'No training decisions on the current contract')
 require(report['seed']==protocol['finalHeldOutSeed'],'Report is not the preregistered final held-out seed')
 require(report['policySHA256']==files['policy']['sha256']==runtime['policySHA256'],'Runtime, evaluation and public policy hashes differ')
 require(report['initialSHA256']==files['initial']['sha256'],'Initial comparison differs from the public initial weights')
 require(report['physicsSHA256']==files['xml']['sha256']==runtime['physicsSHA256'],'Physics hashes differ')
 require(report['judge']==manifest['judge']['id']==runtime['judge'],'Judge versions differ')
 require(report['modelSteps']==manifest['trainingSteps'],'Policy decision counts differ')
 for path,expected in report['sourceSHA256'].items():
  require(hashlib.sha256((ROOT/'training'/path).read_bytes()).hexdigest()==expected,f'Native source changed after evaluation: {path}')
 for path,expected in runtime.get('sourceSHA256',{}).items():
  require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,f'Runtime source changed after evaluation: {path}')
 require(set(runtime.get('sourceSHA256',{}))=={'src/core/physics.js','src/core/control.js','src/core/water.js'},'Runtime physics, control or water source hashes missing')
 readiness=protocol['readiness']
 for index,skill in enumerate(SKILLS):
  sid=skill['id']
  for name,statistics in [('native',report['perSkill']['pretrained'][sid]),('WASM',runtime['perSkillHeldOut'][sid])]:
   require(statistics['episodes']==protocol['perSkillHeldOutCases'],f'{name} {sid}: incomplete held-out sample')
   require(statistics['cleanCompletionRate']>=readiness['perSkillCleanRateAtLeast'],f'{name} {sid}: clean completion below threshold')
   require(statistics['meanExecution']>=readiness['perSkillMeanExecutionAtLeast'],f'{name} {sid}: mean execution below threshold')
  for height in protocol['heightGrid']:
   key=next((key for key in report['heightGrid'] if float(key)==height),None)
   runtime_key=next((key for key in runtime['heightGrid'] if float(key)==height),None)
   require(key is not None and runtime_key is not None,f'{sid}: missing {height} m grid evidence')
   if key is None or runtime_key is None:continue
   for name,statistics in [('native',report['heightGrid'][key][sid]),('WASM',runtime['heightGrid'][runtime_key][sid])]:
    require(statistics['episodes']==protocol['perSkillHeightBinCases'],f'{name} {sid}: incomplete {height} m sample')
    if height>=report['minimumHeights'][index]:require(statistics['cleanCompletionRate']>=readiness['eligibleHeightBinCleanRateAtLeast'],f'{name} {sid}: exposed {height} m bin below threshold')
 require(runtime['invalidLaunchCountedClean']==0,'Invalid launches were counted clean')
 require(runtime['maxTorque']<=180+1e-7,'Runtime exceeded physical actuator limits')
 feedback=report['feedback']
 require(feedback['appliedPush']['fullDurationCount']==feedback['appliedPush']['episodes'],'Feedback intervention was not fully applied to every case')
 require(feedback['meanAbsoluteActionDifferenceBeforeForce']<1e-8,'Paired actions differ before the intervention')
 require(feedback['meanAbsoluteActionDifferenceAfterForce']>1e-4,'No measured closed-loop action response to the intervention')
 return errors

def main():
 p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--write-runtime',action='store_true');args=p.parse_args()
 directory=Path(args.directory);report=json.loads((directory/'evaluation.json').read_text());runtime=json.loads((directory/'runtime-audit.json').read_text());manifest=json.loads((ROOT/'public/asset-manifest.json').read_text());protocol=json.loads((ROOT/'training/evaluation-protocol.json').read_text())
 errors=check(report,runtime,manifest,protocol)
 if errors:raise SystemExit('Not qualified:\n- '+'\n- '.join(errors))
 if args.write_runtime:
  data={'ready':True,'trainingSteps':report['modelSteps'],'policySHA256':report['policySHA256'],'description':f"Original PPO controller. {protocol['perSkillHeldOutCases']} fresh starts per dive, across its measured height range and physical perturbations. These native MuJoCo comparisons include every failure; the same cases are independently evaluated with browser inference and WASM.",'policies':{name:{'completionRate':stats['cleanCompletionRate'],'meanExecution':stats['meanExecution'],'meanAngle':stats['meanEntryAngle']}for name,stats in report['policies'].items()},'perSkill':{skill['id']:{'minHeight':report['minimumHeights'][index],'completionRate':runtime['perSkillHeldOut'][skill['id']]['cleanCompletionRate'],'meanExecution':runtime['perSkillHeldOut'][skill['id']]['meanExecution']}for index,skill in enumerate(SKILLS)}}
  (ROOT/'src/data/evaluation.js').write_text('// Generated only after frozen native and WASM qualification.\nexport const EVALUATION='+json.dumps(data,separators=(',',':'))+';\n')
 print('Qualified: every declared skill and exposed height band passed the frozen evidence checks.')
if __name__=='__main__':main()
