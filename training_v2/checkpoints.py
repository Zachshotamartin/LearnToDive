"""Durable update-boundary checkpoints and evidence-based development selection.
Snapshots are immutable. `latest.json` and `best.json` are atomic pointers, not
claims that an arbitrary unsaved instant or deployment-qualified model exists.
"""
from __future__ import annotations
import copy,datetime,hashlib,json,os,random,shutil,tempfile,time
from pathlib import Path
import numpy as np
import torch

FORMAT='dive-training-state-v2'
SCRATCH={'roll_state','roll_sensor','ctrl','qadr','vadr'}
ENV_SETTINGS=['n','minimum_heights','boundary_rate','execution_weight','ascent_weight','launch_speed_weight','entry_geometry_weight','skill_ids','heights','training','entry_curriculum']

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc_now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic_json(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.'+path.name,delete=False) as handle:
  json.dump(data,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno());temporary=handle.name
 os.replace(temporary,path)

def capture_environment(env):
 # Rollout receives explicit initial state, sensor/target sidecars and controls
 # every call. Its scratch outputs and worker MjData objects do not carry state
 # between transitions. Tests compare continuation in a fresh Arena.
 return {'entryBank':copy.deepcopy(env.entry_bank),'settings':{k:copy.deepcopy(getattr(env,k))for k in ENV_SETTINGS},
         'arrays':{k:v.copy()for k,v in vars(env).items()if isinstance(v,np.ndarray)and k not in SCRATCH},
         'rng':copy.deepcopy(env.rng.bit_generator.state)}

def restore_environment(env,snapshot):
 env.entry_bank=copy.deepcopy(snapshot.get('entryBank',[[] for _ in range(6)]))
 for key,value in snapshot['settings'].items():
  if json.dumps(getattr(env,key))!=json.dumps(value):raise ValueError(f'Exact resume environment differs: {key}')
 expected={k for k,v in vars(env).items()if isinstance(v,np.ndarray)and k not in SCRATCH}
 if expected!=set(snapshot['arrays']):raise ValueError('Exact resume environment state fields differ')
 for key,value in snapshot['arrays'].items():
  target=getattr(env,key)
  if target.shape!=value.shape or target.dtype!=value.dtype:raise ValueError(f'Exact resume array differs: {key}')
  target[:]=value
 env.rng.bit_generator.state=copy.deepcopy(snapshot['rng'])

def capture_rng():
 return {'torch':torch.get_rng_state().clone(),'numpy':copy.deepcopy(np.random.get_state()),'python':random.getstate()}
def restore_rng(state):
 torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])

def checkpoint_payload(agent,optimizer,env,contract,config,provenance,steps,training_state):
 return {'checkpointFormat':FORMAT,'contract':copy.deepcopy(contract),'actionDistribution':'tanh-squashed-gaussian-v1',
         'model':copy.deepcopy(agent.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'steps':steps,
         'config':copy.deepcopy(config),'provenance':copy.deepcopy(provenance),'rng':capture_rng(),
         'environment':capture_environment(env),'trainingState':copy.deepcopy(training_state)}

def validate_exact(checkpoint,source_hashes):
 if checkpoint.get('checkpointFormat')!=FORMAT:raise ValueError('Legacy checkpoint lacks complete environment/RNG state. Use --resume-mode restart and disclose the reset.')
 if checkpoint['provenance']['sourceHashes']!=source_hashes:raise ValueError('Exact resume requires the identical frozen training sources; use a named restart for changed code.')
 for key in ['optimizer','rng','environment','trainingState']:
  if key not in checkpoint:raise ValueError(f'Exact resume missing {key}')

def save_checkpoint(run,agent,optimizer,env,contract,config,provenance,steps,training_state,reason='periodic'):
 run=Path(run);directory=run/'checkpoints';directory.mkdir(parents=True,exist_ok=True)
 name=f"step-{steps:012d}-update-{training_state['updatesCompleted']:08d}"
 destination=directory/name
 if destination.exists():
  # A stop/final event at an already saved boundary updates only the pointer.
  manifest=json.loads((destination/'manifest.json').read_text())
  atomic_json(run/'latest.json',{'snapshot':str(destination.relative_to(run)),**manifest});return destination
 temporary=Path(tempfile.mkdtemp(prefix='.saving-',dir=directory))
 try:
  payload=checkpoint_payload(agent,optimizer,env,contract,config,provenance,steps,training_state)
  with (temporary/'checkpoint.pt').open('wb') as handle:torch.save(payload,handle);handle.flush();os.fsync(handle.fileno())
  agent.save_json(temporary/'policy.json',steps,config,copy.deepcopy(contract))
  with (temporary/'policy.json').open('rb')as handle:os.fsync(handle.fileno())
  manifest={'format':FORMAT,'savedUTC':utc_now(),'reason':reason,'steps':steps,'actorUpdateInteractions':contract.get('actorUpdateInteractions',0),
            'criticCalibrationInteractions':contract.get('criticCalibrationInteractions',0),'updatesCompleted':training_state['updatesCompleted'],
            'segmentElapsedSeconds':training_state['segmentElapsedSeconds'],'cumulativeElapsedSeconds':training_state['cumulativeElapsedSeconds'],
            'checkpointSHA256':sha256(temporary/'checkpoint.pt'),'policySHA256':sha256(temporary/'policy.json')}
  atomic_json(temporary/'manifest.json',manifest)
  os.rename(temporary,destination)
  # latest.pt remains a backwards-compatible atomic convenience copy. The
  # immutable directory + latest.json are authoritative and bind both files.
  for source,target in [('checkpoint.pt','latest.pt'),('policy.json','policy.json')]:
   tmp=run/('.'+target+'.tmp');shutil.copyfile(destination/source,tmp);os.replace(tmp,run/target)
  atomic_json(run/'latest.json',{'snapshot':str(destination.relative_to(run)),**manifest})
  return destination
 except BaseException:
  shutil.rmtree(temporary,ignore_errors=True);raise

def requested_interactions(additional,target,current,batch_size):
 if additional is not None and target is not None:raise ValueError('Choose additional steps or cumulative target steps, not both')
 count=(target-current)if target is not None else(additional if additional is not None else 10000000)
 if count<=0:raise ValueError('Target is already reached; choose a larger --target-steps or positive --additional-steps')
 # PPO resumes at complete update boundaries, never an invented partial batch.
 return int(np.ceil(count/batch_size))*batch_size

def register_evaluation(run,directory,checkpoint_path=None):
 """Rank only identical fixed-case, same-judge evaluations. All failures count.
 This chooses a development best; release qualification is a separate process.
 """
 run=Path(run);directory=Path(directory).resolve();report=json.loads((directory/'evaluation.json').read_text());cases=json.loads((directory/'cases.json').read_text())
 policy=directory/'policy.json';checkpoint=Path(checkpoint_path or directory/'checkpoint.pt').resolve()
 if sha256(policy)!=report['policySHA256']:raise ValueError('Evaluation does not match policy bytes')
 if not checkpoint.is_file():raise ValueError('Evaluated best requires a resumable native checkpoint')
 c=torch.load(checkpoint,weights_only=False,map_location='cpu')
 if c['steps']!=report['modelSteps'] or c['contract']['xmlSHA256']!=report['physicsSHA256']:raise ValueError('Evaluation/checkpoint counters or physics differ')
 # Check every exported coefficient against the native checkpoint, not merely
 # the step counter (two actors can share a counter).
 p=json.loads(policy.read_text())
 for layer,index in zip(p['layers'],[0,2,4]):
  for field,key in [('weight','weight'),('bias','bias')]:
   expected=c['model'][f'actor.{index}.{key}'].numpy().round(8)
   expected=expected.astype(np.float64)
   # freeze_policy additionally trims the already float32-rounded JSON
   # coefficients to eight decimal places. Accept these two exact encodings.
   trimmed=np.vectorize(lambda x:round(float(x),8))(expected)
   if not (np.array_equal(np.asarray(layer[field]),expected)or np.array_equal(np.asarray(layer[field]),trimmed)):raise ValueError('Evaluated policy is not the supplied checkpoint actor')
 rows=cases['pretrained'];case_hash=hashlib.sha256(json.dumps([r['parameters']for r in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 metrics=report['policies']['pretrained'];per_skill=report['perSkill']['pretrained']
 identity={'physicsSHA256':report['physicsSHA256'],'judge':report['judge'],'sourceSHA256':report['sourceSHA256'],'caseSHA256':case_hash,'episodes':len(rows)}
 rank=[min(v['cleanCompletionRate']for v in per_skill.values()),metrics['cleanCompletionRate'],metrics['meanExecution'],-metrics['meanEntryAngle'],min(v['meanExecution']for v in per_skill.values())]
 evidence={'status':'evaluated development best; not deployment qualification','identity':identity,'rank':rank,'policySHA256':sha256(policy),'checkpointSHA256':sha256(checkpoint),'checkpoint':str(checkpoint),'policy':str(policy),'evaluation':str(directory/'evaluation.json'),'evaluationSHA256':sha256(directory/'evaluation.json'),'casesSHA256':sha256(directory/'cases.json'),'steps':report['modelSteps'],'meanExecution':metrics['meanExecution'],'worstAngleMean':metrics['meanEntryAngle'],'perSkillGates':{k:{'cleanRate':v['cleanCompletionRate'],'meanExecution':v['meanExecution'],'cleanAtLeast95Percent':v['cleanCompletionRate']>=.95,'meanExecutionAtLeastEight':v['meanExecution']>=8}for k,v in per_skill.items()},'registeredUTC':utc_now()}
 history=run/'evaluations';history.mkdir(parents=True,exist_ok=True);record=history/(evidence['policySHA256']+'.json')
 if not record.exists():atomic_json(record,evidence)
 best_path=run/'best.json'
 if best_path.exists():
  previous=json.loads(best_path.read_text())
  if previous['identity']!=identity:raise ValueError('Best comparison requires the identical evaluator, judge, physics and ordered cases')
  if rank<=previous['rank']:return False
 atomic_json(best_path,evidence);return True
