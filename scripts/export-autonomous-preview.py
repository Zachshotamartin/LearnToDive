"""Freeze a selected diving checkpoint (v11 or v13 layout) and independent native/browser parity cases.
Only reads training output. The preview retains its unqualified development status.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training_v3'))
from environment import Arena
from evaluation import matching_arena
from policy import Policy
from rules import DIVES,CODES

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def digest(b):return hashlib.sha256(b).hexdigest()
a=argparse.ArgumentParser();a.add_argument('checkpoint');a.add_argument('evaluation');args=a.parse_args()
torch.set_num_threads(1);saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
print('checkpoint keys',list(saved))
config=saved['config']
def build(observation_size,initial_action=None):
 return Policy(observation_size,config['widths'],rho=config['rho'],initial_action=initial_action,exploration=config.get('exploration','diagonal'),
  architecture=config.get('architecture','shared'),noise_rho=config.get('noise_rho',0.),normalize_inputs=config.get('input_normalization',False))
p=build(saved['contract']['observationSize']);p.load_state_dict(saved['model']);p.eval()
# The parity arena has the policy's observation layout and the full declaration scope.
e=matching_arena(p,1,109310,reward_mode=config.get('reward_mode','v12'))
torch.manual_seed(109310)
initial=build(e.observation_size,None if config.get('random_motor_init') else e.initial_action)
tag=p.format.rsplit('-',1)[-1]
model=dict(**p.export(),contract=saved['contract'],steps=saved['training']['steps'],qualified=False)
files={}
for key,obj in [('policy',model),('initial',dict(**initial.export(),contract=saved['contract'],steps=0,qualified=False))]:
 raw=(json.dumps(obj,separators=(',',':'))+'\n').encode();name=f'models/{key}-{tag}-{digest(raw)[:16]}.json';(ROOT/'public'/name).write_bytes(raw);files[key]=name
for old in list((ROOT/'public/models').glob('*-v11-*.json'))+list((ROOT/'public/models').glob('*-v13-*.json')):
 if str(old.relative_to(ROOT/'public')) not in files.values():old.unlink()
raw=(ROOT/'training_v3/diver.xml').read_bytes();files['xml']=f'physics/diver-{tag}-{digest(raw)[:16]}.xml';(ROOT/'public'/files['xml']).write_bytes(raw)
(ROOT/'src/data/autonomousAssets.js').write_text('export const ASSETS={'+','.join(k+':new URL('+json.dumps('../../public/'+v)+',import.meta.url).href'for k,v in files.items())+'};\n')
(ROOT/'src/data/declarations.js').write_text('export const DIVES='+json.dumps(DIVES,separators=(',',':'))+';\nexport const CODES='+json.dumps(CODES)+';\n')
report=json.loads(Path(args.evaluation).read_text());write(ROOT/'public/autonomous-evaluation.json',report)
write(ROOT/'public/autonomous-manifest.json',dict(format=model['format'],status='DEVELOPMENT',qualified=False,steps=model['steps'],checkpointSHA256=digest(Path(args.checkpoint).read_bytes()),metrics=report['summary']['full'],files=[dict(key=k,path=v,bytes=(ROOT/'public'/v).stat().st_size,sha256=digest((ROOT/'public'/v).read_bytes()))for k,v in files.items()]))
(ROOT/'src/data/autonomousEvaluation.js').write_text('export const EVALUATION='+json.dumps(dict(steps=model['steps'],metrics=report['summary']['full']))+';\n')
cases=[]
for group,apparatus,height in [(g,'platform',7.5) for g in range(1,7)]+[(1,'springboard',1),(2,'springboard',3),(1,'platform',5),(6,'platform',10)]:
 e.reset([0],new_routine=True);e.apparatus[0]=int(apparatus=='platform');e.height[0]=height;e.schedule[0]=[group]*6;e.group[0]=group;e.used[0]=False;e.round[0]=0;e.choosing[0]=True;e.previous_actions[0]=0;e.stand(0,0)
 o=e.observe();mask=e.mask()
 with torch.no_grad():r=p(torch.tensor(o),torch.tensor(mask),torch.tensor(e.choosing),deterministic=True)
 choice=int(r['choice'][0]);first=o[0].tolist();e.declare(0,choice);rows=[]
 for t in range(240):
  o=e.observe()
  with torch.no_grad():r=p(torch.tensor(o),torch.tensor(e.mask()),torch.tensor(e.choosing),deterministic=True)
  action=r['action'][0].numpy();_,_,done,info=e.physics.step(action[None],auto_reset=False);e.previous_actions[0]=action;e.previous_noise[0]=0
  rows.append(dict(observation=o[0].tolist(),action=action.tolist(),state=e.physics.state[0].tolist()))
  if done[0]:
   from judge import judge
   m=info[0];m['positionQuality']=m['positionQualities']['ABCD'.index(DIVES[choice]['position'])];score=judge(choice,apparatus,height,m);break
 cases.append(dict(parameters=dict(category=group,height=height,apparatus=apparatus,used=[],round=0),observation=first,choice=choice,rows=rows,result=score))
write(ROOT/'tests/fixtures/autonomous-native.json',dict(cases=cases,sourceHashes=saved['contract']['sourceHashes']))
e.close();print('Exported',model['steps'],'steps and',len(cases),'native cases')
