"""Freeze one atomic optimizer checkpoint and derive its exact runtime JSON.
Reading checkpoint bytes once avoids policy/PT races during ongoing training.
"""
import argparse,hashlib,io,json
from pathlib import Path
import torch
from train import ActorCritic
from contract import validate_contract

def freeze(checkpoint,directory):
 payload=Path(checkpoint).read_bytes();data=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
 validate_contract(data.get('contract'))
 out=Path(directory);out.mkdir(parents=True,exist_ok=False)
 (out/'checkpoint.pt').write_bytes(payload)
 agent=ActorCritic(phase_exploration=data['model']['logstd'].ndim==3);agent.load_state_dict(data['model']);agent.save_json(out/'policy.json',data['steps'],data['config'],data['contract'])
 model=json.loads((out/'policy.json').read_bytes())
 for layer in model['layers']:
  layer['weight']=[[round(float(x),8)for x in row]for row in layer['weight']]
  layer['bias']=[round(float(x),8)for x in layer['bias']]
 runtime=json.dumps(model,separators=(',',':')).encode();(out/'policy.json').write_bytes(runtime)
 provenance={'checkpointSHA256':hashlib.sha256(payload).hexdigest(),'policySHA256':hashlib.sha256(runtime).hexdigest(),'steps':data['steps'],'contract':data['contract'],'provenance':data['provenance']}
 (out/'freeze.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps({k:v for k,v in provenance.items()if k not in ['contract','provenance']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('directory');a=p.parse_args();freeze(a.checkpoint,a.directory)
