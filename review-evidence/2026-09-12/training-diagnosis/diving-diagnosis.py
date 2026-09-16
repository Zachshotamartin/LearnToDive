import sys,json,pathlib,torch,hashlib,numpy as np,collections
root=pathlib.Path('/Users/zacharymartin/Desktop/GraphicsExperiments/LearnToDive')
sys.path.insert(0,str(root/'training_v3'))
from policy import Policy
from evaluation import evaluate
from checkpointing import hashes
out=root/'review-evidence/2026-09-12/training-diagnosis'
torch.set_num_threads(1)
paths={
 'initial':root/'public/models/initial-v11-91764a7d56adfb63.json',
 'production':root/'public/models/policy-v11-320b4c71e7fdaef6.json',
 'v13_best':pathlib.Path('/Users/zacharymartin/Desktop/Portfolio-about-skills/output/model-preview-refresh-2026-09-12/diver-model.json')}
for label,path in paths.items():
 raw=path.read_bytes();saved=json.loads(raw)
 p=Policy(saved['observationSize'],saved['widths'],saved['rho'])
 p.load_state_dict({k:torch.tensor(v) for k,v in saved['state'].items()});p.eval().requires_grad_(False)
 for sampled in [False,True]:
  report=evaluate(p,seed=771100,cases=24,reward_mode='continuous-entry',deterministic=not sampled)
  report.update(label=label,assetSHA256=hashlib.sha256(raw).hexdigest(),sourceHashes=hashes(),logstd=p.logstd.tolist())
  rows=report['episodes']
  report['diagnosis']={'failures':dict(collections.Counter(x for r in rows for x in r['failures'])),
   'entryArmsValid':np.mean([r['measurements']['entryArmPositionValid'] for r in rows]),
   'twistMagnitude':np.mean([abs(r['measurements']['twist']) for r in rows]),
   'somersaultMagnitude':np.mean([abs(r['measurements']['rotation']) for r in rows]),
   'declarations':dict(collections.Counter(r['declaration'] for r in rows)),
   'meanTrainingCosts':{k:np.mean([r['trainingRewardComponents']['deductions'][k] for r in rows]) for k in rows[0]['trainingRewardComponents']['deductions']}}
  file=out/f'{label}-{"sampled" if sampled else "mean"}.json';file.write_text(json.dumps(report,indent=2))
  print(json.dumps({'label':label,'sampled':sampled,'summary':report['summary'],'diagnosis':report['diagnosis'],'logstd':report['logstd']}),flush=True)
