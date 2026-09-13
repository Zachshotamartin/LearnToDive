exec(open('/tmp/diving-diagnosis.py').read().split('for label,path in paths.items():')[0])
from rules import IDS
base=json.loads((out/'v13_best-mean.json').read_text())
queues={i:[] for i in range(24)}
for r in base['episodes']:queues[r['measurements']['index']].append(IDS[r['declaration']])
for label in paths:
 saved=json.loads(paths[label].read_text());p=Policy(saved['observationSize'],saved['widths'],saved['rho']);p.load_state_dict({k:torch.tensor(v) for k,v in saved['state'].items()});p.eval().requires_grad_(False)
 counts=np.zeros(24,int);actions=[];joint=[];cats=[];timing=[]
 def wrapped(obs,mask,choosing,**kw):
  result=p(obs,mask,choosing,**kw)
  for i in np.flatnonzero(choosing.numpy()):
   if counts[i]<6:result['choice'][i]=queues[i][counts[i]]
   counts[i]+=1
  valid=(counts<=6)&~choosing.numpy()&(obs[:,57].numpy()>0.5)&(obs[:,60].numpy()==0)
  if valid.any():
   actions.extend(result['action'][valid].tolist());joint.extend((obs[valid,10:24]*3).tolist());cats.extend((obs[valid,68:74].argmax(-1)+1).tolist());timing.extend(obs[valid,56].tolist())
  return result
 report=evaluate(wrapped,seed=771100,cases=24,reward_mode='continuous-entry',deterministic=True)
 a=np.array(actions);q=np.array(joint);g=np.array(cats)
 stats={}
 for c in range(1,7):
  qs=q[g==c];ac=a[g==c]
  stats[str(c)]={'frames':len(qs),'jointMean':qs.mean(0).tolist(),'jointRange':np.ptp(qs,axis=0).tolist(),'tuckFraction':float(np.mean((qs[:,0]>1)&(qs[:,1]>1.3))),'armsOverheadFraction':float(np.mean(np.all(qs[:,[6,9]]>2.4,axis=1))),'saturatedActionFraction':np.mean(abs(ac)>.9,axis=0).tolist()}
 report['motion']=stats;report['fixedDeclarationsFrom']='v13_best-mean';(out/f'{label}-fixed.json').write_text(json.dumps(report,indent=2));print(label,report['summary']['full'],flush=True)
