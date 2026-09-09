"""Re-evaluate frozen coefficients without training. JSON on stdout; does not change assets."""
import json,pathlib
import numpy as np
from sim import contexts,actions,baseline,simulate
root=pathlib.Path(__file__).resolve().parents[1]
model=json.loads((root/'training/checkpoint.json').read_text())
p=contexts(np.random.default_rng(990517),2000)
report={'seed':990517,'episodes':2000,'policies':{}}
for name,w in [('pretrained',model['weights']),('untrained',model['initialWeights']),('baseline',None)]:
    r=simulate(p,baseline(p) if w is None else actions(np.asarray(w),p))
    proportion=float(r['valid'].mean());n=len(p);z=1.959963984540054;denom=1+z*z/n
    center=(proportion+z*z/(2*n))/denom
    margin=z*np.sqrt(proportion*(1-proportion)/n+z*z/(4*n*n))/denom
    report['policies'][name]={'validRate':proportion,'validRate95Wilson':[center-margin,center+margin],'meanScore':float(r['score'].mean()),'meanScoreSE':float(r['score'].std(ddof=1)/np.sqrt(n)),'meanEntryAngle':float(r['angle'].mean())}
grid=np.asarray([[h,s,w,t] for h in [2,3,5,8,10,12] for s in [1.5,2.5,3.5] for w in [-.6,0,.6] for t in [-.09,0,.09]])
r=simulate(grid,actions(np.asarray(model['weights']),grid))
report['boundaryGrid']={'episodes':len(grid),'valid':int(r['valid'].sum()),'worstEntryAngle':float(r['angle'].max())}
print(json.dumps(report,indent=2))
