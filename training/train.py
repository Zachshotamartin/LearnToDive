"""Reward-based cross-entropy motor-policy search. No expert labels or analytic policy fitting."""
import argparse,json,time,pathlib
import numpy as np
from sim import contexts,features,decode,actions,baseline,simulate
parser=argparse.ArgumentParser();parser.add_argument('--generations',type=int,default=180);parser.add_argument('--population',type=int,default=96);parser.add_argument('--contexts',type=int,default=48);parser.add_argument('--seed',type=int,default=731);args=parser.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];rng=np.random.default_rng(args.seed)
mean=np.zeros((4,5));std=np.full((4,5),.65);std[:,0]=1.;std[0,1]=1.6
initial=mean.copy();validation=contexts(np.random.default_rng(17039),384);history=[];best=None;bestfitness=-1e9;start=time.perf_counter()
for generation in range(args.generations):
    population=mean+rng.normal(size=(args.population,4,5))*std
    population[0]=mean
    p=contexts(rng,args.contexts);f=features(p)
    raw=np.einsum('pij,cj->pci',population,f).reshape(-1,4);allp=np.tile(p,(args.population,1))
    result=simulate(allp,decode(raw));fitness=result['reward'].reshape(args.population,args.contexts).mean(axis=1)
    elite=population[np.argsort(fitness)[-max(8,args.population//8):]]
    mean=.15*mean+.85*elite.mean(axis=0);std=np.maximum(.02,.7*std+.3*elite.std(axis=0))
    if generation%5==0 or generation==args.generations-1:
        candidates=[mean,population[int(np.argmax(fitness))]]
        for candidate in candidates:
            val=simulate(validation,actions(candidate,validation));metric=float(val['reward'].mean())
            if metric>bestfitness:bestfitness=metric;best=candidate.copy()
        val=simulate(validation,actions(best,validation));row={'generation':generation,'episodes':(generation+1)*args.population*args.contexts,'validationScore':float(val['score'].mean()),'validationValidRate':float(val['valid'].mean()),'seconds':round(time.perf_counter()-start,2)};history.append(row);print(json.dumps(row),flush=True)
        checkpoint={'method':'Cross-entropy policy search','seed':args.seed,'features':['bias','inverse flight-time estimate','squared inverse flight-time estimate','observed takeoff lean','launch velocity'],'weights':best.tolist(),'initialWeights':initial.tolist(),'trainingEpisodes':row['episodes'],'history':history,'ranges':{'height':[2,12],'spring':[1.5,3.5],'wind':[-.6,.6],'tilt':[-.09,.09]}}
        (root/'src/data/model.js').write_text('export const MODEL = '+json.dumps(checkpoint,separators=(',',':'))+';\n')
        (root/'training/checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
test=contexts(np.random.default_rng(990517),2000);report={'seed':990517,'episodes':len(test),'trainingSeed':args.seed,'trainingEpisodes':args.generations*args.population*args.contexts,'trainingSeconds':round(time.perf_counter()-start,2),'evaluations':{}}
for name,w in [('pretrained',best),('untrained',initial),('baseline',None)]:
    result=simulate(test,baseline(test) if w is None else actions(w,test));bins=[]
    for lo in [2,4,6,8,10]:
        mask=(test[:,0]>=lo)&(test[:,0]<lo+2);bins.append({'height':[lo,lo+2],'n':int(mask.sum()),'validRate':float(result['valid'][mask].mean()),'meanScore':float(result['score'][mask].mean())})
    report['evaluations'][name]={'validRate':float(result['valid'].mean()),'meanScore':float(result['score'].mean()),'meanAngle':float(result['angle'].mean()),'meanFlips':float(result['flips'].mean()),'meanTwists':float(result['twists'].mean()),'meanSplash':float(result['splash'].mean()),'heightBins':bins}
(root/'training/evaluation.json').write_text(json.dumps(report,indent=2));(root/'src/data/evaluation.js').write_text('export const EVALUATION = '+json.dumps(report,separators=(',',':'))+';\n');print(json.dumps(report),flush=True)
fixture=[]
for p in [[5,2.4,0,0],[2.2,1.6,.4,-.07],[11.8,3.4,-.5,.065]]:
    pa=np.asarray([p]);a=actions(best,pa);r=simulate(pa,a,trace=True);fixture.append({'parameters':dict(zip(['height','spring','wind','tilt'],p)),'actions':dict(zip(['spin','twist','tuck','openAt'],a[0].tolist())),'trace':r['trace'],'score':float(r['score'][0])})
(root/'tests/parity-fixtures.json').write_text(json.dumps(fixture))
