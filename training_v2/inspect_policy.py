"""Quick all-skill checkpoint diagnostic. Release benchmarks use evaluate.py instead."""
import argparse,json
import numpy as np
import torch
from sim import Arena,SKILLS
from train import ActorCritic
from evaluate import score,summary
from contract import validate_contract

def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--n',type=int,default=256);p.add_argument('--skills',nargs='+',type=int,default=list(range(6)));p.add_argument('--height',type=float);p.add_argument('--trace',action='store_true');args=p.parse_args()
 checkpoint=torch.load(args.checkpoint,weights_only=False,map_location='cpu');validate_contract(checkpoint.get('contract'));agent=ActorCritic(phase_exploration=checkpoint['model']['logstd'].ndim==3);agent.load_state_dict(checkpoint['model'])
 env=Arena(args.n,seed=808,training=False,skills=args.skills);active=np.ones(args.n,bool);rows=[];trace=[]
 if args.height is not None:env.reset(np.arange(args.n),[{'skill':int(s),'height':args.height}for s in env.skill])
 try:
  for i in range(231):
   with torch.no_grad():action=(agent.mean_action(torch.from_numpy(env.observe())) if checkpoint.get("actionDistribution")=="tanh-squashed-gaussian-v1" else agent.actor(torch.from_numpy(env.observe()))).numpy()
   _,_,done,info=env.step(action,auto_reset=False)
   if args.trace and i%10==0:trace.append({'time':float(env.state[0,0]),'action':action[0].tolist(),'joint':env.state[0,1:21][env.qadr].tolist(),'theta':float(env.theta[0])})
   for row in info:
    if active[row['index']]:active[row['index']]=False;rows.append({**row,**score(row)})
   if not active.any():break
  print(json.dumps({'contract':checkpoint['contract'],'checkpointSteps':checkpoint['steps'],'kind':'development diagnostic, not held-out release qualification','perSkill':{SKILLS[s]['id']:summary(group)for s in args.skills if(group:=[r for r in rows if r['skill']==s])},'trace':trace},indent=2))
 finally:env.close()
if __name__=='__main__':main()
