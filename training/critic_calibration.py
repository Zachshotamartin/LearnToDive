"""Actor-frozen Monte Carlo value calibration on fresh training interactions.
No motion demonstrations, action labels or future-state inputs are supplied.
Validation rollouts are distinct from the training batches and never optimized.
"""
import hashlib,json
import numpy as np
import torch

def actor_hash(agent):
 h=hashlib.sha256()
 for p in list(agent.actor.parameters())+[agent.logstd]:h.update(p.detach().numpy().tobytes())
 return h.hexdigest()

def collect(env,agent,horizon=240):
 env.reset(np.arange(env.n))
 observations=[];rewards=[];dones=[];skills=[];episodes=[]
 for _ in range(horizon):
  obs=torch.from_numpy(env.observe());observations.append(obs);skills.append(env.skill.copy())
  with torch.no_grad():a,*_=agent.get(obs)
  _,r,d,info=env.step(a.numpy());rewards.append(r);dones.append(d);episodes.extend(info)
 rewards=np.asarray(rewards);dones=np.asarray(dones);returns=np.zeros_like(rewards);mask=np.zeros_like(dones);seen=np.zeros(env.n,bool);value=np.zeros(env.n)
 # Only transitions with a real subsequent terminal boundary get MC targets.
 # The final incomplete tails are excluded, rather than bootstrapped by old V7 values.
 for t in reversed(range(horizon)):
  value=rewards[t]+.995*value*(~dones[t]);seen|=dones[t];returns[t]=value;mask[t]=seen
 obs=torch.stack(observations).flatten(0,1);target=torch.from_numpy(returns.flatten());valid=torch.from_numpy(mask.flatten())
 return obs[valid],target[valid],np.asarray(skills).reshape(-1)[mask.reshape(-1)],episodes

def metrics(agent,data):
 obs,target,skills,_=data
 with torch.no_grad():prediction=agent.critic(obs).flatten();error=(prediction-target).numpy()
 def group(ids):
  e=error[ids];t=target.numpy()[ids]
  return dict(samples=int(len(e)),mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e*e))),bias=float(np.mean(e)),targetMean=float(np.mean(t)))
 return dict(all=group(np.ones(len(error),bool)),perSkill={str(s):group(skills==s)for s in range(6)if np.any(skills==s)})

def calibrate(env,agent,rounds,lr,epochs,path):
 before_hash=actor_hash(agent);opt=torch.optim.Adam(agent.critic.parameters(),lr=lr,eps=1e-5)
 validation=collect(env,agent);before=metrics(agent,validation);history=[];steps=env.n*240;retained=0
 for i in range(rounds):
  batch=collect(env,agent);steps+=env.n*240;obs,target,_,_=batch;retained+=len(obs)
  for _ in range(epochs):
   for ids in torch.randperm(len(obs)).split(1024):
    prediction=agent.critic(obs[ids]).flatten();loss=((prediction-target[ids])**2).mean()
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(agent.critic.parameters(),1);opt.step()
  history.append(dict(round=i+1,training=metrics(agent,batch),validation=metrics(agent,validation)))
  print('CRITIC_CALIBRATION',json.dumps(history[-1]),flush=True)
 after_hash=actor_hash(agent)
 if after_hash!=before_hash:raise RuntimeError('Actor or action distribution changed during critic-only calibration')
 report=dict(kind='actor-frozen value calibration on fresh varied training contexts; not qualification',discount=.995,bootstrap=False,heldoutIncompleteTailsExcluded=True,actorBeforeSHA256=before_hash,actorAfterSHA256=after_hash,trainingRounds=rounds,lr=lr,epochs=epochs,interactions=steps,trainingInteractions=rounds*env.n*240,retainedMCSamples=retained,validationInteractions=env.n*240,before=before,after=metrics(agent,validation),history=history)
 path.write_text(json.dumps(report,indent=2));return report
