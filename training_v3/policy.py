"""Shared actor trunk with hybrid self-declaration / motor outputs.
No per-dive specialists. Conditional autoregressive Gaussian motor exploration
has an explicit, correct likelihood; no noise is filtered after sampling.
"""
import math
import torch
from torch import nn
from torch.distributions import Normal,Categorical
from rules import DIVES
FORMAT='self-declared-diver-v11'

class Policy(nn.Module):
 def __init__(self,obs,widths=(256,256),rho=.6):
  super().__init__();self.widths=tuple(widths);self.obs=obs;self.rho=rho
  layers=[];size=obs
  for width in widths:layers.extend([nn.Linear(size,width),nn.Tanh()]);size=width
  self.trunk=nn.Sequential(*layers);self.motor=nn.Linear(size,9);self.choice=nn.Linear(size,len(DIVES));self.value=nn.Linear(size,1)
  self.logstd=nn.Parameter(torch.full((9,),-.5));self.register_buffer('value_mean',torch.zeros(()));self.register_buffer('value_std',torch.ones(()))
  for layer in self.modules():
   if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,math.sqrt(2));nn.init.zeros_(layer.bias)
  for layer in [self.motor,self.choice]:nn.init.orthogonal_(layer.weight,.01)
 @staticmethod
 def log_jacobian(z):return 2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
 def forward(self,obs,mask,choosing,raw=None,choice=None,deterministic=False):
  h=self.trunk(obs);previous=obs[:,-9:].clamp(-.999,.999).atanh()
  mean=(1-self.rho)*2*torch.tanh(self.motor(h)/2)+self.rho*previous
  dist=Normal(mean,self.logstd.clamp(-2.8,0).exp()*math.sqrt(1-self.rho**2))
  if not mask.any(-1).all():raise ValueError('Empty legal-declaration set')
  selection=Categorical(logits=self.choice(h).masked_fill(~mask,-torch.inf))
  if choice is None:choice=selection.logits.argmax(-1) if deterministic else selection.sample()
  if raw is None:raw=mean if deterministic else dist.sample()
  motor_logp=(dist.log_prob(raw)-self.log_jacobian(raw)).sum(-1)
  logp=torch.where(choosing,selection.log_prob(choice),motor_logp)
  z=dist.rsample();motor_entropy=(dist.entropy()+self.log_jacobian(z)).sum(-1)
  entropy=torch.where(choosing,selection.entropy(),motor_entropy)
  normalized=self.value(h).squeeze(-1)
  return dict(action=raw.tanh(),choice=choice,raw=raw,logp=logp,entropy=entropy,value=normalized*self.value_std+self.value_mean,normalizedValue=normalized)
 @torch.no_grad()
 def update_value_scale(self,returns):
  # PopArt affine correction preserves all unnormalized predictions exactly.
  mean=.99*self.value_mean+.01*returns.mean()
  second=.99*(self.value_std.square()+self.value_mean.square())+.01*returns.square().mean()
  std=(second-mean.square()).clamp(min=.01).sqrt()
  ratio=self.value_std/std
  self.value.weight.mul_(ratio);self.value.bias.copy_((self.value_std*self.value.bias+self.value_mean-mean)/std)
  self.value_mean.copy_(mean);self.value_std.copy_(std)
  return float(ratio)
 def export(self):
  return dict(format=FORMAT,observationSize=self.obs,actionSize=9,choiceSize=len(DIVES),widths=self.widths,rho=self.rho,
    state={k:v.detach().cpu().tolist() for k,v in self.state_dict().items()})
