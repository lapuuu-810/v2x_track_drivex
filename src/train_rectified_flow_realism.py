#!/usr/bin/env python3
"""Point-space rectified-flow generator for V2X-Real aggregate LiDAR realism."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0,str(Path(__file__).resolve().parent))
from generate_realism_baselines import PC_RANGE,collect_real_frames,pooled_points

class Velocity(nn.Module):
 def __init__(self,width=192):
  super().__init__(); self.net=nn.Sequential(nn.Linear(7,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,3))
 def forward(self,x,t):
  phase=torch.cat([t,torch.sin(2*torch.pi*t),torch.cos(2*torch.pi*t),t*t],1)
  return self.net(torch.cat([x,phase],1))

def parse():
 p=argparse.ArgumentParser(); p.add_argument('--real-root',type=Path,nargs='+',required=True); p.add_argument('--forbidden',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
 p.add_argument('--steps',type=int,default=8000); p.add_argument('--batch-size',type=int,default=4096); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--device',default='cpu'); p.add_argument('--frames',type=int,default=50); p.add_argument('--points-per-frame',type=int,default=2000); p.add_argument('--ode-steps',type=int,default=64); p.add_argument('--seed',type=int,default=20260724); return p.parse_args()

def main():
 c=parse(); torch.manual_seed(c.seed); np.random.seed(c.seed); device=torch.device(c.device)
 fit,val=[],[]
 for root in c.real_root:
  current_fit,current_val=collect_real_frames(root,c.forbidden); fit.extend(current_fit); val.extend(current_val)
 target=pooled_points(fit,640,12000,c.seed); mean=target.mean(0); cov=np.cov(target,rowvar=False); chol=np.linalg.cholesky(cov+np.eye(3)*1e-4).astype(np.float32)
 target_t=torch.from_numpy(target); mean_t=torch.from_numpy(mean.astype(np.float32)); chol_t=torch.from_numpy(chol); model=Velocity().to(device); opt=torch.optim.AdamW(model.parameters(),lr=c.lr,weight_decay=1e-5); history=[]
 for step in range(1,c.steps+1):
  idx=torch.randint(len(target_t),(c.batch_size,)); y=target_t[idx].to(device); z=torch.randn(c.batch_size,3,device=device)@chol_t.to(device).T+mean_t.to(device)
  # Sliced minibatch matching reduces crossing paths and stabilizes 3-D flow.
  direction=torch.randn(3,device=device); direction/=direction.norm(); z=z[torch.argsort(z@direction)]; y=y[torch.argsort(y@direction)]
  t=torch.rand(c.batch_size,1,device=device); x=(1-t)*z+t*y; goal=y-z; pred=model(x,t); loss=(pred-goal).square().mean()
  opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
  if step==1 or step%200==0:
   history.append({'step':step,'loss':float(loss)}); print(json.dumps(history[-1]),flush=True)
 c.output.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'mean':mean,'chol':chol,'args':vars(c)},c.output/'flow.pt')
 n=c.frames*c.points_per_frame; g=torch.Generator(device=device).manual_seed(c.seed+1); x=torch.randn(n,3,generator=g,device=device)@chol_t.to(device).T+mean_t.to(device)
 model.eval(); dt=1/c.ode_steps
 with torch.inference_mode():
  for i in range(c.ode_steps):
   t=torch.full((n,1),(i+.5)*dt,device=device); x=x+dt*model(x,t)
 points=x.cpu().numpy(); points=np.clip(points,PC_RANGE[:3],PC_RANGE[3:]).astype(np.float32); out=c.output/'realism'; out.mkdir(exist_ok=True)
 rng=np.random.default_rng(c.seed+2); rng.shuffle(points)
 for i,frame in enumerate(points.reshape(c.frames,c.points_per_frame,3),1): frame.tofile(out/f'{i:04d}.bin')
 manifest={'method':'point-space rectified flow with sliced minibatch coupling','fit_frames':len(fit),'validation_paths':[str(x) for x in val],'frames':c.frames,'points_per_frame':c.points_per_frame,'steps':c.steps,'ode_steps':c.ode_steps,'history':history}; (c.output/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
if __name__=='__main__': main()
