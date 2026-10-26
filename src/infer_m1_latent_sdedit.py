#!/usr/bin/env python3
"""M1-native SDEdit range translation using the adapted latent diffusion prior."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL,DDIMScheduler,UNet2DModel

def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed-data',type=Path,required=True); p.add_argument('--corpus',type=Path,required=True); p.add_argument('--vae',type=Path,required=True); p.add_argument('--unet',type=Path,required=True); p.add_argument('--scheduler',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--strength',type=float,required=True); p.add_argument('--steps',type=int,default=50); p.add_argument('--seed',type=int,default=20260730); c=p.parse_args()
 dev=torch.device('cuda'); torch.manual_seed(c.seed); vae=AutoencoderKL.from_pretrained(c.vae).to(dev).eval(); unet=UNet2DModel.from_pretrained(c.unet).to(dev).eval(); sch=DDIMScheduler.from_pretrained(c.scheduler); sch.set_timesteps(c.steps,device=dev); angles=np.load(c.corpus/'ring_angles.npy').astype(np.float32); out=c.output/'synthetic'; out.mkdir(parents=True,exist_ok=True); gen=torch.Generator(device=dev).manual_seed(c.seed); counts=[]
 with torch.inference_mode(),torch.cuda.amp.autocast(dtype=torch.float16):
  for i,path in enumerate(sorted(c.seed_data.glob('*.npz'))):
   with np.load(path) as d: image=np.stack((d['log_range'].astype(np.float32)*2-1,d['mask'].astype(np.float32)))
   image=torch.from_numpy(image)[None].to(dev); image=F.interpolate(image,(64,256),mode='bilinear',align_corners=False); latent=vae.encode(image).latent_dist.mode()*vae.config.scaling_factor; n=max(1,int(round(c.steps*c.strength))); ts=sch.timesteps[c.steps-n:]; x=sch.add_noise(latent,torch.randn(latent.shape,device=dev,generator=gen,dtype=latent.dtype),ts[0].expand(1))
   for t in ts: x=sch.step(unet(x,t).sample,t,x).prev_sample
   decoded=F.interpolate(vae.decode(x/vae.config.scaling_factor).sample.float(),(64,1024),mode='bilinear',align_corners=False)[0]; depth=torch.clamp((decoded[0]+1)/2,0,1).cpu().numpy(); mask=torch.sigmoid(decoded[1]).cpu().numpy()>=.5; r=(2**(depth*8)-1)*mask; az=((np.arange(1024)+.5)/1024*2*np.pi-np.pi)[None]; el=angles[:,None]; xyz=np.stack((r*np.cos(el)*np.cos(az),r*np.cos(el)*np.sin(az),np.broadcast_to(r*np.sin(el),r.shape)),-1); pts=xyz[mask].astype(np.float32); pts.tofile(out/f'{i+1:04d}.bin'); counts.append(len(pts))
 print({'frames':len(counts),'median_points':int(np.median(counts)),'total_points':int(sum(counts))})
if __name__=='__main__': main()
