#!/usr/bin/env python3
"""Semantic-guided synthetic-to-real LiDAR translation with SG-LDM-V2X."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL,DDIMScheduler,UNet2DModel

def args():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('data/sgldm_v2x_2k/synthetic'))
 p.add_argument('--vae',type=Path,default=Path('outputs/rangeldm_vae_m1_full_obj2/epoch_060'))
 p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--scheduler',type=Path,required=True)
 p.add_argument('--profile',type=Path,default=Path('outputs/sensor_profile_v2')); p.add_argument('--output',type=Path,required=True)
 p.add_argument('--strengths',type=float,nargs='+',default=[.15,.25,.35,.45]); p.add_argument('--guidance',type=float,default=2.0)
 p.add_argument('--steps',type=int,default=50); p.add_argument('--frames',type=int,default=200); p.add_argument('--start',type=int,default=0)
 p.add_argument('--batch-size',type=int,default=16); p.add_argument('--width',type=int,default=256); p.add_argument('--seed',type=int,default=20260724)
 return p.parse_args()

def load(paths,width):
 images,sems,raw=[],[],[]
 for p in paths:
  with np.load(p) as z:
   depth=torch.from_numpy(z['log_range'].copy()).float()[None]; mask=torch.from_numpy(z['mask'].copy()).float()[None]
   fg=torch.from_numpy(z['semantic'].copy()).float().permute(2,0,1); source={k:z[k].copy() for k in ('range','mask','semantic')}
  bg=(mask*(1-fg.amax(0,keepdim=True))).clamp(0,1); sem=torch.cat([bg,fg],0)
  image=torch.cat([depth.clamp(0,1)*2-1,mask],0)
  images.append(F.interpolate(image[None],(128,width),mode='bilinear',align_corners=False)[0]); sems.append(F.interpolate(sem[None],(16,width//8),mode='nearest')[0]); raw.append(source)
 return torch.stack(images),torch.stack(sems),raw

def reconstruct(decoded,rings):
 decoded=F.interpolate(decoded,(128,1024),mode='bilinear',align_corners=False)
 lr=((decoded[:,0]+1)/2).clamp(0,1); ranges=torch.pow(2.,lr*8)-1; masks=torch.sigmoid(decoded[:,1])>=.5; ranges*=masks
 r=ranges.cpu().numpy(); m=masks.cpu().numpy(); az=((np.arange(1024)+.5)/1024*2*np.pi-np.pi)[None,:]; el=rings[:,None]
 dirs=np.stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.broadcast_to(np.sin(el),(128,1024))],-1).astype(np.float32)
 return r,m,r[...,None]*dirs[None]

def main():
 c=args(); device=torch.device('cuda'); torch.manual_seed(c.seed); paths=sorted(c.data.glob('*.npz'))[c.start:c.start+c.frames]
 vae=AutoencoderKL.from_pretrained(c.vae).to(device).eval(); unet=UNet2DModel.from_pretrained(c.checkpoint/'unet').to(device).eval()
 state=torch.load(c.checkpoint/'conditioning.pt',map_location='cpu'); mapper=torch.nn.Conv2d(4,8,1).to(device).eval(); mapper.load_state_dict(state['mapper'])
 sched=DDIMScheduler.from_pretrained(c.scheduler); rings=np.load(c.profile/'ring_angles.npy').astype(np.float32); c.output.mkdir(parents=True,exist_ok=True); summaries=[]
 with torch.inference_mode(),torch.cuda.amp.autocast(dtype=torch.float16):
  for strength in c.strengths:
   out=c.output/f'strength_{strength:.2f}'; out.mkdir(parents=True,exist_ok=True); rows=[]; gen=torch.Generator(device=device).manual_seed(c.seed+round(strength*1000))
   for start in range(0,len(paths),c.batch_size):
    subset=paths[start:start+c.batch_size]; image,sem,raw=load(subset,c.width); image,sem=image.to(device),sem.to(device)
    latent=vae.encode(image).latent_dist.mode()*vae.config.scaling_factor; sched.set_timesteps(c.steps,device=device); n=max(1,min(c.steps,round(c.steps*strength))); ts=sched.timesteps[c.steps-n:]
    noise=torch.randn(latent.shape,generator=gen,device=device,dtype=latent.dtype); sample=sched.add_noise(latent,noise,ts[0].expand(len(subset)))
    cond=mapper(sem); null=mapper(torch.zeros_like(sem))
    for t in ts:
     both=torch.cat([sample,sample]); pred=unet(torch.cat([both,torch.cat([null,cond])],1),t).sample; pu,pc=pred.chunk(2); pred=pu+c.guidance*(pc-pu); sample=sched.step(pred,t,sample).prev_sample
    decoded=vae.decode(sample/vae.config.scaling_factor).sample.float(); ranges,masks,xyz=reconstruct(decoded,rings)
    for i,p in enumerate(subset):
     srcmask=raw[i]['mask']>0; obj=raw[i]['semantic'].sum(-1)>0; common=srcmask&masks[i]; objcommon=common&obj
     rows.append({'frame':p.stem,'points':int(masks[i].sum()),'mask_iou':float(common.sum()/max((srcmask|masks[i]).sum(),1)),'depth_mae':float(np.abs(ranges[i][common]-raw[i]['range'][common]).mean()) if common.any() else None,'object_depth_mae':float(np.abs(ranges[i][objcommon]-raw[i]['range'][objcommon]).mean()) if objcommon.any() else None,'object_ray_recall':float(objcommon.sum()/max(obj.sum(),1))})
     np.save(out/f'{p.stem}.npy',xyz[i][masks[i]].astype(np.float32))
   valid=lambda k:[r[k] for r in rows if r[k] is not None]; summary={'strength':strength,'frames':len(rows),'points_median':float(np.median(valid('points'))),'mask_iou_median':float(np.median(valid('mask_iou'))),'depth_mae_median':float(np.median(valid('depth_mae'))),'object_depth_mae_median':float(np.median(valid('object_depth_mae'))),'object_ray_recall_median':float(np.median(valid('object_ray_recall')))}
   (out/'audit.json').write_text(json.dumps({'summary':summary,'frames':rows},indent=2)+'\n'); summaries.append(summary); print(json.dumps(summary),flush=True)
 (c.output/'strength_scan.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__': main()
