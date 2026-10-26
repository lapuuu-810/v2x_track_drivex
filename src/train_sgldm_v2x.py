#!/usr/bin/env python3
"""SG-LDM-V2X: ICCV'25 semantic alignment adapted to native 128-beam LiDAR."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler

class RealSemanticDataset(Dataset):
    def __init__(self,root,width):
        self.files=sorted((root/'real').glob('*.npz')); self.width=width
        if not self.files: raise FileNotFoundError(root/'real')
        with np.load(self.files[0]) as z:
            if 'semantic' not in z: raise ValueError('real NPZ lacks semantic')
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        with np.load(self.files[i]) as z:
            depth=torch.from_numpy(z['log_range'].copy()).float()[None]
            mask=torch.from_numpy(z['mask'].copy()).float()[None]
            fg=torch.from_numpy(z['semantic'].copy()).float().permute(2,0,1)
        image=torch.cat([depth.clamp(0,1)*2-1,mask],0)
        bg=(mask*(1-fg.amax(0,keepdim=True))).clamp(0,1)
        sem=torch.cat([bg,fg],0)
        if image.shape[-1]!=self.width:
            image=F.interpolate(image[None],(128,self.width),mode='bilinear',align_corners=False)[0]
            sem=F.interpolate(sem[None],(128,self.width),mode='nearest')[0]
        return image,sem

class SGModel(nn.Module):
    def __init__(self,shape):
        super().__init__(); self.features=None
        self.mapper=nn.Conv2d(4,8,1)
        self.unet=UNet2DModel(sample_size=shape,in_channels=12,out_channels=4,layers_per_block=2,
          block_out_channels=(128,256,256),down_block_types=('DownBlock2D','AttnDownBlock2D','AttnDownBlock2D'),
          up_block_types=('AttnUpBlock2D','AttnUpBlock2D','UpBlock2D'),norm_num_groups=32)
        self.projector=nn.Sequential(nn.Conv2d(256,128,3,padding=1),nn.SiLU(),
          nn.Conv2d(128,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,4,1))
        self.unet.down_blocks[-1].register_forward_hook(self._capture)
    def _capture(self,module,inputs,output): self.features=output[0] if isinstance(output,tuple) else output
    def forward(self,noisy,t,sem,uncond):
        sem_in=sem.clone(); sem_in[uncond]=0
        pred=self.unet(torch.cat([noisy,self.mapper(sem_in)],1),t).sample
        align=pred.new_zeros(())
        if uncond.any():
            projected=self.projector(self.features[uncond])
            target=F.interpolate(sem[uncond],projected.shape[-2:],mode='nearest')
            cosine=F.cosine_similarity(projected.float(),target.float(),dim=1)
            weight=(1-t[uncond].float()/1000)[:,None,None]
            align=-(cosine*weight).mean()
        return pred,align

def parse():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('data/sgldm_v2x_2k'))
    p.add_argument('--vae',type=Path,default=Path('outputs/rangeldm_vae_m1_full_obj2/epoch_060'))
    p.add_argument('--output',type=Path,default=Path('outputs/sgldm_v2x'))
    p.add_argument('--width',type=int,default=256); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--unconditional-prob',type=float,default=.2); p.add_argument('--seed',type=int,default=20260724)
    return p.parse_args()

def main():
    c=parse(); distributed='LOCAL_RANK' in os.environ
    if distributed: dist.init_process_group('nccl')
    lrk=int(os.environ.get('LOCAL_RANK',0)); rank=dist.get_rank() if distributed else 0
    torch.cuda.set_device(lrk); device=torch.device('cuda',lrk); torch.manual_seed(c.seed+rank)
    ds=RealSemanticDataset(c.data,c.width); sampler=DistributedSampler(ds,shuffle=True) if distributed else None
    dl=DataLoader(ds,batch_size=c.batch_size,sampler=sampler,shuffle=sampler is None,num_workers=c.workers,
      pin_memory=True,drop_last=True,persistent_workers=c.workers>0)
    vae=AutoencoderKL.from_pretrained(c.vae).to(device).eval(); vae.requires_grad_(False)
    shape=(16,c.width//8); model=SGModel(shape).to(device)
    if distributed: model=DDP(model,device_ids=[lrk],find_unused_parameters=True)
    sched=DDPMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',prediction_type='epsilon',clip_sample=False)
    opt=torch.optim.AdamW(model.parameters(),lr=c.lr,weight_decay=1e-4); scaler=torch.cuda.amp.GradScaler()
    c.output.mkdir(parents=True,exist_ok=True); history=[]
    for epoch in range(1,c.epochs+1):
        if sampler: sampler.set_epoch(epoch)
        model.train(); total=torch.zeros(4,device=device)
        for image,sem in dl:
            image,sem=image.to(device,non_blocking=True),sem.to(device,non_blocking=True)
            with torch.no_grad(),torch.cuda.amp.autocast(dtype=torch.float16):
                latent=vae.encode(image).latent_dist.sample()*vae.config.scaling_factor
            sem=F.interpolate(sem,shape,mode='nearest'); noise=torch.randn_like(latent)
            t=torch.randint(0,1000,(latent.shape[0],),device=device); noisy=sched.add_noise(latent,noise,t)
            uncond=torch.rand(latent.shape[0],device=device)<c.unconditional_prob
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred,align=model(noisy,t,sem,uncond); ddpm=F.mse_loss(pred.float(),noise.float()); loss=ddpm+align
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt); scaler.update(); total+=torch.tensor([loss.item(),ddpm.item(),align.item(),1],device=device)
        if distributed: dist.all_reduce(total)
        rec={'epoch':epoch,'loss':float((total[0]/total[3]).cpu()),'ddpm':float((total[1]/total[3]).cpu()),'semantic_alignment':float((total[2]/total[3]).cpu())}; history.append(rec)
        if rank==0:
            print(json.dumps(rec),flush=True); (c.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            if epoch%c.save_every==0 or epoch==c.epochs:
                raw=model.module if distributed else model; out=c.output/f'epoch_{epoch:03d}'; out.mkdir(exist_ok=True)
                raw.unet.save_pretrained(out/'unet'); torch.save({'mapper':raw.mapper.state_dict(),'projector':raw.projector.state_dict()},out/'conditioning.pt'); sched.save_pretrained(c.output/'scheduler')
    if distributed: dist.destroy_process_group()
if __name__=='__main__': main()
