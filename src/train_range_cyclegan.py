#!/usr/bin/env python3
"""Unpaired, content-preserving range-image Sim2Real translator.

The generator predicts a bounded residual over synthetic log-range/mask images.
It is trained against real LiDAR with a patch discriminator while an identity,
edge, and cycle loss prevents scene geometry from being discarded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class RangeSet(Dataset):
    def __init__(self, folder: Path, width: int) -> None:
        self.files = sorted(folder.glob("*.npz"))
        self.width = width
        if not self.files:
            raise FileNotFoundError(folder)
    def __len__(self) -> int: return len(self.files)
    def __getitem__(self, i: int) -> torch.Tensor:
        with np.load(self.files[i]) as x:
            depth = torch.from_numpy(x["log_range"].copy()).float()[None] * 2 - 1
            mask = torch.from_numpy(x["mask"].copy()).float()[None]
        img = torch.cat([depth.clamp(-1, 1), mask], 0)
        return F.interpolate(img[None], (128, self.width), mode="bilinear", align_corners=False)[0]


def conv(a: int, b: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1), nn.InstanceNorm2d(b), nn.SiLU())


class Res(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__(); self.net = nn.Sequential(conv(c,c), nn.Conv2d(c,c,3,1,1), nn.InstanceNorm2d(c))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x + self.net(x)


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(conv(2,64), conv(64,128,2), conv(128,192,2), Res(192), Res(192), Res(192))
        self.dec = nn.Sequential(nn.ConvTranspose2d(192,128,4,2,1), nn.SiLU(), nn.ConvTranspose2d(128,64,4,2,1), nn.SiLU(), nn.Conv2d(64,2,3,1,1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.dec(self.enc(x))
        # Depth residual is deliberately bounded; mask is a logit correction.
        depth = (x[:, :1] + 0.22 * torch.tanh(residual[:, :1])).clamp(-1, 1)
        mask = x[:, 1:] + residual[:, 1:]
        return torch.cat([depth, mask], 1)


class PatchD(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(2,64,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(64,128,4,2,1),nn.InstanceNorm2d(128),nn.LeakyReLU(.2),nn.Conv2d(128,256,4,2,1),nn.InstanceNorm2d(256),nn.LeakyReLU(.2),nn.Conv2d(256,1,4,1,1))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)


def edge(x: torch.Tensor) -> torch.Tensor:
    return (x[...,1:] - x[...,:-1]).abs().mean() + (x[...,1:,:] - x[...,:-1,:]).abs().mean()


def parse() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('data/rangeldm_s2r_2k')); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=80); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--width',type=int,default=256); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--disc-lr',type=float,default=1e-4); p.add_argument('--workers',type=int,default=4); p.add_argument('--save-every',type=int,default=10); p.add_argument('--seed',type=int,default=20260728); return p.parse_args()


def main() -> None:
    c=parse(); distributed='LOCAL_RANK' in os.environ
    if distributed: dist.init_process_group('nccl')
    rank=int(os.environ.get('LOCAL_RANK',0)); torch.cuda.set_device(rank); device=torch.device('cuda',rank); torch.manual_seed(c.seed+rank)
    syn,real=RangeSet(c.data/'synthetic',c.width),RangeSet(c.data/'real',c.width)
    ss=DistributedSampler(syn,shuffle=True) if distributed else None; rs=DistributedSampler(real,shuffle=True) if distributed else None
    sl=DataLoader(syn,batch_size=c.batch_size,sampler=ss,shuffle=ss is None,num_workers=c.workers,pin_memory=True,drop_last=True,persistent_workers=c.workers>0); rl=DataLoader(real,batch_size=c.batch_size,sampler=rs,shuffle=rs is None,num_workers=c.workers,pin_memory=True,drop_last=True,persistent_workers=c.workers>0)
    g=Generator().to(device); f=Generator().to(device); d=PatchD().to(device)
    if distributed: g,f,d=(DDP(m,device_ids=[rank],find_unused_parameters=False) for m in (g,f,d))
    go=torch.optim.AdamW(list(g.parameters())+list(f.parameters()),lr=c.lr,betas=(.5,.999)); do=torch.optim.AdamW(d.parameters(),lr=c.disc_lr,betas=(.5,.999)); scaler=torch.cuda.amp.GradScaler(); c.output.mkdir(parents=True,exist_ok=True); hist=[]
    for epoch in range(1,c.epochs+1):
        if ss: ss.set_epoch(epoch); rs.set_epoch(epoch)
        totals=torch.zeros(3,device=device); rit=iter(rl)
        for x in sl:
            try: y=next(rit)
            except StopIteration: rit=iter(rl); y=next(rit)
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16): fake=g(x); dl=.5*((d(y)-1).square().mean()+d(fake.detach()).square().mean())
            do.zero_grad(set_to_none=True); scaler.scale(dl).backward(); scaler.step(do); scaler.update()
            go.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                fake=g(x); back=f(fake); ident=f(y); adv=.5*(d(fake)-1).square().mean(); cycle=F.l1_loss(back,x); identity=F.l1_loss(ident,y); structure=edge(fake[:,:1]-x[:,:1]); loss=adv+8*cycle+2*identity+2*structure
            scaler.scale(loss).backward(); scaler.step(go); scaler.update(); totals+=torch.tensor([loss.item(),adv.item(),dl.item()],device=device)
        if distributed: dist.all_reduce(totals)
        rec={'epoch':epoch,'loss':float(totals[0]/len(sl)),'adv':float(totals[1]/len(sl)),'disc':float(totals[2]/len(sl))}; hist.append(rec)
        if (not distributed or dist.get_rank()==0):
            print(json.dumps(rec),flush=True); (c.output/'history.json').write_text(json.dumps(hist,indent=2)+'\n')
            if epoch%c.save_every==0 or epoch==c.epochs: torch.save({'g':(g.module if distributed else g).state_dict(),'f':(f.module if distributed else f).state_dict(),'epoch':epoch},c.output/f'epoch_{epoch:03d}.pt')
    if distributed: dist.destroy_process_group()

if __name__=='__main__': main()
