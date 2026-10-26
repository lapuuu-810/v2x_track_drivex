#!/usr/bin/env python3
"""Generate bounded residual Sim2Real LiDAR candidates from a range CycleGAN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_range_cyclegan import Generator


def parse() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--profile',type=Path,default=Path('outputs/sensor_profile_v2')); p.add_argument('--output',type=Path,required=True); p.add_argument('--blends',type=float,nargs='+',default=[.25,.5,.75,1.]); p.add_argument('--frames',type=int,default=50); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--width',type=int,default=256); return p.parse_args()


def load(paths: list[Path], width: int) -> torch.Tensor:
    rows=[]
    for p in paths:
        with np.load(p) as z:
            depth=torch.from_numpy(z['log_range'].copy()).float()[None]*2-1
            mask=torch.from_numpy(z['mask'].copy()).float()[None]
        rows.append(F.interpolate(torch.cat([depth.clamp(-1,1),mask],0)[None],(128,width),mode='bilinear',align_corners=False)[0])
    return torch.stack(rows)


def xyz(image: torch.Tensor, rings: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    image=F.interpolate(image,(128,1024),mode='bilinear',align_corners=False)
    log=((image[:,0]+1)/2).clamp(0,1); ranges=(torch.pow(2.,log*8)-1).cpu().numpy()
    masks=(image[:,1]>0.5).cpu().numpy()
    az=((np.arange(1024)+.5)/1024*2*np.pi-np.pi)[None,:]; el=rings[:,None]
    direction=np.stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.broadcast_to(np.sin(el),(128,1024))],-1).astype(np.float32)
    return ranges[...,None]*direction[None],masks


def main() -> None:
    c=parse(); device=torch.device('cuda'); files=sorted(c.data.glob('*.npz'))[:c.frames]
    if len(files)!=c.frames: raise SystemExit(f'expected {c.frames} npz files')
    model=Generator().to(device).eval(); state=torch.load(c.checkpoint,map_location='cpu'); model.load_state_dict(state['g'])
    rings=np.load(c.profile/'ring_angles.npy').astype(np.float32); out={b:c.output/f'blend_{b:.2f}' for b in c.blends}
    for d in out.values(): d.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(),torch.cuda.amp.autocast(dtype=torch.float16):
        for start in range(0,len(files),c.batch_size):
            x=load(files[start:start+c.batch_size],c.width).to(device); translated=model(x)
            for blend,d in out.items():
                points,masks=xyz(x+blend*(translated-x),rings)
                for j in range(len(points)):
                    cloud=points[j][masks[j]].astype(np.float32); cloud.tofile(d/f'{start+j+1:04d}.bin')
    report={str(b):{'frames':len(list(d.glob('*.bin'))),'total_points':sum(p.stat().st_size//12 for p in d.glob('*.bin'))} for b,d in out.items()}
    (c.output/'manifest.json').write_text(json.dumps({'checkpoint':str(c.checkpoint),'blends':c.blends,'report':report},indent=2)+'\n'); print(json.dumps(report))

if __name__=='__main__': main()
