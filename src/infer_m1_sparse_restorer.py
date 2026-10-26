#!/usr/bin/env python3
"""Apply a trained self-supervised M1 restorer to projected synthetic scans."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from train_m1_sparse_restorer import Restorer

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--corpus',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--threshold',type=float,default=.5); p.add_argument('--preserve-observed',action='store_true',help='keep projected synthetic anchor points exactly and generate only missing rays'); p.add_argument('--device',default='cuda'); a=p.parse_args()
    files=sorted(a.input.glob('*.npz'))
    if len(files)!=50: raise SystemExit(f'expected 50 projected source frames, got {len(files)}')
    state=torch.load(a.checkpoint,map_location='cpu'); model=Restorer().to(a.device).eval(); model.load_state_dict(state['state_dict'])
    angles=np.load(a.corpus/'ring_angles.npy').astype(np.float32); az=((np.arange(1024)+.5)/1024*2*np.pi-np.pi)[None]; el=angles[:,None]
    direction=np.stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.broadcast_to(np.sin(el),(64,1024))],-1).astype(np.float32)
    a.output.mkdir(parents=True,exist_ok=True); counts=[]
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
      for index,path in enumerate(files,1):
        with np.load(path) as z:
          log=z['log_range'].astype(np.float32); observed_mask=z['mask'].astype(bool)
          observed_xyz=z['xyz'].astype(np.float32) if a.preserve_observed and 'xyz' in z else None
        source=torch.from_numpy(np.stack([(log*2-1)*observed_mask,observed_mask])[None]).to(a.device)
        pred=model(source)[0]; depth=((pred[0].float().cpu().numpy()+1)/2).clip(0,1); pred_mask=(torch.sigmoid(pred[1]).float().cpu().numpy()>a.threshold)
        ranges=np.power(2.,depth*8)-1
        if a.preserve_observed:
          generated=pred_mask & ~observed_mask
          points=np.concatenate([observed_xyz[observed_mask], (ranges[...,None]*direction)[generated].astype(np.float32)],axis=0)
        else:
          points=(ranges[...,None]*direction)[pred_mask].astype(np.float32)
        points.tofile(a.output/f'{index:04d}.bin'); counts.append(int(len(points)))
    (a.output/'manifest.json').write_text(json.dumps({'method':'self-supervised M1 sparse range restoration','checkpoint':str(a.checkpoint),'input':str(a.input),'threshold':a.threshold,'preserve_observed':a.preserve_observed,'frames':len(files),'point_counts':counts},indent=2)+'\n')
    print(json.dumps({'frames':len(files),'median_points':int(np.median(counts)),'total_points':int(sum(counts))},indent=2))
if __name__=='__main__': main()
