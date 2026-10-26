#!/usr/bin/env python3
"""Project sparse submission XYZ clouds onto the M2 128x1024 range grid."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--profile',type=Path,default=Path('outputs/sensor_profile_v2')); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    files=sorted(a.input.glob('*.bin')); rings=np.load(a.profile/'ring_angles.npy').astype(np.float32); a.output.mkdir(parents=True,exist_ok=True)
    if len(files)!=50: raise SystemExit(f'expected 50 frames, got {len(files)}')
    for i,file in enumerate(files):
        pts=np.fromfile(file,np.float32).reshape(-1,3); r=np.linalg.norm(pts,axis=1); rho=np.linalg.norm(pts[:,:2],axis=1); el=np.arctan2(pts[:,2],np.maximum(rho,1e-6)); az=np.arctan2(pts[:,1],pts[:,0])
        row=np.abs(el[:,None]-rings[None]).argmin(1); col=np.floor((az+np.pi)/(2*np.pi)*1024).astype(np.int64)%1024
        order=np.argsort(r)[::-1]; grid=np.zeros((128,1024),np.float32); xyz=np.zeros((128,1024,3),np.float32); grid[row[order],col[order]]=r[order]; xyz[row[order],col[order]]=pts[order]
        mask=grid>0; log=np.zeros_like(grid); log[mask]=np.log2(grid[mask]+1)/8
        semantic=np.zeros((128,1024,3),np.float32)
        np.savez_compressed(a.output/f'{i:04d}.npz',log_range=log,mask=mask.astype(np.float32),range=grid,xyz=xyz,semantic=semantic)
    print({'frames':len(files),'median_returns':int(np.median([np.load(x)['mask'].sum() for x in a.output.glob('*.npz')]))})

if __name__=='__main__': main()
