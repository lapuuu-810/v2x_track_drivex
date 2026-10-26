#!/usr/bin/env python3
"""Project existing synthetic XYZ frames to the learned M1 laser grid."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--corpus',type=Path,required=True); p.add_argument('--output',type=Path,required=True); c=p.parse_args()
 angles=np.load(c.corpus/'ring_angles.npy').astype(np.float32); c.output.mkdir(parents=True,exist_ok=True)
 for path in sorted(c.input.glob('*.bin')):
  pts=np.fromfile(path,np.float32).reshape(-1,3); rng=np.linalg.norm(pts,axis=1); elev=np.arctan2(pts[:,2],np.linalg.norm(pts[:,:2],axis=1)); az=np.arctan2(pts[:,1],pts[:,0]); row=np.abs(elev[:,None]-angles[None]).argmin(1); col=np.floor((az+np.pi)/(2*np.pi)*1024).astype(int)%1024
  order=np.argsort(rng)[::-1]; grid=np.zeros((64,1024),np.float32); xyz=np.zeros((64,1024,3),np.float32); grid[row[order],col[order]]=rng[order]; xyz[row[order],col[order]]=pts[order]
  mask=grid>0; log=np.zeros_like(grid); log[mask]=np.log2(grid[mask]+1)/8; np.savez_compressed(c.output/f'{path.stem}.npz',log_range=log.astype(np.float16),mask=mask.astype(np.uint8),xyz=xyz)
 print(f'projected {len(list(c.output.glob("*.npz")))} frames')
if __name__=='__main__': main()
