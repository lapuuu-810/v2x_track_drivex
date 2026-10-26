#!/usr/bin/env python3
"""Partial CORAL covariance calibration from allowed M2 real training scans."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

R=np.array([[-40,40],[-40,40],[-8,2]],np.float32)
def crop(x):
 return x[np.isfinite(x).all(1)&np.any(x!=0,1)&((x>=R[:,0])&(x<=R[:,1])).all(1)]
def sqrtm(c,inv=False):
 w,v=np.linalg.eigh(c); w=np.maximum(w,1e-5); return (v*(1/np.sqrt(w) if inv else np.sqrt(w)))@v.T
def main():
 p=argparse.ArgumentParser(); p.add_argument('--anchor',type=Path,required=True); p.add_argument('--real-root',type=Path,required=True); p.add_argument('--forbidden',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--blends',type=float,nargs='+',default=[.25,.5,.75,1.]); p.add_argument('--max-frames',type=int,default=800); p.add_argument('--seed',type=int,default=20260728); a=p.parse_args()
 af=sorted(a.anchor.glob('*.bin')); clouds=[crop(np.fromfile(f,np.float32).reshape(-1,3)) for f in af]; source=np.concatenate(clouds)
 bad={x.strip().split('_')[-1] for x in a.forbidden.read_text().splitlines() if x.strip() and not x.startswith('#')}; paths=sorted(x for x in a.real_root.glob('**/-1/*.bin') if x.stem not in bad); paths=[paths[i] for i in np.linspace(0,len(paths)-1,min(a.max_frames,len(paths)),dtype=int)]; rng=np.random.default_rng(a.seed); parts=[]
 for f in paths:
  x=crop(np.fromfile(f,np.float32).reshape(-1,4)[:,:3]); parts.append(x[rng.choice(len(x),min(12000,len(x)),False)])
 target=np.concatenate(parts); ms,mt=source.mean(0),target.mean(0); A=sqrtm(np.cov(source,rowvar=False),True)@sqrtm(np.cov(target,rowvar=False)); a.output.mkdir(parents=True,exist_ok=True); report={}
 for blend in a.blends:
  d=a.output/f'blend_{blend:.2f}'; d.mkdir(exist_ok=True); off=0
  for i,x in enumerate(clouds,1):
   y=(x-ms)@A+mt; y=(1-blend)*x+blend*y; y=np.clip(y,R[:,0],R[:,1]).astype(np.float32); y.tofile(d/f'{i:04d}.bin'); off+=len(x)
  report[str(blend)]={'points':off}
 (a.output/'manifest.json').write_text(json.dumps({'blends':a.blends,'real_frames':len(paths),'source_mean':ms.tolist(),'target_mean':mt.tolist(),'A':A.tolist(),'report':report},indent=2)+'\n'); print(report)
if __name__=='__main__': main()
