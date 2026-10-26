#!/usr/bin/env python3
"""Partial monotone marginal transport from sparse anchor to allowed M2 LiDAR."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
R=np.array([[-40,40],[-40,40],[-8,2]],np.float32)
def crop(x): return x[np.isfinite(x).all(1)&np.any(x!=0,1)&((x>=R[:,0])&(x<=R[:,1])).all(1)]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--anchor',type=Path,required=True); p.add_argument('--real-root',type=Path,required=True); p.add_argument('--forbidden',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--blends',type=float,nargs='+',default=[.25,.5,.75,1.]); p.add_argument('--max-frames',type=int,default=800); p.add_argument('--seed',type=int,default=20260728); a=p.parse_args(); files=sorted(a.anchor.glob('*.bin')); clouds=[crop(np.fromfile(f,np.float32).reshape(-1,3)) for f in files]; source=np.concatenate(clouds)
 bad={x.strip().split('_')[-1] for x in a.forbidden.read_text().splitlines() if x.strip() and not x.startswith('#')}; paths=sorted(x for x in a.real_root.glob('**/-1/*.bin') if x.stem not in bad); paths=[paths[i] for i in np.linspace(0,len(paths)-1,min(a.max_frames,len(paths)),dtype=int)]; rng=np.random.default_rng(a.seed); parts=[]
 for f in paths:
  x=crop(np.fromfile(f,np.float32).reshape(-1,4)[:,:3]); parts.append(x[rng.choice(len(x),min(12000,len(x)),False)])
 target=np.concatenate(parts); q=np.linspace(0,1,2001); sk=np.quantile(source,q,axis=0); tk=np.quantile(target,q,axis=0); a.output.mkdir(parents=True,exist_ok=True); report={}
 for b in a.blends:
  d=a.output/f'blend_{b:.2f}'; d.mkdir(exist_ok=True); total=0
  for i,x in enumerate(clouds,1):
   y=np.column_stack([np.interp(x[:,j],sk[:,j],tk[:,j]) for j in range(3)]); y=np.clip((1-b)*x+b*y,R[:,0],R[:,1]).astype(np.float32); y.tofile(d/f'{i:04d}.bin'); total+=len(y)
  report[str(b)]={'points':total}
 (a.output/'manifest.json').write_text(json.dumps({'blends':a.blends,'real_frames':len(paths),'report':report},indent=2)+'\n'); print(report)
if __name__=='__main__': main()
