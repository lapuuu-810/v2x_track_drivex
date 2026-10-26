#!/usr/bin/env python3
"""Mix a bounded translated ray subset into an unchanged sparse anchor cloud."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def load(p: Path) -> np.ndarray:
 x=np.fromfile(p,np.float32).reshape(-1,3); ok=np.isfinite(x).all(1)&((x>=[-40,-40,-8])&(x<=[40,40,2])).all(1); return x[ok]
def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--anchor',type=Path,required=True); p.add_argument('--translated',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--fraction',type=float,required=True); p.add_argument('--points',type=int,default=9000); p.add_argument('--seed',type=int,default=20260728); a=p.parse_args()
 aa,bb=sorted(a.anchor.glob('*.bin')),sorted(a.translated.glob('*.bin'))
 if len(aa)!=50 or len(bb)!=50: raise SystemExit('need 50 anchor and 50 translated frames')
 a.output.mkdir(parents=True,exist_ok=True); info=[]
 for i,(pa,pb) in enumerate(zip(aa,bb),1):
  x,y=load(pa),load(pb); n=min(a.points,max(1,round(a.points*a.fraction))); rng=np.random.default_rng(a.seed+i); yi=y[rng.choice(len(y),n,replace=len(y)<n)]; xi=x[rng.choice(len(x),a.points-n,replace=len(x)<a.points-n)]; z=np.concatenate([xi,yi]).astype(np.float32); rng.shuffle(z); z.tofile(a.output/f'{i:04d}.bin'); info.append({'anchor':len(x),'translated':len(y),'inserted':n})
 (a.output/'manifest.json').write_text(json.dumps({'fraction':a.fraction,'points':a.points,'frames':info},indent=2)+'\n'); print({'frames':50,'inserted':n})
if __name__=='__main__': main()
