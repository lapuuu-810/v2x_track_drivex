#!/usr/bin/env python3
"""Linearly blend two pointwise-aligned candidate directories."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser(); p.add_argument('--a',type=Path,required=True); p.add_argument('--b',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--alpha',type=float,required=True); a=p.parse_args(); af,bf=sorted(a.a.glob('*.bin')),sorted(a.b.glob('*.bin'))
 if len(af)!=50 or len(bf)!=50: raise SystemExit('need 50 files each')
 a.output.mkdir(parents=True,exist_ok=True)
 for i,(x,y) in enumerate(zip(af,bf),1):
  x=np.fromfile(x,np.float32).reshape(-1,3); y=np.fromfile(y,np.float32).reshape(-1,3)
  if x.shape!=y.shape: raise SystemExit(f'shape mismatch {x} {y}')
  np.clip((1-a.alpha)*x+a.alpha*y,[-40,-40,-8],[40,40,2]).astype(np.float32).tofile(a.output/f'{i:04d}.bin')
 (a.output/'manifest.json').write_text(json.dumps({'a':str(a.a),'b':str(a.b),'alpha':a.alpha},indent=2)+'\n'); print({'frames':50,'alpha':a.alpha})
if __name__=='__main__': main()
