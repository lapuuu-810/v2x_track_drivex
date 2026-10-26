#!/usr/bin/env python3
"""Apply a small, explicit LiDAR coordinate calibration to a sparse anchor."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--xy-scale',type=float,default=1.); p.add_argument('--x-scale',type=float); p.add_argument('--y-scale',type=float); p.add_argument('--x-offset',type=float,default=0.); p.add_argument('--y-offset',type=float,default=0.); p.add_argument('--z-scale',type=float,default=1.); p.add_argument('--z-offset',type=float,default=0.); p.add_argument('--yaw',type=float,default=0.); a=p.parse_args()
 files=sorted(a.input.glob('*.bin')); a.output.mkdir(parents=True,exist_ok=True)
 c,s=np.cos(a.yaw),np.sin(a.yaw)
 for i,f in enumerate(files,1):
  x=np.fromfile(f,np.float32).reshape(-1,3).copy(); xx,yy=x[:,0].copy(),x[:,1].copy(); sx=a.xy_scale if a.x_scale is None else a.x_scale; sy=a.xy_scale if a.y_scale is None else a.y_scale; x[:,0]=sx*(c*xx-s*yy)+a.x_offset; x[:,1]=sy*(s*xx+c*yy)+a.y_offset; x[:,2]=a.z_scale*x[:,2]+a.z_offset; x=np.clip(x,[-40,-40,-8],[40,40,2]).astype(np.float32); x.tofile(a.output/f'{i:04d}.bin')
 (a.output/'manifest.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n'); print({'frames':len(files),'xy_scale':a.xy_scale,'z_scale':a.z_scale,'z_offset':a.z_offset,'yaw':a.yaw})
if __name__=='__main__': main()
