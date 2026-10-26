#!/usr/bin/env python3
"""Evaluate a generated candidate on multiple scenario-stratified M1 holdouts."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'competition'/'starting_kit'/'scoring_program')]
import metrics  # noqa:E402
from evaluate_realism_baselines import normalized_score,load_real  # noqa:E402

def main():
 p=argparse.ArgumentParser(); p.add_argument('--candidate',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--folds',type=int,default=3);p.add_argument('--frames',type=int,default=50);p.add_argument('--subsample',type=int,default=10000);p.add_argument('--seed',type=int,default=20260804);c=p.parse_args()
 manifest=json.loads((c.corpus/'manifest.json').read_text()); by=defaultdict(list)
 for r in manifest['records']:
  if r['subset']=='validation': by[r['scenario']].append(Path(r['source']))
 source=sorted(c.candidate.glob('*.bin')); 
 if len(source)<c.frames: raise SystemExit(f'candidate has only {len(source)} frames')
 source=[source[i] for i in np.linspace(0,len(source)-1,c.frames,dtype=int)]
 syn=[np.fromfile(q,np.float32).reshape(-1,3) for q in source]
 metrics.N_SUBSAMPLE=c.subsample; rows=[]; names=sorted(by)
 base,extra=divmod(c.frames,len(names))
 for fold in range(c.folds):
  rng=np.random.default_rng(c.seed+fold); selected=[]
  for i,name in enumerate(names):
   n=base+(i<extra); paths=by[name]; idx=rng.choice(len(paths),n,replace=False); selected += [paths[j] for j in idx]
  real=load_real(selected,len(selected)); res=metrics.compute_realism(syn,real,(-40,-40,-8,40,40,2),seed=1234);res['normalized_realism']=normalized_score(res);res['fold']=fold;res['scenario_frame_counts']={n:base+(i<extra) for i,n in enumerate(names)};rows.append(res);print(json.dumps(res),flush=True)
 scores=np.array([r['normalized_realism'] for r in rows]);out={'protocol':'scenario-stratified M1 validation; candidate fixed before holdout selection','candidate':str(c.candidate),'frames':c.frames,'folds':rows,'summary':{'mean':float(scores.mean()),'std':float(scores.std()),'min':float(scores.min()),'max':float(scores.max())}};c.output.parent.mkdir(parents=True,exist_ok=True);c.output.write_text(json.dumps(out,indent=2));print(json.dumps(out['summary'],indent=2))
if __name__=='__main__': main()
