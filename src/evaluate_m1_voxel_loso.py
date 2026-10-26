#!/usr/bin/env python3
"""Leave-one-scenario-out audit for the training-only M1 voxel generator."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "competition" / "starting_kit" / "scoring_program")]
from export_lidm_reconstruction import native_range_to_xyz, PC_MIN, PC_MAX  # noqa: E402
import metrics  # noqa: E402
from evaluate_realism_baselines import normalized_score  # noqa: E402


def load_xyz(path, angles):
    with np.load(path) as item:
        return native_range_to_xyz(item["log_range"], item["mask"].astype(bool), angles)


def fit(paths, angles, step):
    bins = np.rint((PC_MAX - PC_MIN) / step).astype(np.int64)
    density = np.zeros(int(np.prod(bins)), np.float64); counts=[]
    for path in paths:
        xyz=load_xyz(path,angles)
        grid=np.clip(np.floor((xyz-PC_MIN)/step).astype(np.int64),0,bins-1)
        flat=np.ravel_multi_index(grid.T,bins)
        np.add.at(density,flat,1./len(xyz)); counts.append(len(xyz))
    nz=np.flatnonzero(density); p=density[nz]; p/=p.sum()
    return bins,nz,p,counts


def sample(bins,nz,p,counts,step,nframes,rng):
    result=[]
    for _ in range(nframes):
        n=int(counts[rng.integers(len(counts))]); flat=rng.choice(nz,n,replace=True,p=p)
        grid=np.column_stack(np.unravel_index(flat,bins))
        result.append((PC_MIN+(grid+rng.random((n,3)))*step).astype(np.float32))
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--voxel-step',type=float,nargs=3,default=(.4,.4,.2)); p.add_argument('--frames',type=int,default=50); p.add_argument('--subsample',type=int,default=10000); p.add_argument('--seed',type=int,default=20260731); c=p.parse_args()
    manifest=json.loads((c.corpus/'manifest.json').read_text()); by=defaultdict(list)
    for row in manifest['records']:
        # All corpus records were prefiltered against forbidden_frames.txt.
        by[row['scenario']].append(c.corpus/row['subset']/row['output'])
    angles=np.load(c.corpus/'ring_angles.npy').astype(np.float32); step=np.asarray(c.voxel_step,np.float64); metrics.N_SUBSAMPLE=c.subsample; rows=[]
    for fold,(held,paths) in enumerate(sorted(by.items())):
        train=[p for name,group in by.items() if name!=held for p in group]
        bins,nz,prob,counts=fit(train,angles,step)
        rng=np.random.default_rng(c.seed+fold); syn=sample(bins,nz,prob,counts,step,c.frames,rng)
        real_paths=paths if len(paths)<=c.frames else [paths[i] for i in np.linspace(0,len(paths)-1,c.frames,dtype=int)]
        real=[load_xyz(path,angles) for path in real_paths]
        result=metrics.compute_realism(syn,real,tuple(np.r_[PC_MIN,PC_MAX]),seed=1234); result['normalized_realism']=normalized_score(result)
        row={'held_scenario':held,'fit_frames':len(train),'real_frames':len(real),'synthetic_points':int(sum(map(len,syn))),**result}; rows.append(row); print(json.dumps(row),flush=True)
    scores=np.asarray([r['normalized_realism'] for r in rows]); out={'protocol':'leave-one-scenario-out; frame-balanced voxel density fitted only on other allowed scenarios','voxel_step':step.tolist(),'frames':c.frames,'summary':{'mean':float(scores.mean()),'std':float(scores.std()),'min':float(scores.min()),'max':float(scores.max())},'folds':rows}; c.output.parent.mkdir(parents=True,exist_ok=True); c.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out['summary'],indent=2))
if __name__=='__main__': main()
