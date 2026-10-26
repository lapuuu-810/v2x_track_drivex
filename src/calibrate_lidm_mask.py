#!/usr/bin/env python3
"""Calibrate LiDM return-mask threshold on the public M1 holdout only."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
PROJECT = Path(__file__).resolve().parents[1]
LIDM = PROJECT / "third_party" / "lidar_diffusion_cvpr24"
sys.path[:0] = [str(LIDM), str(PROJECT / "src" / "taming-transformers"), str(PROJECT / "src" / "clip")]
from lidm.utils.misc_utils import instantiate_from_config

def load_model(config_path, checkpoint, device):
    config = OmegaConf.load(config_path)
    if "lossconfig" in config.model.params:
        del config.model.params.lossconfig
    model = instantiate_from_config(config.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_m = [x for x in missing if not x.startswith("loss.")]
    bad_u = [x for x in unexpected if not x.startswith("loss.")]
    if bad_m or bad_u:
        raise RuntimeError((bad_m, bad_u))
    return model.to(device).eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    paths = sorted((args.corpus / "validation").glob("*.npz"))
    model = load_model(args.config, args.checkpoint, torch.device(args.device))
    logits, masks = [], []
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            ps = paths[start:start + args.batch_size]
            records = [np.load(x) for x in ps]
            try:
                log_range = np.stack([x["log_range"].astype(np.float32) for x in records])
                masks.append(np.stack([x["mask"].astype(bool) for x in records]))
            finally:
                for x in records: x.close()
            range_input = log_range * 2.0 - 1.0
            if model.encoder.conv_in.in_channels == 2:
                model_input = np.stack((range_input, masks[-1].astype(np.float32) * 2.0 - 1.0), axis=1)
            else:
                model_input = range_input[:, None]
            output, _ = model(torch.from_numpy(model_input).to(args.device))
            logits.append(output[:, 1].float().cpu().numpy())
    logits = np.concatenate(logits)
    masks = np.concatenate(masks)
    rows = []
    for threshold in np.linspace(-1.5, 1.5, 121):
        pred = logits < threshold
        tp = int((pred & masks).sum()); fp = int((pred & ~masks).sum()); fn = int((~pred & masks).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        rows.append({"threshold": float(threshold), "precision": prec, "recall": rec, "f1": 2*prec*rec/max(prec+rec,1e-12), "predicted_returns": int(pred.sum())})
    best = max(rows, key=lambda x: x["f1"])
    payload = {"checkpoint": str(args.checkpoint), "frames": len(paths), "best": best, "rows": rows,
               "logit_quantiles": {str(q): float(np.quantile(logits, q)) for q in (0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
if __name__ == "__main__": main()
