#!/usr/bin/env python3
"""Sample a trained two-channel LiDM and export native M1 XYZ scans.

The script never copies public reference scans.  The only corpus statistic used is
the aggregate return ratio on the training split, used to turn the generated mask
logit into a global threshold.  Geometry uses fixed M1 ring angles.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
LIDM = PROJECT / "third_party" / "lidar_diffusion_cvpr24"
sys.path[:0] = [str(LIDM), str(PROJECT / "src" / "taming-transformers"), str(PROJECT / "src" / "clip")]

from lidm.models.diffusion.ddim import DDIMSampler  # noqa: E402
from lidm.utils.misc_utils import instantiate_from_config  # noqa: E402
from export_lidm_reconstruction import native_range_to_xyz  # noqa: E402


def train_return_ratio(corpus: Path) -> float:
    paths = sorted((corpus / "train").glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No training ranges under {corpus / 'train'}")
    valid = total = 0
    for path in paths:
        with np.load(path) as record:
            mask = record["mask"]
        valid += int(mask.sum())
        total += int(mask.size)
    return valid / total


def load_model(config_path: Path, checkpoint: Path) -> torch.nn.Module:
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [key for key in missing if not key.startswith("first_stage_model.loss.")]
    bad_unexpected = [key for key in unexpected if not key.startswith("first_stage_model.loss.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_missing}, unexpected={bad_unexpected}")
    return model.cuda().eval()


@torch.inference_mode()
def decode_raw(model: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Mirror LatentDiffusion.decode_first_stage but retain its two raw channels."""
    first_stage = model.first_stage_model
    z = latent / model.scale_factor
    quant, _, _ = first_stage.quantize(z)
    return first_stage.decoder(first_stage.post_quant_conv(quant))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Candidate root; scans are written to output/synthetic")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("num-samples and batch-size must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / "raw"
    raw_dir.mkdir(exist_ok=True)
    synthetic = args.output / "synthetic"
    synthetic.mkdir(exist_ok=True)
    ring_angles = np.load(args.corpus / "ring_angles.npy").astype(np.float32)

    model = load_model(args.config, args.checkpoint)
    sampler = DDIMSampler(model)
    image_size = model.model.diffusion_model.image_size
    shape = (model.model.diffusion_model.in_channels, *image_size)
    raw_paths: list[Path] = []
    generated = 0
    with model.ema_scope("native M1 sampling"):
        while generated < args.num_samples:
            batch = min(args.batch_size, args.num_samples - generated)
            latent, _ = sampler.sample(args.ddim_steps, batch_size=batch, shape=shape, eta=args.eta, verbose=False)
            decoded = decode_raw(model, latent).float().cpu().numpy()
            if decoded.shape[1:] != (2, 64, 1024):
                raise RuntimeError(f"Unexpected decoded shape {decoded.shape}")
            for item in decoded:
                path = raw_dir / f"{generated:06d}.npz"
                np.savez_compressed(path, decoded=item.astype(np.float16))
                raw_paths.append(path)
                generated += 1
            print(f"sampled {generated}/{args.num_samples}", flush=True)

    if args.mask_threshold is None:
        target_ratio = train_return_ratio(args.corpus)
        all_mask_values = np.concatenate([np.load(path)["decoded"][1].astype(np.float32).ravel() for path in raw_paths])
        threshold = float(np.quantile(all_mask_values, target_ratio))
        calibration = "global generated quantile matched to aggregate training return ratio"
    else:
        threshold = float(args.mask_threshold)
        target_ratio = None
        calibration = "explicit threshold"

    total_points = 0
    for raw_path in raw_paths:
        with np.load(raw_path) as record:
            decoded = record["decoded"].astype(np.float32)
        log_range = np.clip((decoded[0] + 1.0) * 0.5, 0.0, 1.0)
        mask = decoded[1] < threshold
        xyz = native_range_to_xyz(log_range, mask, ring_angles)
        xyz.tofile(synthetic / f"{raw_path.stem}.bin")
        total_points += len(xyz)

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_samples": args.num_samples,
        "ddim_steps": args.ddim_steps,
        "eta": args.eta,
        "seed": args.seed,
        "mask_threshold": threshold,
        "target_train_return_ratio": target_ratio,
        "calibration": calibration,
        "points": total_points,
        "mean_points_per_scan": total_points / args.num_samples,
        "geometry": "native M1 organized grid with fixed per-ring elevations",
        "reference_scan_reuse": False,
    }
    (args.output / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.keep_raw:
        for path in raw_paths:
            path.unlink()
        raw_dir.rmdir()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
