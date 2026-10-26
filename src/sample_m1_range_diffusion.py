#!/usr/bin/env python3
"""Sample a saved M1 range-diffusion checkpoint without disturbing training."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import UNet2DModel

from train_m1_range_diffusion import sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = UNet2DModel.from_pretrained(args.checkpoint).to(device).eval()
    result = sample(model, args.data, args.output, args.frames, args.steps, args.seed, device)
    print(result)


if __name__ == "__main__":
    main()
