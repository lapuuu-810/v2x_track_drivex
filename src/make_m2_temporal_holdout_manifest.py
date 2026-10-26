#!/usr/bin/env python3
"""Build a frame-disjoint (but scene-shared) M2 reference for renderer audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real-m2-root", type=Path, required=True)
    p.add_argument("--forbidden", type=Path, required=True)
    p.add_argument("--candidate-manifest", type=Path, required=True)
    p.add_argument("--frames-per-map", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=50)
    a = p.parse_args()
    forbidden = {x.strip().split("_")[-1] for x in a.forbidden.read_text().splitlines() if x.strip() and not x.startswith("#")}
    source_scenes = {r["scene"] for r in json.loads(a.candidate_manifest.read_text())["frames"]}
    available: list[tuple[str, list[Path]]] = []
    for sensor in sorted(p for p in a.real_m2_root.rglob("-2") if p.is_dir()):
        name = str(sensor.parent.relative_to(a.real_m2_root))
        if name not in source_scenes:
            continue
        paths = [x for x in sorted(sensor.glob("*.bin")) if x.stem not in forbidden]
        selected = set(np.linspace(0, len(paths) - 1, min(a.frames_per_map, len(paths)), dtype=int).tolist())
        held = [x for i, x in enumerate(paths) if i not in selected]
        if held:
            available.append((name, held))
    if not available:
        raise RuntimeError("no frame-disjoint paths")
    base, extra = divmod(a.frames, len(available))
    selected: list[str] = []
    for i, (_, paths) in enumerate(available):
        n = base + (i < extra)
        if n:
            selected.extend(str(paths[j]) for j in np.linspace(0, len(paths) - 1, n, dtype=int))
    result = {
        "protocol": "M2 temporal holdout: reference frames excluded from per-scene map fusion; static scene geometry is shared",
        "validation_paths": selected,
        "reference_scenes": len(available),
        "frames": len(selected),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"reference_scenes": len(available), "frames": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
