#!/usr/bin/env python3
"""Make a deterministic M2 scene-disjoint 50-frame realism reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real-m2-root", type=Path, required=True)
    p.add_argument("--forbidden", type=Path, required=True)
    p.add_argument("--candidate-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=50)
    a = p.parse_args()
    forbidden = {x.strip().split("_")[-1] for x in a.forbidden.read_text().splitlines() if x.strip() and not x.startswith("#")}
    used = set()
    candidate = json.loads(a.candidate_manifest.read_text())
    for row in candidate["frames"]:
        used.add(row["scene"])
    by_scene = {}
    for sensor in sorted(p for p in a.real_m2_root.rglob("-2") if p.is_dir()):
        name = str(sensor.parent.relative_to(a.real_m2_root))
        if name in used:
            continue
        paths = [x for x in sorted(sensor.glob("*.bin")) if x.stem not in forbidden]
        if paths:
            by_scene[name] = paths
    if not by_scene:
        raise RuntimeError("no scene-disjoint allowed M2 references available")
    names = sorted(by_scene)
    base, extra = divmod(a.frames, len(names))
    paths = []
    for i, name in enumerate(names):
        n = base + (i < extra)
        if n:
            scene = by_scene[name]
            indices = __import__("numpy").linspace(0, len(scene) - 1, n, dtype=int)
            paths.extend(str(scene[j]) for j in indices)
    record = {
        "protocol": "scene-disjoint M2 reference; forbidden frames excluded",
        "candidate_source_scenes": sorted(used),
        "reference_scenes": names,
        "validation_paths": paths,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"reference_scenes": len(names), "frames": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
