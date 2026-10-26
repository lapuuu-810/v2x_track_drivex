#!/usr/bin/env python3
"""Choose unseen validation target frames for static-to-real renderer audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-manifest", type=Path, required=True)
    p.add_argument("--render-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=50)
    a = p.parse_args()
    pairs = json.loads(a.pairs_manifest.read_text())["records"]
    used_conditions = {Path(r["condition"]).stem for r in json.loads(a.render_manifest.read_text())["frames"]}
    rows = [r for r in pairs if r["subset"] == "validation" and r["key"] not in used_conditions]
    if len(rows) < a.frames:
        raise RuntimeError(f"need {a.frames} unseen validation targets, have {len(rows)}")
    rows = [rows[i] for i in np.linspace(0, len(rows) - 1, a.frames, dtype=int)]
    result = {"protocol": "cross-scene renderer validation: model trained only on train scenes; target frames do not occur in any rendered condition archive", "validation_paths": [r["target"] for r in rows], "scenes": sorted({r["scene"] for r in rows}), "frames": len(rows)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"frames": len(rows), "scenes": len(result["scenes"])}, indent=2))


if __name__ == "__main__":
    main()
