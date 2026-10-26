#!/usr/bin/env python3
"""Wait for the four verified train archives, then safely extract atomically."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path


EXPECTED = {
    "train1.zip": 11_924_436_997,
    "train2.zip": 10_818_096_799,
    "train3.zip": 10_403_114_237,
    "train4.zip": 8_180_373_064,
}


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    for member in members:
        path = Path(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe path in archive: {member.filename}")
    return members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    staging = args.target.with_name(args.target.name + "_extracting")
    if args.target.exists():
        raise SystemExit(f"Target already exists; refusing to overwrite: {args.target}")
    if staging.exists():
        raise SystemExit(f"Staging already exists; inspect before retrying: {staging}")

    last_report = 0.0
    while True:
        complete = []
        status = {}
        for name, expected_size in EXPECTED.items():
            path = args.archives / name
            partial = path.with_suffix(path.suffix + ".part")
            if path.exists() and path.stat().st_size == expected_size:
                complete.append(path)
                status[name] = "complete"
            elif partial.exists():
                status[name] = f"partial {partial.stat().st_size}/{expected_size}"
            else:
                status[name] = "missing"
        now = time.monotonic()
        if now - last_report >= 300 or len(complete) == len(EXPECTED):
            print(json.dumps(status, indent=2), flush=True)
            last_report = now
        if len(complete) == len(EXPECTED):
            break
        time.sleep(args.poll_seconds)

    records = []
    total_uncompressed = 0
    for path in complete:
        with zipfile.ZipFile(path) as archive:
            members = safe_members(archive)
            size = sum(member.file_size for member in members)
            total_uncompressed += size
            records.append({
                "archive": str(path), "compressed_bytes": path.stat().st_size,
                "uncompressed_bytes": size, "members": len(members),
            })
    free = shutil.disk_usage(args.target.parent).free
    print(
        f"Required uncompressed: {total_uncompressed / 1e9:.1f} GB; "
        f"free: {free / 1e9:.1f} GB", flush=True,
    )
    if free < total_uncompressed + 10_000_000_000:
        raise SystemExit("Insufficient disk space (requires 10 GB safety margin)")

    staging.mkdir(parents=True)
    for index, path in enumerate(complete, 1):
        print(f"Extracting [{index}/{len(complete)}] {path.name}", flush=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(staging, members=safe_members(archive))
    staging.replace(args.target)
    payload = {
        "status": "complete", "target": str(args.target),
        "total_uncompressed_bytes": total_uncompressed, "archives": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
