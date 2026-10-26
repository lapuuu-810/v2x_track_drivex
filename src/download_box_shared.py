#!/usr/bin/env python3
"""Resumably download public files from Box shared links."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import requests


def metadata(session: requests.Session, shared_url: str) -> dict:
    response = session.get(shared_url, timeout=60)
    response.raise_for_status()
    match = re.search(r"Box\.prefetchedData\s*=\s*(\{.*?\});\s*Box\.", response.text)
    if not match:
        raise RuntimeError(f"No prefetched metadata at {shared_url}")
    data = json.loads(match.group(1))["preview_metadata"]
    return {
        "name": data["name"],
        "size": int(data["size"]),
        "url": data["authenticated_download_url"],
    }


def download(session: requests.Session, shared_url: str, output_dir: Path) -> Path:
    info = metadata(session, shared_url)
    output = output_dir / info["name"]
    partial = output.with_suffix(output.suffix + ".part")
    if output.exists():
        if output.stat().st_size == info["size"]:
            print(f"SKIP complete {output}", flush=True)
            return output
        raise RuntimeError(f"Unexpected completed-file size: {output}")

    existing = partial.stat().st_size if partial.exists() else 0
    if existing > info["size"]:
        raise RuntimeError(f"Partial file is larger than expected: {partial}")
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with session.get(info["url"], headers=headers, stream=True, timeout=(60, 300)) as response:
        response.raise_for_status()
        if existing and response.status_code != 206:
            raise RuntimeError("Server ignored Range request; refusing to overwrite partial file")
        mode = "ab" if existing else "wb"
        downloaded = existing
        started = time.monotonic()
        last_report = started
        print(
            f"GET {info['name']} {existing / 1e9:.2f}/{info['size'] / 1e9:.2f} GB "
            f"(HTTP {response.status_code})",
            flush=True,
        )
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 30:
                    speed = (downloaded - existing) / max(now - started, 1) / 1e6
                    print(
                        f"  {downloaded / 1e9:.2f}/{info['size'] / 1e9:.2f} GB "
                        f"({100 * downloaded / info['size']:.1f}%, {speed:.1f} MB/s)",
                        flush=True,
                    )
                    last_report = now
            handle.flush()
            os.fsync(handle.fileno())
    actual = partial.stat().st_size
    if actual != info["size"]:
        raise RuntimeError(f"Incomplete {partial}: expected {info['size']}, got {actual}")
    partial.replace(output)
    print(f"DONE {output} ({actual:,} bytes)", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("urls", nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required = sum(metadata(requests.Session(), url)["size"] for url in args.urls)
    free = shutil.disk_usage(args.output_dir).free
    print(f"Disk free: {free / 1e9:.1f} GB; requested: {required / 1e9:.1f} GB", flush=True)
    if free < required + 5_000_000_000:
        raise SystemExit("Insufficient free space (requires 5 GB safety margin)")
    session = requests.Session()
    for url in args.urls:
        download(session, url, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
