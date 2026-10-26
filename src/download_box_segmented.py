#!/usr/bin/env python3
"""Parallel, resumable Range downloader for one public Box shared file."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

import requests


PRINT_LOCK = threading.Lock()


def get_metadata(shared_url: str) -> dict:
    response = requests.get(shared_url, timeout=60)
    response.raise_for_status()
    match = re.search(r"Box\.prefetchedData\s*=\s*(\{.*?\});\s*Box\.", response.text)
    if not match:
        raise RuntimeError("No Box prefetched metadata")
    data = json.loads(match.group(1))["preview_metadata"]
    shared_name = shared_url.rstrip("/").rsplit("/", 1)[-1]
    return {
        "name": data["name"], "size": int(data["size"]), "sha1": data["sha1"],
        "download_url": (
            "https://ucla.app.box.com/index.php?rm=box_download_shared_file"
            f"&shared_name={shared_name}&file_id=f_{data['id']}"
        ),
    }


def download_segment(url: str, path: Path, start: int, end: int) -> Path:
    expected = end - start + 1
    while True:
        current = path.stat().st_size if path.exists() else 0
        if current == expected:
            return path
        if current > expected:
            raise RuntimeError(f"Oversized segment: {path}")
        range_start = start + current
        try:
            session = requests.Session()
            with session.get(
                url, headers={"Range": f"bytes={range_start}-{end}"},
                stream=True, timeout=(60, 180),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"Expected HTTP 206, got {response.status_code}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {range_start}-{end}/"):
                    raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                with path.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
        except Exception as error:
            with PRINT_LOCK:
                done = path.stat().st_size if path.exists() else 0
                print(
                    f"RETRY {path.name}: {done}/{expected} bytes after {type(error).__name__}: {error}",
                    flush=True,
                )
            time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = get_metadata(args.url)
    final = args.output_dir / info["name"]
    prefix = final.with_suffix(final.suffix + ".part")
    manifest_path = final.with_suffix(final.suffix + ".segments.json")
    if final.exists() and final.stat().st_size == info["size"]:
        print(f"SKIP complete {final}", flush=True)
        return
    if not prefix.exists():
        prefix.touch()

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["size"] != info["size"] or manifest["sha1"] != info["sha1"]:
            raise SystemExit("Existing segment manifest does not match remote metadata")
        prefix_size = manifest["prefix_size"]
        if prefix.stat().st_size != prefix_size:
            raise SystemExit("Frozen prefix size changed after segment manifest creation")
        ranges = [tuple(value) for value in manifest["ranges"]]
    else:
        prefix_size = prefix.stat().st_size
        remaining = info["size"] - prefix_size
        if remaining <= 0:
            ranges = []
        else:
            count = min(args.segments, remaining)
            base, extra = divmod(remaining, count)
            ranges = []
            cursor = prefix_size
            for index in range(count):
                length = base + (1 if index < extra else 0)
                ranges.append((cursor, cursor + length - 1))
                cursor += length
        manifest = {
            "name": info["name"], "size": info["size"], "sha1": info["sha1"],
            "prefix_size": prefix_size, "ranges": ranges,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"{info['name']}: frozen prefix {prefix_size / 1e9:.2f} GB, "
        f"{len(ranges)} parallel ranges, total {info['size'] / 1e9:.2f} GB",
        flush=True,
    )
    paths = [
        args.output_dir / f"{info['name']}.segment_{index:02d}.part"
        for index in range(len(ranges))
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(ranges))) as executor:
        futures = [
            executor.submit(download_segment, info["download_url"], path, start, end)
            for path, (start, end) in zip(paths, ranges)
        ]
        while not all(future.done() for future in futures):
            downloaded = prefix_size + sum(path.stat().st_size if path.exists() else 0 for path in paths)
            print(
                f"PROGRESS {info['name']}: {downloaded / 1e9:.2f}/{info['size'] / 1e9:.2f} GB "
                f"({100 * downloaded / info['size']:.1f}%)", flush=True,
            )
            time.sleep(60)
        for future in futures:
            future.result()

    assembling = final.with_suffix(final.suffix + ".assembling")
    required = info["size"] + 5_000_000_000
    if shutil.disk_usage(args.output_dir).free < required:
        raise SystemExit("Insufficient free space to assemble and retain segment files")
    digest = hashlib.sha1()
    with assembling.open("wb") as target:
        for source_path in [prefix] + paths:
            with source_path.open("rb") as source:
                for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
                    target.write(chunk)
                    digest.update(chunk)
        target.flush()
        os.fsync(target.fileno())
    if assembling.stat().st_size != info["size"] or digest.hexdigest() != info["sha1"]:
        raise RuntimeError(
            f"Assembly verification failed: size={assembling.stat().st_size}, sha1={digest.hexdigest()}"
        )
    assembling.replace(final)
    for temporary in [prefix, manifest_path] + paths:
        temporary.unlink()
    print(f"DONE {final} ({info['size']:,} bytes, SHA1 {info['sha1']})", flush=True)


if __name__ == "__main__":
    main()
