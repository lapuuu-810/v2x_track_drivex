#!/usr/bin/env python3
"""Range-download only /-1/*.bin records from a remote Box ZIP."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
import zipfile
from pathlib import Path

import requests

from inspect_remote_box_zip import RangeReader, box_info


PRINT_LOCK = threading.Lock()


def split_span(start: int, end: int, maximum: int) -> list[tuple[int, int]]:
    chunks = []
    while start <= end:
        current_end = min(end, start + maximum - 1)
        chunks.append((start, current_end))
        start = current_end + 1
    return chunks


def selected_chunks(members, start_dir: int, archive_size: int, maximum: int):
    physical = sorted(members, key=lambda member: member.header_offset)
    spans = []
    for index, member in enumerate(physical):
        if not (member.filename.endswith(".bin") and "/-1/" in f"/{member.filename.lstrip('/')}"):
            continue
        end = (physical[index + 1].header_offset if index + 1 < len(physical) else start_dir) - 1
        spans.append((member.header_offset, end))
    merged = []
    for start, end in spans:
        if merged and start == merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    chunks = []
    for start, end in merged:
        chunks.extend(split_span(start, end, maximum))
    chunks.extend(split_span(start_dir, archive_size - 1, maximum))
    return chunks


def local_coverages(archive_path: Path):
    coverages = []
    prefix = archive_path.with_suffix(archive_path.suffix + ".part")
    if prefix.exists() and prefix.stat().st_size:
        coverages.append((0, prefix.stat().st_size - 1, prefix, 0))
    segment_manifest = archive_path.with_suffix(archive_path.suffix + ".segments.json")
    if segment_manifest.exists():
        payload = json.loads(segment_manifest.read_text(encoding="utf-8"))
        for index, (start, _end) in enumerate(payload["ranges"]):
            path = archive_path.parent / f"{archive_path.name}.segment_{index:02d}.part"
            if path.exists() and path.stat().st_size:
                coverages.append((start, start + path.stat().st_size - 1, path, start))
    return coverages


def copy_local(source: Path, source_base: int, sparse_fd: int, start: int, end: int):
    remaining = end - start + 1
    source_offset = start - source_base
    target_offset = start
    with source.open("rb") as handle:
        handle.seek(source_offset)
        while remaining:
            data = handle.read(min(16 * 1024 * 1024, remaining))
            if not data:
                raise RuntimeError(f"Unexpected EOF in {source}")
            os.pwrite(sparse_fd, data, target_offset)
            remaining -= len(data)
            target_offset += len(data)


def fetch_chunk(shared_url: str, sparse_path: Path, marker: Path, start: int, end: int):
    if marker.exists():
        return
    expected = end - start + 1
    for attempt in range(1000):
        try:
            signed_url = box_info(shared_url)["url"]
            with requests.get(
                signed_url, headers={"Range": f"bytes={start}-{end}"},
                stream=True, timeout=(60, 180),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"Expected 206, got {response.status_code}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                fd = os.open(sparse_path, os.O_WRONLY)
                try:
                    offset = start
                    for data in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if data:
                            os.pwrite(fd, data, offset)
                            offset += len(data)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                if offset - start != expected:
                    raise RuntimeError(f"Short chunk: {offset - start}/{expected}")
            marker.touch()
            return
        except Exception as error:
            with PRINT_LOCK:
                print(
                    f"RETRY {start}-{end} after {type(error).__name__}: {error}", flush=True
                )
            time.sleep(min(60, 5 + attempt * 5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=32)
    args = parser.parse_args()
    info = box_info(args.url)
    archive_path = args.archive_dir / info["name"]
    stem = archive_path.stem
    target = args.output_dir / stem
    completion = target / ".selective_extract.json"
    if completion.exists():
        print(f"SKIP complete {target}", flush=True)
        return
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"Non-empty target without completion marker: {target}")

    remote = RangeReader(info["url"], info["size"])
    with zipfile.ZipFile(remote) as archive:
        members = archive.infolist()
        start_dir = archive.start_dir
    selected = [
        member for member in members
        if member.filename.endswith(".bin") and "/-1/" in f"/{member.filename.lstrip('/')}"
    ]
    chunks = selected_chunks(
        members, start_dir, info["size"], args.chunk_mib * 1024 * 1024
    )
    work = args.archive_dir / f".{stem}_selective"
    work.mkdir(parents=True, exist_ok=True)
    sparse = work / info["name"]
    if not sparse.exists():
        with sparse.open("wb") as handle:
            handle.truncate(info["size"])
    elif sparse.stat().st_size != info["size"]:
        raise SystemExit("Existing sparse ZIP has wrong logical size")
    markers = [work / f"chunk_{index:05d}.done" for index in range(len(chunks))]

    coverages = local_coverages(archive_path)
    reused = 0
    fd = os.open(sparse, os.O_WRONLY)
    try:
        for marker, (start, end) in zip(markers, chunks):
            if marker.exists():
                continue
            for cover_start, cover_end, source, source_base in coverages:
                if cover_start <= start and end <= cover_end:
                    copy_local(source, source_base, fd, start, end)
                    marker.touch()
                    reused += end - start + 1
                    break
        os.fsync(fd)
    finally:
        os.close(fd)
    required = sum(end - start + 1 for start, end in chunks)
    pending = [
        (marker, start, end) for marker, (start, end) in zip(markers, chunks)
        if not marker.exists()
    ]
    print(json.dumps({
        "archive": info["name"], "selected_members": len(selected),
        "selected_compressed_bytes": sum(member.compress_size for member in selected),
        "range_bytes_with_headers_and_central_directory": required,
        "locally_reused_bytes": reused, "chunks": len(chunks),
        "pending_chunks": len(pending), "workers": args.workers,
    }, indent=2), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(fetch_chunk, args.url, sparse, marker, start, end)
            for marker, start, end in pending
        ]
        while not all(future.done() for future in futures):
            done = sum(marker.exists() for marker in markers)
            print(f"PROGRESS {info['name']}: {done}/{len(markers)} chunks", flush=True)
            time.sleep(60)
        for future in futures:
            future.result()

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(sparse) as archive:
        for index, member in enumerate(selected, 1):
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe member path: {member.filename}")
            archive.extract(member, target)
            if index % 250 == 0:
                print(f"EXTRACT {info['name']}: {index}/{len(selected)}", flush=True)
    payload = {
        "archive": info["name"], "remote_size": info["size"],
        "selected_members": len(selected),
        "selected_uncompressed_bytes": sum(member.file_size for member in selected),
        "selected_compressed_bytes": sum(member.compress_size for member in selected),
        "network_range_bytes_upper_bound": required - reused,
        "locally_reused_bytes": reused,
    }
    completion.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sparse.unlink()
    for marker in markers:
        marker.unlink(missing_ok=True)
    work.rmdir()
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
