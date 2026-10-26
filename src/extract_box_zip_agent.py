#!/usr/bin/env python3
"""Range-download one selected agent's .bin records from a remote Box ZIP."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import zipfile
from pathlib import Path

from extract_box_zip_selective import (
    box_info,
    copy_local,
    fetch_chunk,
    local_coverages,
    split_span,
)
from inspect_remote_box_zip import RangeReader


def selected_chunks(members, start_dir: int, archive_size: int, maximum: int, agent: str):
    physical = sorted(members, key=lambda member: member.header_offset)
    spans = []
    marker = f"/{agent}/"
    for index, member in enumerate(physical):
        if not (member.filename.endswith(".bin") and marker in f"/{member.filename.lstrip('/')}"):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=32)
    args = parser.parse_args()
    info = box_info(args.url)
    archive_path = args.archive_dir / info["name"]
    stem = archive_path.stem
    safe_agent = args.agent.replace("-", "m")
    target = args.output_dir / f"{stem}_agent_{safe_agent}"
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
    marker = f"/{args.agent}/"
    selected = [
        member for member in members
        if member.filename.endswith(".bin") and marker in f"/{member.filename.lstrip('/')}"
    ]
    if not selected:
        raise SystemExit(f"No {args.agent} .bin members in {info['name']}")
    chunks = selected_chunks(
        members, start_dir, info["size"], args.chunk_mib * 1024 * 1024, args.agent
    )
    work = args.archive_dir / f".{stem}_selective_{safe_agent}"
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
        for done, (start, end) in zip(markers, chunks):
            if done.exists():
                continue
            for cover_start, cover_end, source, source_base in coverages:
                if cover_start <= start and end <= cover_end:
                    copy_local(source, source_base, fd, start, end)
                    done.touch()
                    reused += end - start + 1
                    break
        os.fsync(fd)
    finally:
        os.close(fd)
    required = sum(end - start + 1 for start, end in chunks)
    pending = [
        (done, start, end) for done, (start, end) in zip(markers, chunks)
        if not done.exists()
    ]
    print(json.dumps({
        "archive": info["name"], "agent": args.agent,
        "selected_members": len(selected),
        "selected_compressed_bytes": sum(member.compress_size for member in selected),
        "range_bytes": required, "locally_reused_bytes": reused,
        "chunks": len(chunks), "pending_chunks": len(pending),
    }, indent=2), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(fetch_chunk, args.url, sparse, done, start, end)
            for done, start, end in pending
        ]
        while not all(future.done() for future in futures):
            print(
                f"PROGRESS {info['name']} agent {args.agent}: "
                f"{sum(done.exists() for done in markers)}/{len(markers)} chunks",
                flush=True,
            )
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
        "archive": info["name"], "agent": args.agent,
        "selected_members": len(selected),
        "selected_uncompressed_bytes": sum(member.file_size for member in selected),
        "selected_compressed_bytes": sum(member.compress_size for member in selected),
        "network_range_bytes_upper_bound": required - reused,
        "locally_reused_bytes": reused,
    }
    completion.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sparse.unlink()
    for done in markers:
        done.unlink(missing_ok=True)
    work.rmdir()
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
