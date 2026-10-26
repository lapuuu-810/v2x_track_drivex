#!/usr/bin/env python3
"""Inspect a remote Box ZIP central directory using HTTP Range requests."""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile

import requests


def box_info(shared_url: str) -> dict:
    response = requests.get(shared_url, timeout=60)
    response.raise_for_status()
    match = re.search(r"Box\.prefetchedData\s*=\s*(\{.*?\});\s*Box\.", response.text)
    data = json.loads(match.group(1))["preview_metadata"]
    token = shared_url.rstrip("/").rsplit("/", 1)[-1]
    endpoint = (
        "https://ucla.app.box.com/index.php?rm=box_download_shared_file"
        f"&shared_name={token}&file_id=f_{data['id']}"
    )
    signed = requests.get(endpoint, allow_redirects=False, timeout=60)
    signed.raise_for_status()
    return {"name": data["name"], "size": int(data["size"]), "url": signed.headers["location"]}


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unknown whence {whence}")
        if position < 0:
            raise ValueError("Negative seek")
        self.position = min(position, self.size)
        return self.position

    def read(self, size=-1):
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        end = min(self.position + size, self.size) - 1
        start = self.position
        for attempt in range(8):
            try:
                response = requests.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(60, 180)
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"Expected HTTP 206, got {response.status_code}")
                data = response.content
                if len(data) != end - start + 1:
                    raise RuntimeError(f"Short Range response: {len(data)} bytes")
                self.position = end + 1
                return data
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(5 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    info = box_info(args.url)
    with zipfile.ZipFile(RangeReader(info["url"], info["size"])) as archive:
        members = archive.infolist()
    lidar = [member for member in members if member.filename.endswith(".bin")]
    infra = [
        member for member in lidar
        if "/-1/" in f"/{member.filename.lstrip('/')}"
    ]
    def summary(values):
        return {
            "members": len(values),
            "compressed_bytes": sum(value.compress_size for value in values),
            "uncompressed_bytes": sum(value.file_size for value in values),
        }
    payload = {
        "archive": info["name"], "archive_bytes": info["size"],
        "all_members": summary(members), "all_bin": summary(lidar),
        "infra_minus1_bin": summary(infra),
        "infra_fraction_of_archive": sum(value.compress_size for value in infra) / info["size"],
        "sample_names": [member.filename for member in infra[:10]],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
