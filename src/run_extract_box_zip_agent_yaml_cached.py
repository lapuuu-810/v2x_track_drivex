#!/usr/bin/env python3
"""Run the generic agent extractor with a shared cached Box signed URL."""

from __future__ import annotations

import os
import threading
import time

import requests

import extract_box_zip_agent_yaml as extractor
from inspect_remote_box_zip import box_info


class Provider:
    def __init__(self, shared_url: str):
        self.shared_url = shared_url
        self.url = box_info(shared_url)["url"]
        self.lock = threading.Lock()

    def get(self, refresh: bool = False) -> str:
        if refresh:
            with self.lock:
                self.url = box_info(self.shared_url)["url"]
        return self.url


PROVIDERS = {}
PROVIDERS_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()


def provider_for(shared_url: str) -> Provider:
    with PROVIDERS_LOCK:
        if shared_url not in PROVIDERS:
            PROVIDERS[shared_url] = Provider(shared_url)
        return PROVIDERS[shared_url]


def fetch_cached(shared_url, sparse_path, marker, start, end):
    if marker.exists():
        return
    expected = end - start + 1
    provider = provider_for(shared_url)
    for attempt in range(1000):
        try:
            signed_url = provider.get(refresh=attempt > 0)
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


extractor.fetch_chunk = fetch_cached


if __name__ == "__main__":
    extractor.main()
