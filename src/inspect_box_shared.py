#!/usr/bin/env python3
"""Inspect public metadata embedded in a Box shared-file page."""

from __future__ import annotations

import argparse
import json
import re

import requests


INTERESTING = ("name", "size", "download", "typed", "item", "file")


def walk(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if any(term in key.lower() for term in INTERESTING) and not isinstance(child, (dict, list)):
                rendered = repr(child)
                print(f"{child_path} = {rendered[:500]}")
            walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk(child, f"{path}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    args = parser.parse_args()
    for url in args.urls:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        match = re.search(r"Box\.prefetchedData\s*=\s*(\{.*?\});\s*Box\.", response.text)
        if not match:
            raise SystemExit(f"No prefetchedData found at {url}")
        print(f"\n## {url}")
        walk(json.loads(match.group(1)))


if __name__ == "__main__":
    main()
