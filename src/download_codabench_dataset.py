#!/usr/bin/env python3
"""Download an authorized Codabench dataset by UUID key."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

import requests


BASE = "https://www.codabench.org"


def checked(response: requests.Response, action: str) -> requests.Response:
    if response.ok:
        return response
    raise SystemExit(f"{action} failed: HTTP {response.status_code}: {response.text[:1000]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"Refusing to overwrite existing file: {args.output}")

    password = getpass.getpass("Codabench password: ")
    auth = checked(
        requests.post(
            f"{BASE}/api/api-token-auth/",
            json={"username": args.username, "password": password},
            timeout=30,
        ),
        "Authentication",
    ).json()
    password = None
    headers = {"Authorization": f"Token {auth['token']}"}
    metadata = checked(
        requests.get(f"{BASE}/api/datasets/{args.key}/", headers=headers, timeout=30),
        "Dataset metadata",
    ).json()
    download_url = metadata.get("data_file") or metadata.get("download_url")
    if not download_url:
        raise SystemExit(f"Dataset metadata lacks a download URL; fields={sorted(metadata)}")
    response = checked(requests.get(download_url, stream=True, timeout=300), "Dataset download")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    print(f"Downloaded {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
