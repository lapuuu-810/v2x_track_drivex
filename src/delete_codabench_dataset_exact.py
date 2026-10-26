#!/usr/bin/env python3
"""Delete one exact Codabench dataset after strict metadata verification."""

from __future__ import annotations

import argparse
import getpass
import os
import json

import requests


BASE = "https://www.codabench.org"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--expected-file", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    args = parser.parse_args()

    password = os.environ.get("CODABENCH_PASSWORD") or getpass.getpass("Codabench password: ")
    session = requests.Session()
    auth = session.post(
        f"{BASE}/api/api-token-auth/",
        json={"username": args.username, "password": password}, timeout=30,
    )
    auth.raise_for_status()
    password = None
    headers = {"Authorization": f"Token {auth.json()['token']}"}

    candidates = [
        f"{BASE}/api/datasets/{args.key}/",
        f"{BASE}/api/datasets/{args.id}/",
    ]
    detail = None
    detail_url = None
    failures = []
    for url in candidates:
        response = session.get(url, headers=headers, timeout=30)
        if response.ok:
            detail = response.json()
            detail_url = url
            break
        failures.append((url, response.status_code, response.text[:200]))
    if detail is None:
        raise SystemExit(f"Could not resolve dataset detail: {failures}")

    actual_id = int(detail.get("id", -1))
    actual_key = str(detail.get("key", ""))
    actual_file = detail.get("file_name") or detail.get("name")
    actual_size = int(float(detail.get("file_size", -1)))
    verified = (
        actual_id == args.id
        and actual_key == args.key
        and actual_file == args.expected_file
        and actual_size == args.expected_size
        and detail.get("type") == "submission"
    )
    print(json.dumps({
        "resolved_url": detail_url,
        "id": actual_id,
        "key": actual_key,
        "file_name": actual_file,
        "file_size": actual_size,
        "type": detail.get("type"),
        "verified": verified,
    }, indent=2), flush=True)
    if not verified:
        raise SystemExit("Refusing deletion: remote metadata did not exactly match")

    response = session.delete(detail_url, headers=headers, timeout=30)
    if response.status_code not in (200, 202, 204):
        raise SystemExit(
            f"Delete failed: HTTP {response.status_code}: {response.text[:1000]}"
        )
    print(f"Deleted exact dataset id={args.id}, key={args.key}", flush=True)


if __name__ == "__main__":
    main()
