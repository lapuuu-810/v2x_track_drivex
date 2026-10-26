#!/usr/bin/env python3
"""Delete one exact Codabench submission after verifying its dataset link."""

from __future__ import annotations

import argparse
import getpass
import json

import requests


BASE = "https://www.codabench.org"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--phase", type=int, required=True)
    args = parser.parse_args()
    password = getpass.getpass("Codabench password: ")
    session = requests.Session()
    auth = session.post(
        f"{BASE}/api/api-token-auth/",
        json={"username": args.username, "password": password}, timeout=30,
    )
    auth.raise_for_status()
    password = None
    headers = {"Authorization": f"Token {auth.json()['token']}"}
    url = f"{BASE}/api/submissions/{args.submission_id}/"
    response = session.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    detail = response.json()

    data = detail.get("data")
    if isinstance(data, dict):
        data_key = data.get("key")
    else:
        data_key = data
    phase = detail.get("phase")
    if isinstance(phase, dict):
        phase_id = phase.get("id")
    else:
        phase_id = phase
    verified = (
        int(detail.get("id", -1)) == args.submission_id
        and str(data_key) == args.dataset_key
        and int(phase_id) == args.phase
    )
    print(json.dumps({
        "id": detail.get("id"), "status": detail.get("status"),
        "data_key": data_key, "phase": phase_id, "verified": verified,
    }, indent=2), flush=True)
    if not verified:
        raise SystemExit("Refusing deletion: submission metadata did not exactly match")
    response = session.delete(url, headers=headers, timeout=30)
    if response.status_code not in (200, 202, 204):
        raise SystemExit(
            f"Delete failed: HTTP {response.status_code}: {response.text[:1000]}"
        )
    print(f"Deleted exact submission #{args.submission_id}", flush=True)


if __name__ == "__main__":
    main()
