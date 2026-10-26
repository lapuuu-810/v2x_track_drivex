#!/usr/bin/env python3
"""Print only safe linkage/status metadata for one Codabench submission."""

from __future__ import annotations

import argparse
import getpass
import json

import requests


BASE = "https://www.codabench.org"
TERMS = ("id", "data", "file", "name", "phase", "status", "owner", "task")


def safe(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if any(term in key.lower() for term in TERMS):
                if isinstance(child, (str, int, float, bool)) or child is None:
                    print(f"{child_path} = {json.dumps(child)}")
            safe(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            safe(child, f"{path}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--submission-id", type=int, required=True)
    args = parser.parse_args()
    password = getpass.getpass("Codabench password: ")
    session = requests.Session()
    response = session.post(
        f"{BASE}/api/api-token-auth/",
        json={"username": args.username, "password": password}, timeout=30,
    )
    response.raise_for_status()
    password = None
    headers = {"Authorization": f"Token {response.json()['token']}"}
    response = session.get(
        f"{BASE}/api/submissions/{args.submission_id}/", headers=headers, timeout=30
    )
    response.raise_for_status()
    safe(response.json())


if __name__ == "__main__":
    main()
