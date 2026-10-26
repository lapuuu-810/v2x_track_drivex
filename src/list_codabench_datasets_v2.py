#!/usr/bin/env python3
"""Read-only, resilient listing of datasets owned by a Codabench user."""

from __future__ import annotations

import getpass
import os
import json
from urllib.parse import urlsplit

import requests


BASE = "https://www.codabench.org"


def main() -> None:
    username = os.environ.get("CODABENCH_USER") or input("Codabench username: ").strip()
    password = os.environ.get("CODABENCH_PASSWORD") or getpass.getpass("Codabench password: ")
    session = requests.Session()
    auth = session.post(
        f"{BASE}/api/api-token-auth/",
        json={"username": username, "password": password}, timeout=30,
    )
    auth.raise_for_status()
    password = None
    headers = {"Authorization": f"Token {auth.json()['token']}"}
    url = f"{BASE}/api/datasets/?page_size=100"
    page = 0
    while url:
        page += 1
        response = session.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            rows, next_url = payload, None
        else:
            rows, next_url = payload.get("results", []), payload.get("next")
        for row in rows:
            owner = row.get("owner")
            owner_name = owner.get("username") if isinstance(owner, dict) else owner
            if owner_name in (None, username) or row.get("owner_username") == username:
                selected = {
                    key: row.get(key) for key in (
                        "key", "id", "name", "file_name", "file_size", "type",
                        "owner", "owner_username", "created", "created_when", "is_public",
                    ) if key in row
                }
                print(json.dumps(selected, ensure_ascii=False), flush=True)
        if next_url:
            parsed = urlsplit(next_url)
            url = f"{BASE}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")
        else:
            url = None
        if page >= 20:
            raise RuntimeError("Unexpectedly exceeded 20 dataset pages")


if __name__ == "__main__":
    main()
