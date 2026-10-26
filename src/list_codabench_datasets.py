#!/usr/bin/env python3
"""Read-only listing of datasets owned by the authenticated Codabench user."""

from __future__ import annotations

import getpass
import json

import requests


BASE = "https://www.codabench.org"


def main() -> None:
    username = input("Codabench username: ").strip()
    password = getpass.getpass("Codabench password: ")
    auth = requests.post(
        f"{BASE}/api/api-token-auth/",
        json={"username": username, "password": password},
        timeout=30,
    )
    auth.raise_for_status()
    password = None
    headers = {"Authorization": f"Token {auth.json()['token']}"}
    url = f"{BASE}/api/datasets/"
    owned = []
    while url:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            rows, url = payload, None
        else:
            rows, url = payload.get("results", []), payload.get("next")
        for row in rows:
            owner = row.get("owner")
            owner_name = owner.get("username") if isinstance(owner, dict) else owner
            if owner_name in (None, username) or row.get("owner_username") == username:
                owned.append({
                    key: row.get(key)
                    for key in (
                        "key", "id", "name", "file_name", "file_size", "type",
                        "owner", "owner_username", "created", "created_when",
                        "data_file", "is_public",
                    )
                    if key in row
                })
    print(json.dumps(owned, indent=2))


if __name__ == "__main__":
    main()
