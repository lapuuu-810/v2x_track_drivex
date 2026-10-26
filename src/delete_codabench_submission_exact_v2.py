#!/usr/bin/env python3
"""Delete one submission after exact ID/owner/phase/task/filename verification."""

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
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--expected-file", required=True)
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
    url = f"{BASE}/api/submissions/{args.submission_id}/"
    response = session.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    detail = response.json()
    task = detail.get("task")
    task_id = task.get("id") if isinstance(task, dict) else task
    actual = {
        "id": detail.get("id"), "owner": detail.get("owner"),
        "phase": detail.get("phase"), "task": task_id,
        "filename": detail.get("filename"), "status": detail.get("status"),
    }
    verified = (
        int(actual["id"]) == args.submission_id
        and actual["owner"] == args.username
        and int(actual["phase"]) == args.phase
        and int(actual["task"]) == args.task
        and actual["filename"] == args.expected_file
        and str(actual["status"]).lower() == "finished"
    )
    actual["verified"] = verified
    print(json.dumps(actual, indent=2), flush=True)
    if not verified:
        raise SystemExit("Refusing deletion: exact submission metadata mismatch")
    response = session.delete(url, headers=headers, timeout=30)
    if response.status_code not in (200, 202, 204):
        raise SystemExit(f"Delete failed: HTTP {response.status_code}: {response.text[:1000]}")
    print(f"Deleted exact submission #{args.submission_id}", flush=True)


if __name__ == "__main__":
    main()
