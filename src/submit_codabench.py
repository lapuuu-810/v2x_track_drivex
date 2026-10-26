#!/usr/bin/env python3
"""Upload a bundle to Codabench without persisting credentials or tokens."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import time
from pathlib import Path

import requests


BASE_URL = "https://www.codabench.org"


def checked(response, action):
    if response.ok:
        return response
    detail = response.text[:1000]
    raise SystemExit(f"{action} failed: HTTP {response.status_code}: {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--task", type=int, action="append", default=[])
    parser.add_argument("--description", default="GBlobs epoch20, full-public calibration")
    parser.add_argument("--poll-seconds", type=int, default=1200)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    if not bundle.is_file():
        raise SystemExit(f"Bundle not found: {bundle}")
    password = getpass.getpass("Codabench password: ")
    token_response = checked(
        requests.post(
            f"{BASE_URL}/api/api-token-auth/",
            json={"username": args.username, "password": password},
            timeout=30,
        ),
        "Authentication",
    )
    password = None
    token = token_response.json()["token"]
    headers = {"Authorization": f"Token {token}"}

    allowed = checked(
        requests.get(f"{BASE_URL}/api/can_make_submission/{args.phase}/", headers=headers, timeout=30),
        "Submission quota check",
    )
    print("Quota check:", json.dumps(allowed.json()), flush=True)

    size = bundle.stat().st_size
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    data_response = requests.post(
        f"{BASE_URL}/api/data/",
        headers=headers,
        json={"type": "submission", "file_size": size, "filename": bundle.name},
        timeout=30,
    )
    if data_response.status_code == 404:
        data_response = requests.post(
            f"{BASE_URL}/api/datasets/",
            headers=headers,
            json={
                "type": "submission",
                "request_sassy_file_name": bundle.name,
                "file_name": bundle.name,
                "file_size": size,
                "name": bundle.name,
                "is_public": False,
            },
            timeout=30,
        )
    data_record = checked(data_response, "Create upload record").json()
    key = data_record["key"]
    signed_url = data_record.get("sassy_url") or data_record.get("data_file")
    if not signed_url:
        raise SystemExit("Upload record did not include a signed upload URL")
    print(f"Upload record created: key={key}, bytes={size}, sha256={digest}", flush=True)

    with bundle.open("rb") as stream:
        checked(
            requests.put(
                signed_url,
                data=stream,
                headers={"Content-Type": "application/zip", "Content-Length": str(size)},
                timeout=300,
            ),
            "Bundle upload",
        )

    completed = requests.put(
        f"{BASE_URL}/api/data/upload_completed/{key}/", headers=headers, timeout=30
    )
    if completed.status_code == 404:
        completed = requests.put(
            f"{BASE_URL}/api/datasets/completed/{key}/", headers=headers, timeout=30
        )
    checked(completed, "Finalize upload")

    body = {"data": key, "phase": args.phase}
    if args.task:
        body["tasks"] = args.task
    submission = checked(
        requests.post(f"{BASE_URL}/api/submissions/", headers=headers, json=body, timeout=30),
        "Create submission",
    ).json()
    submission_id = submission["id"]
    print(f"Submission created: id={submission_id}, status={submission.get('status')}", flush=True)

    deadline = time.monotonic() + args.poll_seconds
    previous = None
    terminal = {"finished", "failed", "cancelled"}
    while time.monotonic() < deadline:
        current = checked(
            requests.get(f"{BASE_URL}/api/submissions/{submission_id}/", headers=headers, timeout=30),
            "Poll submission",
        ).json()
        status = str(current.get("status", "")).lower()
        if status != previous:
            print(f"Submission {submission_id}: {current.get('status')}", flush=True)
            previous = status
        if status in terminal:
            print(json.dumps(current, indent=2), flush=True)
            return
        time.sleep(5)
    print(f"Submission {submission_id}: still running after {args.poll_seconds}s", flush=True)


if __name__ == "__main__":
    main()
