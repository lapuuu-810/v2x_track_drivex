#!/usr/bin/env python3
"""Publish the authorized V2X-Real scoring question on Codabench."""

from __future__ import annotations

import argparse
import getpass
import html
import re
from urllib.parse import urljoin

import requests


BASE = "https://www.codabench.org"
FORUM_ID = 15799
TITLE = "All detection AP values are exactly zero despite valid, point-supported predictions"
BODY = """Hello organizers,

Could you please help us verify the frame-ID and coordinate alignment between the released V2X-Real detection-test point clouds and the server-side detection ground truth?

Our Development submission #855402 finished successfully and the scorer read all 4,081 predictions on 1,547 GT boxes:

- vehicle: 1,009 predictions / 470 GT, AP@0.5 = 0.0000
- pedestrian: 2,240 predictions / 953 GT, AP@0.5 = 0.0000
- truck: 832 predictions / 124 GT, AP@0.5 = 0.0000
- 3D mAP@0.5 R40 = 0.0000
- BEV mAP@0.5 R40 = 0.0000
- Realism score = 0.8755

We checked the following locally:

1. predictions.json has exactly the 50 canonical frame IDs.
2. Class strings are exactly vehicle, pedestrian, and truck.
3. Every released .bin file is read as float32 XYZ (50,000 x 3), with zero-padding removed.
4. Submitted boxes use [x, y, z, dx, dy, dz, heading] in the LiDAR frame.
5. High-confidence predicted boxes are supported by many observed points (for example, the median point count inside high-confidence vehicle boxes is about 140), so the boxes are not simply outside the clouds.
6. The LUMPI and V2X-Real starting kits contain byte-identical score.py, detection_eval.py, metrics.py, and run.sh files, while the public LUMPI leaderboard has a non-zero detection result. This suggests the generic evaluator runs correctly.

Because both 3D and BEV AP are exactly zero for every class, could you please confirm that the V2X-Real detection_gt.json keys and box coordinates correspond to the released detection_test_frames files and their canonical IDs? If there is an additional preprocessing or coordinate transform expected for V2X-Real, please clarify it in the starting kit.

Thank you.
"""


def csrf(text: str) -> str:
    patterns = (
        r'name=["\']csrfmiddlewaretoken["\'][^>]*value=["\']([^"\']+)',
        r'csrfmiddlewaretoken["\']\s+value=["\']([^"\']+)',
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return html.unescape(match.group(1))
    raise SystemExit("CSRF token not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    session = requests.Session()
    login_url = f"{BASE}/accounts/login/"
    bootstrap = session.get(login_url, timeout=30)
    if not bootstrap.ok:
        bootstrap = session.get(f"{BASE}/competitions/16048/", timeout=30)
    bootstrap.raise_for_status()
    token = csrf(bootstrap.text)
    password = getpass.getpass("Codabench password: ")
    login = session.post(
        login_url,
        data={"csrfmiddlewaretoken": token, "username": args.username, "password": password},
        headers={"Referer": bootstrap.url, "X-CSRFToken": token},
        timeout=30,
        allow_redirects=True,
    )
    password = None
    login.raise_for_status()

    forum_url = f"{BASE}/forums/{FORUM_ID}/"
    forum = session.get(forum_url, timeout=30)
    forum.raise_for_status()
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", forum.text))
    if "Log In or Sign Up to view this competition forum" in plain:
        raise SystemExit("Codabench login did not establish an authenticated forum session")

    new_url = f"{forum_url}new_thread/"
    page = session.get(new_url, timeout=30)
    page.raise_for_status()
    forms = re.findall(r"<form\b.*?</form>", page.text, re.I | re.S)
    candidate = next((form for form in forms if "title" in form.lower()), None)
    if candidate is None:
        raise SystemExit("New-thread form not found")
    opening = re.search(r"<form\b[^>]*>", candidate, re.I).group(0)
    action_match = re.search(r"\baction=[\"']([^\"']*)", opening, re.I)
    action = urljoin(new_url, action_match.group(1) if action_match else new_url)
    names = re.findall(
        r"<(?:input|textarea|select)\b[^>]*\bname=[\"']([^\"']+)", candidate, re.I
    )
    title_field = next((name for name in names if name.lower() in {"title", "subject"}), None)
    body_field = next(
        (name for name in names if name.lower() in {"content", "body", "message", "text"}), None
    )
    print(f"New-thread action: {action}")
    print(f"Form fields: {names}")
    print(f"Title: {TITLE}")
    if not args.publish:
        print("Dry run only; pass --publish to create the topic")
        return
    if title_field is None or body_field is None:
        raise SystemExit("Could not identify title/body fields; refusing to publish")

    form_token = csrf(candidate)
    response = session.post(
        action,
        data={"csrfmiddlewaretoken": form_token, title_field: TITLE, body_field: BODY},
        headers={"Referer": new_url, "X-CSRFToken": form_token},
        timeout=30,
        allow_redirects=True,
    )
    response.raise_for_status()
    listing = session.get(forum_url, timeout=30)
    listing.raise_for_status()
    if TITLE not in html.unescape(listing.text):
        raise SystemExit(f"POST returned HTTP {response.status_code}, but the topic title was not found")
    match = re.search(
        rf'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>\s*{re.escape(TITLE)}\s*</a>',
        html.unescape(listing.text),
        re.I,
    )
    print(f"Published: {urljoin(forum_url, match.group(1)) if match else forum_url}")


if __name__ == "__main__":
    main()
