#!/usr/bin/env python3
"""Publish the authorized V2X-Real forum issue with Codabench's redirect field."""

from __future__ import annotations

import argparse
import getpass
import html
import re
from urllib.parse import urljoin

import requests

from post_v2xreal_forum_issue import BASE, BODY, FORUM_ID, TITLE, csrf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    session = requests.Session()
    login_url = f"{BASE}/accounts/login/"
    forum_path = f"/forums/{FORUM_ID}/"
    bootstrap = session.get(login_url, timeout=30)
    bootstrap.raise_for_status()
    token = csrf(bootstrap.text)
    password = getpass.getpass("Codabench password: ")
    login = session.post(
        login_url,
        data={
            "csrfmiddlewaretoken": token,
            "username": args.username,
            "password": password,
            "next": forum_path,
        },
        headers={"Referer": login_url, "X-CSRFToken": token},
        timeout=30,
        allow_redirects=True,
    )
    password = None
    login.raise_for_status()
    forum_url = urljoin(BASE, forum_path)
    forum = session.get(forum_url, timeout=30)
    forum.raise_for_status()
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", forum.text))
    if "Log In or Sign Up to view this competition forum" in plain:
        raise SystemExit("Codabench login did not establish an authenticated forum session")

    new_url = urljoin(forum_url, "new_thread/")
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
    decoded = html.unescape(listing.text)
    if TITLE not in decoded:
        raise SystemExit(f"POST returned HTTP {response.status_code}, but topic title was not found")
    match = re.search(
        rf'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>\s*{re.escape(TITLE)}\s*</a>',
        decoded,
        re.I,
    )
    print(f"Published: {urljoin(forum_url, match.group(1)) if match else forum_url}")


if __name__ == "__main__":
    main()
