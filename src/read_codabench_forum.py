#!/usr/bin/env python3
"""Read a Codabench competition forum using an authenticated web session."""

from __future__ import annotations

import argparse
import getpass
import html
import re

import requests


BASE = "https://www.codabench.org"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--forum", type=int, required=True)
    args = parser.parse_args()

    session = requests.Session()
    login_url = f"{BASE}/accounts/login/"
    login_page = session.get(login_url, timeout=30)
    login_page.raise_for_status()
    match = re.search(
        r'name=["\']csrfmiddlewaretoken["\'][^>]*value=["\']([^"\']+)',
        login_page.text,
    )
    if match is None:
        raise SystemExit("Could not find the Codabench login CSRF token")

    password = getpass.getpass("Codabench password: ")
    response = session.post(
        login_url,
        data={
            "csrfmiddlewaretoken": match.group(1),
            "login": args.username,
            "password": password,
            "next": f"/forums/{args.forum}/",
        },
        headers={"Referer": login_url},
        timeout=30,
        allow_redirects=True,
    )
    password = None
    response.raise_for_status()
    forum_url = f"{BASE}/forums/{args.forum}/"
    page = session.get(forum_url, timeout=30)
    page.raise_for_status()
    if "Log In or Sign Up to view this competition forum" in page.text:
        raise SystemExit("Codabench web login did not establish a forum session")

    # Keep table rows and links readable while stripping unrelated page chrome.
    body_match = re.search(r"<body[^>]*>(.*)</body>", page.text, re.I | re.S)
    body = body_match.group(1) if body_match else page.text
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.I | re.S)
    body = re.sub(r"<br\s*/?>|</p>|</tr>|</h[1-6]>", "\n", body, flags=re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    body = html.unescape(body)
    lines = [re.sub(r"\s+", " ", line).strip() for line in body.splitlines()]
    print("\n".join(line for line in lines if line))


if __name__ == "__main__":
    main()
