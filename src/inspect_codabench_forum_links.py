#!/usr/bin/env python3
"""Authenticate to Codabench and display relevant forum links."""

from __future__ import annotations

import argparse
import getpass
import html
import re

import requests


BASE = "https://www.codabench.org"


def token(text: str) -> str:
    match = re.search(
        r'name=["\']csrfmiddlewaretoken["\'][^>]*value=["\']([^"\']+)', text
    )
    if match is None:
        raise SystemExit("CSRF token not found")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--forum", type=int, required=True)
    args = parser.parse_args()
    session = requests.Session()
    login_url = f"{BASE}/accounts/login/"
    login = session.get(login_url, timeout=30)
    login.raise_for_status()
    password = getpass.getpass("Codabench password: ")
    response = session.post(
        login_url,
        data={
            "csrfmiddlewaretoken": token(login.text),
            "username": args.username,
            "password": password,
        },
        headers={"Referer": login_url},
        timeout=30,
    )
    password = None
    response.raise_for_status()
    page = session.get(f"{BASE}/forums/{args.forum}/", timeout=30)
    page.raise_for_status()
    for href, body in re.findall(
        r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", page.text, re.I | re.S
    ):
        label = html.unescape(re.sub(r"<[^>]+>", " ", body))
        label = re.sub(r"\s+", " ", label).strip()
        if "forum" in href.lower() or "topic" in label.lower() or "thread" in label.lower():
            print(f"{label}\t{href}")


if __name__ == "__main__":
    main()
