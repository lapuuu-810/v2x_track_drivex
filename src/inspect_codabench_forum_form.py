#!/usr/bin/env python3
"""Authenticate to Codabench and display forum form metadata."""

from __future__ import annotations

import argparse
import getpass
import re

import requests


BASE = "https://www.codabench.org"


def csrf(text: str) -> str:
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
    result = session.post(
        login_url,
        data={
            "csrfmiddlewaretoken": csrf(login.text),
            "username": args.username,
            "password": password,
        },
        headers={"Referer": login_url},
        timeout=30,
    )
    password = None
    result.raise_for_status()

    forum_url = f"{BASE}/forums/{args.forum}/"
    page = session.get(forum_url, timeout=30)
    page.raise_for_status()
    print(f"forum_url={forum_url}")
    for index, form in enumerate(re.findall(r"<form\b.*?</form>", page.text, re.I | re.S)):
        opening = re.search(r"<form\b[^>]*>", form, re.I).group(0)
        names = re.findall(r"<(?:input|textarea|select)\b[^>]*\bname=[\"']([^\"']+)", form, re.I)
        print(f"form[{index}]={opening}")
        print(f"fields[{index}]={names}")


if __name__ == "__main__":
    main()
