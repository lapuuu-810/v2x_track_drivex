#!/usr/bin/env python3
"""Authenticate to Codabench and inspect a new-thread form."""

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
    response = session.post(
        login_url,
        data={"csrfmiddlewaretoken": csrf(login.text), "username": args.username, "password": password},
        headers={"Referer": login_url},
        timeout=30,
    )
    password = None
    response.raise_for_status()
    url = f"{BASE}/forums/{args.forum}/new_thread/"
    page = session.get(url, timeout=30)
    page.raise_for_status()
    print(f"url={url}")
    for index, form in enumerate(re.findall(r"<form\b.*?</form>", page.text, re.I | re.S)):
        opening = re.search(r"<form\b[^>]*>", form, re.I).group(0)
        fields = []
        for tag in re.findall(r"<(?:input|textarea|select)\b[^>]*>", form, re.I):
            name = re.search(r"\bname=[\"']([^\"']+)", tag, re.I)
            kind = re.search(r"\btype=[\"']([^\"']+)", tag, re.I)
            fields.append((name.group(1) if name else None, kind.group(1) if kind else tag.split()[0][1:]))
        print(f"form[{index}]={opening}")
        print(f"fields[{index}]={fields}")


if __name__ == "__main__":
    main()
