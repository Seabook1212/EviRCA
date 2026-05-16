#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialize Shipping Address and Payment Card information for Sock Shop users.

Logic:
- For each account in BASE_USERS:
  1) GET /login (Basic Auth) to establish session (cookie)
  2) GET /address, if not exists then POST /addresses to create
  3) GET /card, if not exists then POST /cards to create

Default front-end URL: http://sockshop.local:31728
To modify, change BASE_URL or override with environment variable BASE_URL.
"""

import base64
import json
import os
import sys
from typing import List, Dict

import requests

# Import shared user credentials
from user_credentials import BASE_USERS

# Front-end URL (with /login, /addresses, /cards routes)
BASE_URL = os.environ.get("BASE_URL", "http://sockshop.local:31728")

TIMEOUT = 5  # seconds


def basic_auth_value(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def login_user(username: str, password: str) -> requests.Session | None:
    """
    Call /login with Basic Auth, return Session with cookie.
    Returns None if login fails.
    """
    s = requests.Session()
    headers = {"Authorization": basic_auth_value(username, password)}

    try:
        resp = s.get(
            BASE_URL + "/login",
            headers=headers,
            timeout=TIMEOUT,
            allow_redirects=True,
        )
    except requests.RequestException as e:
        print(f"[ERROR] {username}: Request to /login failed -> {e}", file=sys.stderr)
        return None

    if resp.status_code == 200:
        print(f"[OK]    {username}: Login successful")
        return s
    else:
        print(
            f"[FAIL] {username}: Login failed, HTTP {resp.status_code}, body={resp.text[:200]}",
            file=sys.stderr,
        )
        return None


def has_address(session: requests.Session) -> bool:
    """
    Check if the current logged-in user has an address:
    - GET /address
    - If returned JSON has status_code != 500, consider address exists
    """
    try:
        resp = session.get(BASE_URL + "/address", timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"[WARN] Check address failed: {e}", file=sys.stderr)
        return False

    if resp.status_code != 200:
        # Front-end code: Even without address, returns 200 + {"status_code":500}
        print(
            f"[WARN] GET /address non-200: {resp.status_code}, body={resp.text[:200]}",
            file=sys.stderr,
        )
        return False

    try:
        data = resp.json()
    except Exception:
        print(f"[WARN] GET /address returned non-JSON: {resp.text[:200]}", file=sys.stderr)
        return False

    # When no address exists, front-end code returns {"status_code": 500}
    if isinstance(data, dict) and data.get("status_code") == 500:
        return False

    return True


def has_card(session: requests.Session) -> bool:
    """
    Check if the current logged-in user has a card:
    - GET /card
    - If returned JSON has status_code != 500, consider card exists
    """
    try:
        resp = session.get(BASE_URL + "/card", timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"[WARN] Check card failed: {e}", file=sys.stderr)
        return False

    if resp.status_code != 200:
        print(
            f"[WARN] GET /card non-200: {resp.status_code}, body={resp.text[:200]}",
            file=sys.stderr,
        )
        return False

    try:
        data = resp.json()
    except Exception:
        print(f"[WARN] GET /card returned non-JSON: {resp.text[:200]}", file=sys.stderr)
        return False

    if isinstance(data, dict) and data.get("status_code") == 500:
        return False

    return True


def create_address(session: requests.Session) -> bool:
    """
    Create an address for the current logged-in user.
    Front-end /addresses will automatically add userID to body.
    """
    payload = {
        "number": "42",
        "street": "Load Test Street",
        "city": "Bangkok",
        "postcode": "10110",
        "country": "Thailand",
    }

    try:
        resp = session.post(
            BASE_URL + "/addresses",
            json=payload,
            timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        print(f"[ERROR] Create address failed: {e}", file=sys.stderr)
        return False

    if resp.status_code in (200, 201):
        print("      -> Address created successfully")
        return True
    else:
        print(
            f"[FAIL] Create address failed: HTTP {resp.status_code}, body={resp.text[:200]}",
            file=sys.stderr,
        )
        return False


def create_card(session: requests.Session) -> bool:
    """
    Create a credit card for the current logged-in user.
    Front-end /cards will automatically add userID to body.
    """
    payload = {
        "longNum": "4111111111111111",
        "expires": "12/30",
        "ccv": "123",
    }

    try:
        resp = session.post(
            BASE_URL + "/cards",
            json=payload,
            timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        print(f"[ERROR] Create card failed: {e}", file=sys.stderr)
        return False

    if resp.status_code in (200, 201):
        print("      -> Card created successfully")
        return True
    else:
        print(
            f"[FAIL] Create card failed: HTTP {resp.status_code}, body={resp.text[:200]}",
            file=sys.stderr,
        )
        return False


def init_user_profile(username: str, password: str) -> None:
    """
    For a single user:
    - Login
    - If no address exists, create address
    - If no card exists, create card
    """
    print(f"\n=== Processing user {username} ===")
    session = login_user(username, password)
    if session is None:
        return

    # Address
    if has_address(session):
        print("      -> Address already exists, skipping creation")
    else:
        print("      -> No Address detected, preparing to create")
        create_address(session)

    # Card
    if has_card(session):
        print("      -> Card already exists, skipping creation")
    else:
        print("      -> No Card detected, preparing to create")
        create_card(session)


def main():
    print(f"Target front-end BASE_URL = {BASE_URL}")
    for u in BASE_USERS:
        init_user_profile(u["username"], u["password"])


if __name__ == "__main__":
    main()
