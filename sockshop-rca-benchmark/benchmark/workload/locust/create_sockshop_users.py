#!/usr/bin/env python3
"""
Bulk create Sock Shop users via REST API.

Assumes Sock Shop front-end is at:
    http://sockshop.local:31728/index.html

And the registration API endpoint is:
    POST http://sockshop.local:31728/register

If your endpoint differs, update API_BASE_URL or REGISTER_PATH below.
"""

import random
import string
import sys
from typing import List, Dict

import requests

# Import shared user credentials
from user_credentials import BASE_USERS

API_BASE_URL = "http://sockshop.local:31728"
REGISTER_PATH = "/register"  # change to "/users/register" etc. if needed
TIMEOUT = 5  # seconds


def random_suffix(length: int = 6) -> str:
    """Generate a random alphanumeric suffix, useful to avoid username clashes."""
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def build_user_payload(username: str, password: str) -> Dict[str, str]:
    """
    Build the JSON payload expected by the Sock Shop registration endpoint.
    Adjust keys here if your API is different.
    """
    # Derive some simple fake profile data from username
    base = username.split("_")[0]
    first_name = base.capitalize()
    last_name = "User"

    email = f"{username}@example.com"

    return {
        "username": username,
        "password": password,
        "email": email,
        "firstName": first_name,
        "lastName": last_name,
    }


def create_user(session: requests.Session, user: Dict[str, str]) -> None:
    """Send a POST to create a single user and print the result."""
    url = API_BASE_URL + REGISTER_PATH
    payload = build_user_payload(user["username"], user["password"])

    try:
        resp = session.post(url, json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"[ERROR] {user['username']}: request failed -> {e}", file=sys.stderr)
        return

    # Sock Shop usually returns 201 or 200 on success; 409 if user exists, etc.
    if resp.status_code in (200, 201):
        print(f"[OK]    Created user: {user['username']}")
    else:
        # Print status and body for debugging
        body = resp.text.strip()
        print(
            f"[FAIL] User {user['username']} -> "
            f"HTTP {resp.status_code} | Response: {body}",
            file=sys.stderr,
        )


def main() -> None:
    # If you want to automatically avoid collisions, set this to True.
    AVOID_COLLISIONS = False

    users_to_create = []

    if AVOID_COLLISIONS:
        for u in BASE_USERS:
            # Add a random suffix to each username to avoid "already exists" errors
            uname = f"{u['username']}_{random_suffix(4)}"
            users_to_create.append(
                {"username": uname, "password": u["password"]}
            )
    else:
        users_to_create = BASE_USERS

    print(f"Target API endpoint: {API_BASE_URL}{REGISTER_PATH}")
    print(f"Creating {len(users_to_create)} users...\n")

    with requests.Session() as session:
        for u in users_to_create:
            create_user(session, u)


if __name__ == "__main__":
    main()
