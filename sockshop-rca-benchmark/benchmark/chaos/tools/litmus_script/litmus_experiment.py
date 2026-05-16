#!/usr/bin/env python3
import os
import sys
import json
import requests


# ==== Config (you can change these or set env vars) ====
LITMUS_URL = os.getenv("LITMUS_URL", "http://34.28.33.102:32054")
AUTH_ENDPOINT = f"{LITMUS_URL}/auth/login"
GRAPHQL_ENDPOINT = f"{LITMUS_URL}/api/query"

USERNAME = os.getenv("LITMUS_USERNAME", "admin")
PASSWORD = os.getenv("LITMUS_PASSWORD", "Seabook1111_")
EXPERIMENT_ID = os.getenv(
    "LITMUS_EXPERIMENT_ID",
    "f5c79272-a934-4209-a6a1-b3c84985ca89",
)


def login():
    print("==> Login Litmus...")
    try:
        resp = requests.post(
            AUTH_ENDPOINT,
            json={"username": USERNAME, "password": PASSWORD},
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"ERROR: request to auth endpoint failed: {e}")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"ERROR: login HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("ERROR: login response is not valid JSON")
        print(resp.text)
        sys.exit(1)

    access_token = data.get("accessToken")
    project_id = data.get("projectID")

    if not access_token or not project_id:
        print("ERROR: login succeeded but accessToken or projectID missing")
        print(json.dumps(data, indent=2))
        sys.exit(1)

    print(f"ProjectID={project_id}")
    return access_token, project_id


def trigger_experiment(access_token, project_id, experiment_id):
    print(f"==> Trigger Chaos Experiment {experiment_id} ...")

    query = (
        "mutation RunChaosExperiment($eid: String!, $pid: ID!) { "
        "runChaosExperiment(experimentID: $eid, projectID: $pid) { notifyID } }"
    )

    payload = {
        "query": query,
        "variables": {
            "eid": experiment_id,
            "pid": project_id,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    try:
        resp = requests.post(
            GRAPHQL_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"ERROR: request to GraphQL endpoint failed: {e}")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"ERROR: GraphQL HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    print("Response:", resp.text)

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("ERROR: GraphQL response is not valid JSON")
        sys.exit(1)

    if "errors" in data:
        print("ERROR: GraphQL errors:")
        print(json.dumps(data["errors"], indent=2))
        sys.exit(1)

    notify_id = (
        data.get("data", {})
        .get("runChaosExperiment", {})
        .get("notifyID")
    )

    if not notify_id:
        print("WARN: no notifyID found in response")
    else:
        print(f"notifyID={notify_id}")


def main():
    access_token, project_id = login()
    trigger_experiment(access_token, project_id, EXPERIMENT_ID)


if __name__ == "__main__":
    main()
