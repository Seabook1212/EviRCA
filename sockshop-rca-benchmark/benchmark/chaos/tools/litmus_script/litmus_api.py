#!/usr/bin/env python3
"""
Litmus API Client for fetching chaos experiment information.

This module provides functions to:
1. Authenticate with Litmus
2. Fetch experiment run details including fault type, timing, and injection parameters
"""

import os
import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timezone


# ==== Config ====
LITMUS_URL = os.getenv("LITMUS_URL", "http://34.28.33.102:32054")
AUTH_ENDPOINT = f"{LITMUS_URL}/auth/login"
GRAPHQL_ENDPOINT = f"{LITMUS_URL}/api/query"

USERNAME = os.getenv("LITMUS_USERNAME", "admin")
PASSWORD = os.getenv("LITMUS_PASSWORD", "Seabook1111_")


def login() -> tuple:
    """
    Login to Litmus and return access token and project ID.

    Returns:
        tuple: (access_token, project_id)
    """
    try:
        resp = requests.post(
            AUTH_ENDPOINT,
            json={"username": USERNAME, "password": PASSWORD},
            timeout=10,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to connect to Litmus auth endpoint: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"Login failed with HTTP {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except json.JSONDecodeError:
        raise RuntimeError(f"Login response is not valid JSON: {resp.text}")

    access_token = data.get("accessToken")
    project_id = data.get("projectID")

    if not access_token or not project_id:
        raise RuntimeError(f"Login succeeded but accessToken or projectID missing: {data}")

    return access_token, project_id


def graphql_query(access_token: str, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a GraphQL query against Litmus API.

    Args:
        access_token: Bearer token for authentication
        query: GraphQL query string
        variables: Query variables

    Returns:
        dict: Response data
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    payload = {
        "query": query,
        "variables": variables,
    }

    try:
        resp = requests.post(
            GRAPHQL_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"GraphQL request failed: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"GraphQL HTTP {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except json.JSONDecodeError:
        raise RuntimeError(f"GraphQL response is not valid JSON: {resp.text}")

    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {json.dumps(data['errors'], indent=2)}")

    return data.get("data", {})


def get_experiment_runs(access_token: str, project_id: str, experiment_id: str, limit: int = 1) -> list:
    """
    Get recent experiment runs for a given experiment ID.

    Args:
        access_token: Bearer token
        project_id: Litmus project ID
        experiment_id: The chaos experiment ID
        limit: Number of runs to fetch (default: 1 for latest)

    Returns:
        list: List of experiment run details
    """
    query = """
    query ListExperimentRun($projectID: ID!, $request: ListExperimentRunRequest!) {
        listExperimentRun(projectID: $projectID, request: $request) {
            totalNoOfExperimentRuns
            experimentRuns {
                experimentRunID
                experimentID
                experimentName
                phase
                resiliencyScore
                updatedAt
                createdAt
                runSequence
                infra {
                    infraID
                    name
                    infraNamespace
                }
                executionData
            }
        }
    }
    """

    variables = {
        "projectID": project_id,
        "request": {
            "experimentIDs": [experiment_id],
            "pagination": {
                "page": 0,
                "limit": limit
            }
        }
    }

    data = graphql_query(access_token, query, variables)
    return data.get("listExperimentRun", {}).get("experimentRuns", [])


def get_experiment_details(access_token: str, project_id: str, experiment_id: str) -> Dict[str, Any]:
    """
    Get experiment definition details.

    Args:
        access_token: Bearer token
        project_id: Litmus project ID
        experiment_id: The chaos experiment ID

    Returns:
        dict: Experiment details including manifest
    """
    query = """
    query GetExperiment($projectID: ID!, $experimentID: String!) {
        getExperiment(projectID: $projectID, experimentID: $experimentID) {
            experimentDetails {
                experimentID
                name
                description
                cronSyntax
                isCustomExperiment
                updatedAt
                createdAt
                infra {
                    infraID
                    name
                    infraNamespace
                }
                experimentManifest
            }
        }
    }
    """

    variables = {
        "projectID": project_id,
        "experimentID": experiment_id,
    }

    data = graphql_query(access_token, query, variables)
    return data.get("getExperiment", {}).get("experimentDetails", {})


def parse_execution_data(execution_data_str: str) -> Dict[str, Any]:
    """
    Parse the executionData JSON string from experiment run.

    Args:
        execution_data_str: JSON string containing execution details

    Returns:
        dict: Parsed execution data
    """
    if not execution_data_str:
        return {}

    try:
        return json.loads(execution_data_str)
    except json.JSONDecodeError:
        return {}


def extract_fault_info_from_manifest(manifest_str: str) -> Dict[str, Any]:
    """
    Extract fault information from experiment manifest YAML/JSON.

    Args:
        manifest_str: The experiment manifest string

    Returns:
        dict: Extracted fault information
    """
    import yaml

    if not manifest_str:
        return {}

    try:
        # Try parsing as JSON first
        manifest = json.loads(manifest_str)
    except json.JSONDecodeError:
        # If not JSON, try YAML
        try:
            manifest = yaml.safe_load(manifest_str)
        except Exception:
            return {}

    fault_info = {}

    # Navigate through the manifest structure to find chaos experiment details
    # Litmus manifests typically have templates with chaos experiments
    if isinstance(manifest, dict):
        templates = manifest.get("spec", {}).get("templates", [])

        for template in templates:
            if not isinstance(template, dict):
                continue

            template_name = template.get("name", "")

            # Look for inputs with artifacts containing chaos experiment
            inputs = template.get("inputs", {})
            artifacts = inputs.get("artifacts", [])

            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue

                raw_data = artifact.get("raw", {}).get("data", "")
                if raw_data:
                    try:
                        chaos_data = yaml.safe_load(raw_data)
                        if isinstance(chaos_data, dict):
                            # Extract ChaosEngine or ChaosExperiment details
                            kind = chaos_data.get("kind", "")

                            # Debug output
                            print(f"DEBUG: Found artifact in template '{template_name}', kind='{kind}'")

                            if kind == "ChaosEngine":
                                spec = chaos_data.get("spec", {})

                                # Get app info
                                app_info = spec.get("appinfo", {})
                                fault_info["app_namespace"] = app_info.get("appns", "")
                                fault_info["app_label"] = app_info.get("applabel", "")
                                fault_info["app_kind"] = app_info.get("appkind", "")

                                # Get experiments
                                experiments = spec.get("experiments", [])
                                for exp in experiments:
                                    if isinstance(exp, dict):
                                        fault_info["fault_name"] = exp.get("name", "")

                                        # Get experiment spec
                                        exp_spec = exp.get("spec", {})
                                        components = exp_spec.get("components", {})
                                        env_vars = components.get("env", [])

                                        # Extract environment variables (injection parameters)
                                        for env in env_vars:
                                            if isinstance(env, dict):
                                                name = env.get("name", "")
                                                value = env.get("value", "")
                                                if name and value:
                                                    fault_info[name.lower()] = value

                                # Debug: print extracted fault_info
                                print(f"DEBUG: Extracted fault_info from ChaosEngine: {json.dumps(fault_info, indent=2)}")

                    except Exception as e:
                        print(f"DEBUG: Failed to parse artifact data: {e}")
                        continue

    return fault_info


def extract_chaos_result_from_execution(execution_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract chaos result information from execution data.

    Args:
        execution_data: Parsed execution data

    Returns:
        dict: Chaos result information including timing and status
    """
    result = {}

    nodes = execution_data.get("nodes", {})

    for node_id, node_data in nodes.items():
        if not isinstance(node_data, dict):
            continue

        node_type = node_data.get("type", "")
        node_name = node_data.get("name", "")

        # Look for ChaosEngine nodes - they contain the actual chaos experiment data
        if node_type == "ChaosEngine":
            # Debug: Print full ChaosEngine node data
            print("\n" + "=" * 60)
            print(f"DEBUG: ChaosEngine node '{node_name}' full data:")
            print("=" * 60)
            print(json.dumps(node_data, indent=2, default=str))
            print("=" * 60 + "\n")

            result["node_name"] = node_name
            result["phase"] = node_data.get("phase", "")
            result["started_at"] = node_data.get("startedAt", "")
            result["finished_at"] = node_data.get("finishedAt", "")

            # Extract chaosData which contains experiment details and targets
            chaos_data = node_data.get("chaosData", {})
            if chaos_data:
                result["experiment_name"] = chaos_data.get("experimentName", "")
                result["experiment_status"] = chaos_data.get("experimentStatus", "")
                result["experiment_verdict"] = chaos_data.get("experimentVerdict", "")
                result["probe_success_percentage"] = chaos_data.get("probeSuccessPercentage", "")

                # Extract target info from chaosResult
                chaos_result = chaos_data.get("chaosResult", {})
                if chaos_result:
                    status = chaos_result.get("status", {})
                    history = status.get("history", {})
                    targets = history.get("targets", [])
                    if targets:
                        result["targets"] = targets
                        # Get the first target pod name
                        result["target_pod"] = targets[0].get("name", "") if targets else ""

            # Get chaos result from outputs
            outputs = node_data.get("outputs", {})
            if outputs:
                result["outputs"] = outputs

            # Check for chaosData in message or outputs
            message = node_data.get("message", "")
            if message:
                result["message"] = message

            # Found ChaosEngine node, no need to continue
            break

    # Also get overall timing
    result["workflow_started_at"] = execution_data.get("startedAt", "")
    result["workflow_finished_at"] = execution_data.get("finishedAt", "")

    return result


def get_latest_experiment_info(experiment_id: str) -> Dict[str, Any]:
    """
    Get comprehensive information about the latest run of an experiment.

    Args:
        experiment_id: The Litmus experiment ID

    Returns:
        dict: Complete experiment information including:
            - experiment_name
            - fault_type
            - target_namespace
            - target_pod/app
            - injection_start
            - injection_end
            - injection_parameters (latency, loss, cpu%, etc.)
            - phase/status
    """
    # Login
    access_token, project_id = login()

    # Get experiment details (for manifest)
    experiment_details = get_experiment_details(access_token, project_id, experiment_id)

    # Debug: Print experiment manifest
    manifest_str = experiment_details.get("experimentManifest", "")
    if manifest_str:
        print("\n" + "=" * 60)
        print("DEBUG: Experiment Manifest:")
        print("=" * 60)
        try:
            import yaml
            manifest = yaml.safe_load(manifest_str)
            print(json.dumps(manifest, indent=2, default=str))
        except Exception as e:
            print(f"Failed to parse manifest: {e}")
            print(manifest_str[:2000] + "..." if len(manifest_str) > 2000 else manifest_str)
        print("=" * 60 + "\n")

    # Get latest experiment run
    runs = get_experiment_runs(access_token, project_id, experiment_id, limit=1)

    if not runs:
        return {
            "error": "No experiment runs found",
            "experiment_id": experiment_id
        }

    latest_run = runs[0]

    # Debug: Print latest_run data
    print("\n" + "=" * 60)
    print("DEBUG: latest_run data from Litmus API:")
    print("=" * 60)
    # Print without executionData to keep it readable (executionData is very large)
    debug_run = {k: v for k, v in latest_run.items() if k != "executionData"}
    print(json.dumps(debug_run, indent=2, default=str))
    print("=" * 60 + "\n")

    # Parse execution data
    execution_data = parse_execution_data(latest_run.get("executionData", ""))

    # Extract fault info from manifest
    manifest_str = experiment_details.get("experimentManifest", "")
    fault_info = extract_fault_info_from_manifest(manifest_str)

    # Extract chaos result from execution
    chaos_result = extract_chaos_result_from_execution(execution_data)

    # Compile comprehensive info
    info = {
        "experiment_id": experiment_id,
        "experiment_name": latest_run.get("experimentName", experiment_details.get("name", "")),
        "experiment_run_id": latest_run.get("experimentRunID", ""),
        "phase": latest_run.get("phase", ""),
        "resiliency_score": latest_run.get("resiliencyScore"),
        "created_at": latest_run.get("createdAt", ""),
        "updated_at": latest_run.get("updatedAt", ""),
        "infra": latest_run.get("infra", {}),
        "fault_info": fault_info,
        "chaos_result": chaos_result,
        "execution_data": execution_data,  # Include full execution data for debugging
    }

    return info


def _convert_unix_timestamp(ts: str) -> str:
    """
    Convert Unix timestamp (seconds or milliseconds) to ISO format.

    Args:
        ts: Unix timestamp string (seconds or milliseconds)

    Returns:
        str: ISO formatted timestamp or original string if conversion fails
    """
    if not ts:
        return ""

    # If already in ISO format, return as-is
    if "T" in ts or "-" in ts:
        return ts

    try:
        ts_num = int(ts)
        # If timestamp is in milliseconds (> year 2100 in seconds), convert to seconds
        if ts_num > 4102444800:  # Year 2100 in seconds
            ts_num = ts_num // 1000
        dt = datetime.fromtimestamp(ts_num, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError, OSError):
        return ts


def format_fault_metadata(experiment_info: Dict[str, Any], exp_id: str) -> Dict[str, Any]:
    """
    Format experiment info into fault_metadata.json structure.

    Args:
        experiment_info: Raw experiment info from get_latest_experiment_info
        exp_id: The experiment ID used in the dataset (e.g., "pod_cpu_hog_user_001")

    Returns:
        dict: Formatted fault metadata
    """
    fault_info = experiment_info.get("fault_info", {})
    chaos_result = experiment_info.get("chaos_result", {})

    # Determine fault type from chaos_result experiment_name (e.g., "pod-cpu-hog")
    # or from experiment_info experiment_name
    fault_type = chaos_result.get("experiment_name", "") or fault_info.get("fault_name", "")
    if not fault_type:
        # Fall back to experiment name, but try to extract just the fault part
        experiment_name = experiment_info.get("experiment_name", "")
        fault_type = experiment_name

    # Extract target service from multiple sources in priority order:
    # 1. From target_pod in chaos_result (e.g., "carts-db-0" -> "carts-db")
    # 2. From app_label in fault_info
    # 3. From exp_id pattern matching
    target_service = ""

    # Try to get from target_pod first
    target_pod = chaos_result.get("target_pod", "")
    if target_pod:
        # Extract service name from pod name
        # StatefulSet pods: "carts-db-0" -> "carts-db"
        # Deployment pods: "user-abc123-xyz789" -> "user"
        known_services = ["carts", "catalogue", "user", "orders", "payment", "shipping",
                         "front-end", "queue-master", "rabbitmq", "session-db",
                         "carts-db", "user-db", "orders-db", "catalogue-db"]

        # Check if pod name starts with a known service
        for svc in sorted(known_services, key=len, reverse=True):  # Check longer names first
            if target_pod.startswith(svc):
                target_service = svc
                break

    # Try app_label if target_service not found
    if not target_service:
        app_label = fault_info.get("app_label", "")
        if app_label:
            # Parse "app=carts" or "name=user" format
            if "=" in app_label:
                target_service = app_label.split("=")[-1]
            else:
                target_service = app_label

    # Fall back to exp_id pattern matching
    if not target_service:
        parts = exp_id.replace("-", "_").split("_")
        known_services = ["carts", "catalogue", "user", "orders", "payment", "shipping",
                         "front-end", "queue-master", "rabbitmq", "session-db",
                         "carts-db", "user-db", "orders-db", "catalogue-db"]
        for part in parts:
            if part.lower() in known_services or part.lower().replace("-", "") in [s.replace("-", "") for s in known_services]:
                target_service = part.lower()
                break

    # Get timing information and convert to ISO format
    inject_start = chaos_result.get("started_at") or chaos_result.get("workflow_started_at", "")
    inject_end = chaos_result.get("finished_at") or chaos_result.get("workflow_finished_at", "")

    # If no timing from chaos result, use created_at/updated_at
    if not inject_start:
        inject_start = experiment_info.get("created_at", "")
    if not inject_end:
        inject_end = experiment_info.get("updated_at", "")

    # Convert Unix timestamps to ISO format
    inject_start = _convert_unix_timestamp(inject_start)
    inject_end = _convert_unix_timestamp(inject_end)

    # Build injection_info with essential parameters only
    injection_info = {
        "fault_type": fault_type,
        "inject_start": inject_start,
        "inject_end": inject_end,
    }

    # Add injection parameters (environment variables from chaos experiment)
    param_keys = [
        "total_chaos_duration", "chaos_interval",
        # CPU-related
        "cpu_load", "cpu_load_fraction", "node_cpu_core",
        # Memory-related
        "memory_consumption", "memory_fill_fraction",
        # Network-related
        "network_latency", "network_loss", "network_packet_loss_percentage",
        "network_packet_corruption_percentage", "latency",
        # Chaos Monkey
        "cm_level", "cm_exceptions_type", "cm_exceptions_arguments",
        # Execution
        "ramp_time", "pods_affected_perc", "number_of_workers",
    ]

    for key in param_keys:
        if key in fault_info:
            injection_info[key] = fault_info[key]

    metadata = {
        "fault_id": exp_id,
        "target_service": target_service,
        "injection_tool": "litmus",
        "injection_info": injection_info,
    }

    return metadata


# For direct testing
if __name__ == "__main__":
    import sys

    experiment_id = os.getenv("LITMUS_EXPERIMENT_ID", "30ecda3e-423c-41bf-bb29-64944f208a5b")
    if len(sys.argv) > 1:
        experiment_id = sys.argv[1]

    if not experiment_id:
        print("Usage: python litmus_api.py <experiment_id>")
        print("Or set LITMUS_EXPERIMENT_ID environment variable")
        sys.exit(1)

    print(f"Fetching info for experiment: {experiment_id}")
    print("=" * 60)

    try:
        info = get_latest_experiment_info(experiment_id)
        print(json.dumps(info, indent=2, default=str))

        print("\n" + "=" * 60)
        print("Formatted fault metadata:")
        print("=" * 60)

        metadata = format_fault_metadata(info, "test_experiment_001")
        print(json.dumps(metadata, indent=2))

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
