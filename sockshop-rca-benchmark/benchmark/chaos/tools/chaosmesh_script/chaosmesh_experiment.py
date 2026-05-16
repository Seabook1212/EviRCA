#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChaosMesh Experiment Trigger Script

This script triggers a ChaosMesh experiment by:
1. Reading the YAML file to extract kind and metadata.name
2. Deleting any existing chaos resource with the same name
3. Applying the chaos YAML to create a new experiment

The kubectl commands are executed remotely on a GCP VM via gcloud compute ssh.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path


# ===== Configuration =====
# GCE instance configuration for remote kubectl execution
INSTANCE_NAME = os.getenv("GCE_INSTANCE_NAME", "instance-20250929-043614")
ZONE = os.getenv("GCE_ZONE", "us-central1-c")
REMOTE_USER = os.getenv("GCE_REMOTE_USER", "g6878310023")
REMOTE_KUBECONFIG = os.getenv("REMOTE_KUBECONFIG", f"/home/{REMOTE_USER}/.kube/config")

# gcloud/ssh settings (configurable)
SSH_TIMEOUT_SECONDS = int(os.getenv("CHAOSMESH_SSH_TIMEOUT_SECONDS", "180"))
SSH_MAX_RETRIES = int(os.getenv("CHAOSMESH_SSH_MAX_RETRIES", "3"))
SSH_RETRY_INTERVAL_SECONDS = int(os.getenv("CHAOSMESH_SSH_RETRY_INTERVAL_SECONDS", "5"))
SCP_TIMEOUT_SECONDS = int(os.getenv("CHAOSMESH_SCP_TIMEOUT_SECONDS", "180"))
SCP_MAX_RETRIES = int(os.getenv("CHAOSMESH_SCP_MAX_RETRIES", "3"))
SCP_RETRY_INTERVAL_SECONDS = int(os.getenv("CHAOSMESH_SCP_RETRY_INTERVAL_SECONDS", "5"))
GCLOUD_QUIET = os.getenv("GCLOUD_QUIET", "true").lower() in {"1", "true", "yes"}
SSH_CONNECT_TIMEOUT_SECONDS = int(os.getenv("CHAOSMESH_SSH_CONNECT_TIMEOUT_SECONDS", "15"))
SSH_SERVER_ALIVE_INTERVAL_SECONDS = int(os.getenv("CHAOSMESH_SSH_SERVER_ALIVE_INTERVAL_SECONDS", "10"))
SSH_SERVER_ALIVE_COUNT_MAX = int(os.getenv("CHAOSMESH_SSH_SERVER_ALIVE_COUNT_MAX", "2"))
KUBECTL_REQUEST_TIMEOUT_SECONDS = int(os.getenv("CHAOSMESH_KUBECTL_REQUEST_TIMEOUT_SECONDS", "30"))
DELETE_WAIT_TIMEOUT_SECONDS = int(os.getenv("CHAOSMESH_DELETE_WAIT_TIMEOUT_SECONDS", "90"))

# ChaosMesh YAML file path (can be overridden by environment variable)
SCRIPT_DIR = Path(__file__).parent
DEFAULT_YAML_FILE = SCRIPT_DIR.parent / "chaosmesh_yaml_v3" / "pod_io_fault_carts-db_001.yaml"
CHAOSMESH_YAML_FILE = os.getenv("CHAOSMESH_YAML_FILE", str(DEFAULT_YAML_FILE))

# Kubernetes namespace (default from YAML, can be overridden)
DEFAULT_NAMESPACE = os.getenv("CHAOSMESH_NAMESPACE", "sock-shop")


def _gcloud_base_command() -> list:
    cmd = ["gcloud"]
    if GCLOUD_QUIET:
        cmd.append("--quiet")
    return cmd


def _gcloud_ssh_flags() -> list:
    return [
        "--ssh-flag=-oBatchMode=yes",
        f"--ssh-flag=-oConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
        f"--ssh-flag=-oServerAliveInterval={SSH_SERVER_ALIVE_INTERVAL_SECONDS}",
        f"--ssh-flag=-oServerAliveCountMax={SSH_SERVER_ALIVE_COUNT_MAX}",
    ]


def _gcloud_scp_flags() -> list:
    return [
        "--scp-flag=-oBatchMode=yes",
        f"--scp-flag=-oConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
        f"--scp-flag=-oServerAliveInterval={SSH_SERVER_ALIVE_INTERVAL_SECONDS}",
        f"--scp-flag=-oServerAliveCountMax={SSH_SERVER_ALIVE_COUNT_MAX}",
    ]


def _kubectl_prefix() -> str:
    if KUBECTL_REQUEST_TIMEOUT_SECONDS <= 0:
        return "kubectl"
    return f"kubectl --request-timeout={KUBECTL_REQUEST_TIMEOUT_SECONDS}s"


def _is_ssh_transport_error(returncode: int, stderr: str) -> bool:
    if returncode == 255:
        return True
    lowered = (stderr or "").lower()
    return any(
        marker in lowered
        for marker in [
            "connection closed",
            "connection timed out",
            "operation timed out",
            "broken pipe",
            "connection reset",
            "exited with return code [255]",
            "gcloud crashed",
            "sslerror",
            "unexpected_eof",
            "max retries exceeded",
            "compute.googleapis.com",
        ]
    )


def load_yaml_file(yaml_path: str) -> dict:
    """
    Load and parse the ChaosMesh YAML file.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        dict: Parsed YAML content
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"ERROR: YAML file not found: {yaml_path}")
        sys.exit(1)

    print(f"==> Loading YAML file: {yaml_path}")
    with open(yaml_path, 'r') as f:
        content = yaml.safe_load(f)

    return content


def extract_chaos_info(yaml_content: dict) -> tuple:
    """
    Extract kind, name, and namespace from the YAML content.

    Args:
        yaml_content: Parsed YAML content

    Returns:
        tuple: (kind, name, namespace)
    """
    kind = yaml_content.get("kind", "")
    metadata = yaml_content.get("metadata", {})
    name = metadata.get("name", "")
    namespace = metadata.get("namespace", DEFAULT_NAMESPACE)

    if not kind:
        print("ERROR: 'kind' not found in YAML file")
        sys.exit(1)

    if not name:
        print("ERROR: 'metadata.name' not found in YAML file")
        sys.exit(1)

    print(f"  Kind: {kind}")
    print(f"  Name: {name}")
    print(f"  Namespace: {namespace}")

    return kind, name, namespace


def run_remote_kubectl(command: str) -> tuple:
    """
    Execute kubectl command remotely on GCP VM via gcloud compute ssh.

    Args:
        command: The kubectl command to execute

    Returns:
        tuple: (return_code, stdout, stderr)
    """
    # Build the gcloud ssh command with KUBECONFIG
    remote = f"{REMOTE_USER}@{INSTANCE_NAME}"
    remote_cmd = f"KUBECONFIG={REMOTE_KUBECONFIG} {command}"

    ssh_command = _gcloud_base_command() + [
        "compute", "ssh", remote,
        "--zone", ZONE,
    ] + _gcloud_ssh_flags() + [
        "--command", remote_cmd,
    ]

    print(f"==> Executing remote command: {command}")

    for attempt in range(1, max(1, SSH_MAX_RETRIES) + 1):
        print(f"  SSH attempt {attempt}/{max(1, SSH_MAX_RETRIES)}")
        try:
            result = subprocess.run(
                ssh_command,
                capture_output=True,
                text=True,
                timeout=SSH_TIMEOUT_SECONDS
            )
            if result.returncode == 0 or not _is_ssh_transport_error(result.returncode, result.stderr):
                return result.returncode, result.stdout, result.stderr

            print(f"WARN: SSH transport failed (return code={result.returncode})")
            if result.stderr and result.stderr.strip():
                print(f"  STDERR: {result.stderr.strip()}")
        except subprocess.TimeoutExpired as e:
            print(f"ERROR: Command timed out after {SSH_TIMEOUT_SECONDS} seconds")
            if e.stdout:
                out = e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout)
                print(f"  STDOUT(before timeout): {out.strip()}")
            if e.stderr:
                err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
                print(f"  STDERR(before timeout): {err.strip()}")
            result = None
        except Exception as e:
            print(f"ERROR: Failed to execute command: {e}")
            return -1, "", str(e)

        if attempt < max(1, SSH_MAX_RETRIES):
            print(f"  Retrying in {SSH_RETRY_INTERVAL_SECONDS} seconds...")
            import time
            time.sleep(max(0, SSH_RETRY_INTERVAL_SECONDS))

    return -1, "", "SSH transport failed after retries"


def copy_yaml_to_vm(yaml_path: str, remote_path: str) -> bool:
    """
    Copy the YAML file to the GCP VM using gcloud compute scp.

    Args:
        yaml_path: Local path to the YAML file
        remote_path: Remote path on the VM

    Returns:
        bool: True if successful, False otherwise
    """
    remote = f"{REMOTE_USER}@{INSTANCE_NAME}"
    scp_command = _gcloud_base_command() + [
        "compute", "scp",
        yaml_path,
        f"{remote}:{remote_path}",
        "--zone", ZONE,
    ] + _gcloud_scp_flags()

    print(f"==> Copying YAML file to VM: {remote_path}")

    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        print(f"ERROR: Local YAML file not found: {yaml_file}")
        return False

    for attempt in range(1, max(1, SCP_MAX_RETRIES) + 1):
        print(f"  SCP attempt {attempt}/{max(1, SCP_MAX_RETRIES)}")
        try:
            result = subprocess.run(
                scp_command,
                capture_output=True,
                text=True,
                timeout=SCP_TIMEOUT_SECONDS
            )
            if result.returncode == 0:
                return True

            print(f"ERROR: Failed to copy file (return code={result.returncode})")
            if result.stdout and result.stdout.strip():
                print(f"  STDOUT: {result.stdout.strip()}")
            if result.stderr and result.stderr.strip():
                print(f"  STDERR: {result.stderr.strip()}")
        except subprocess.TimeoutExpired as e:
            print(f"ERROR: SCP command timed out after {SCP_TIMEOUT_SECONDS} seconds")
            if e.stdout:
                out = e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout)
                print(f"  STDOUT(before timeout): {out.strip()}")
            if e.stderr:
                err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
                print(f"  STDERR(before timeout): {err.strip()}")
        except Exception as e:
            print(f"ERROR: Failed to copy file: {e}")

        if attempt < max(1, SCP_MAX_RETRIES):
            print(f"  Retrying in {SCP_RETRY_INTERVAL_SECONDS} seconds...")
            import time
            time.sleep(max(0, SCP_RETRY_INTERVAL_SECONDS))

    return False


def delete_chaos_resource(kind: str, name: str, namespace: str) -> bool:
    """
    Delete existing chaos resource if it exists.

    Args:
        kind: The Kubernetes resource kind (e.g., HTTPChaos, JVMChaos)
        name: The resource name
        namespace: The namespace

    Returns:
        bool: True if successful (including when resource doesn't exist)
    """
    # Convert kind to lowercase for kubectl (e.g., HTTPChaos -> httpchaos)
    kind_lower = kind.lower()

    command = (
        f"{_kubectl_prefix()} delete {kind_lower} {name} -n {namespace} "
        f"--ignore-not-found --wait=true --timeout={DELETE_WAIT_TIMEOUT_SECONDS}s"
    )

    print(f"==> Deleting existing chaos resource...")
    returncode, stdout, stderr = run_remote_kubectl(command)

    if returncode == 0:
        if stdout.strip():
            print(f"  {stdout.strip()}")
        else:
            print(f"  No existing resource found (or deleted successfully)")
        return True
    else:
        print(f"ERROR: Failed to delete resource")
        if stderr:
            print(f"  STDERR: {stderr}")
        return False


def apply_chaos_yaml(yaml_path: str, kind: str = "", name: str = "", namespace: str = "") -> bool:
    """
    Apply the chaos YAML file to create the experiment.

    Args:
        yaml_path: Path to the YAML file (on the VM)

    Returns:
        bool: True if successful
    """
    command = f"{_kubectl_prefix()} apply -f {yaml_path}"

    print(f"==> Applying chaos experiment...")
    returncode, stdout, stderr = run_remote_kubectl(command)

    if returncode == 0:
        print(f"  {stdout.strip()}")
        return True

    if "cannot update chaos spec" in (stderr or "").lower() and kind and name and namespace:
        print("WARN: Existing ChaosMesh resource has immutable spec; deleting it and retrying apply once...")
        if not delete_chaos_resource(kind, name, namespace):
            print("ERROR: Failed to delete immutable existing resource before retry")
            if stderr:
                print(f"  Original apply STDERR: {stderr}")
            return False
        returncode, stdout, stderr = run_remote_kubectl(command)
        if returncode == 0:
            print(f"  {stdout.strip()}")
            return True

    print(f"ERROR: Failed to apply chaos YAML")
    if stderr:
        print(f"  STDERR: {stderr}")
    return False


def trigger_chaosmesh_experiment(yaml_file: str = None) -> bool:
    """
    Main function to trigger a ChaosMesh experiment.

    Args:
        yaml_file: Optional path to the YAML file (uses env var or default if not provided)

    Returns:
        bool: True if successful
    """
    yaml_path = yaml_file or CHAOSMESH_YAML_FILE

    print("=" * 60)
    print("ChaosMesh Experiment Trigger")
    print("=" * 60)
    print(f"GCE Instance: {INSTANCE_NAME}")
    print(f"Zone: {ZONE}")
    print(f"Remote User: {REMOTE_USER}")
    print()

    # Step 1: Load and parse YAML file
    yaml_content = load_yaml_file(yaml_path)
    kind, name, namespace = extract_chaos_info(yaml_content)
    print()

    # Step 2: Copy YAML file to VM
    yaml_filename = Path(yaml_path).name
    remote_yaml_path = f"/tmp/{yaml_filename}"

    if not copy_yaml_to_vm(yaml_path, remote_yaml_path):
        print("ERROR: Failed to copy YAML file to VM")
        return False
    print()

    # Step 3: Delete existing chaos resource (if any)
    if not delete_chaos_resource(kind, name, namespace):
        print("ERROR: Failed to delete existing resource; aborting to avoid immutable ChaosMesh spec update")
        return False
    print()

    # Step 4: Apply the chaos YAML
    if not apply_chaos_yaml(remote_yaml_path, kind, name, namespace):
        print("ERROR: Failed to apply chaos experiment")
        return False

    print()
    print("=" * 60)
    print(f"ChaosMesh experiment '{name}' triggered successfully!")
    print("=" * 60)

    return True


def main():
    """Main entry point."""
    success = trigger_chaosmesh_experiment()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
