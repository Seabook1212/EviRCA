#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChaosMesh Experiment Trigger Script

This script triggers a ChaosMesh experiment by:
1. Reading the YAML file to extract kind and metadata.name
2. Copying the YAML file to a remote VM via native scp
3. Deleting any existing chaos resource with the same name
4. Applying the chaos YAML to create a new experiment

The kubectl commands are executed remotely on a target VM via native ssh.
This version is suitable for running on VM 075527 and connecting to VM 043614
using a private SSH key such as ~/.ssh/google_compute_engine.
"""

import os
import sys
import time
import shlex
import subprocess
import yaml
from pathlib import Path


# ===== Configuration =====
# Remote VM configuration
REMOTE_HOST = os.getenv("REMOTE_HOST", "10.128.0.2")
REMOTE_USER = os.getenv("REMOTE_USER", "g6878310023")
REMOTE_PORT = int(os.getenv("REMOTE_PORT", "22"))
REMOTE_KUBECONFIG = os.getenv("REMOTE_KUBECONFIG", f"/home/{REMOTE_USER}/.kube/config")
REMOTE_SSH_KEY = os.getenv("REMOTE_SSH_KEY", f"/home/{REMOTE_USER}/.ssh/google_compute_engine")

# ssh/scp settings
SSH_TIMEOUT_SECONDS = int(os.getenv("CHAOSMESH_SSH_TIMEOUT_SECONDS", "180"))
SCP_TIMEOUT_SECONDS = int(os.getenv("CHAOSMESH_SCP_TIMEOUT_SECONDS", "180"))
SCP_MAX_RETRIES = int(os.getenv("CHAOSMESH_SCP_MAX_RETRIES", "3"))
SCP_RETRY_INTERVAL_SECONDS = int(os.getenv("CHAOSMESH_SCP_RETRY_INTERVAL_SECONDS", "5"))

# ChaosMesh YAML file path (can be overridden by environment variable)
SCRIPT_DIR = Path(__file__).parent
DEFAULT_YAML_FILE = SCRIPT_DIR.parent / "chaosmesh_yaml" / "pod_io_fault_carts-db_001.yaml"
CHAOSMESH_YAML_FILE = os.getenv("CHAOSMESH_YAML_FILE", str(DEFAULT_YAML_FILE))

# Kubernetes namespace (default from YAML, can be overridden)
DEFAULT_NAMESPACE = os.getenv("CHAOSMESH_NAMESPACE", "sock-shop")


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
    with open(yaml_path, "r", encoding="utf-8") as f:
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


def _ensure_ssh_key_exists() -> None:
    """Validate that the SSH private key exists."""
    key_path = Path(REMOTE_SSH_KEY)
    if not key_path.exists():
        print(f"ERROR: SSH private key not found: {key_path}")
        sys.exit(1)


def _ssh_base_command() -> list:
    """
    Build the base ssh command.

    Returns:
        list: Base ssh command arguments
    """
    return [
        "ssh",
        "-i", REMOTE_SSH_KEY,
        "-p", str(REMOTE_PORT),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/home/{}/.ssh/known_hosts".format(REMOTE_USER),
        f"{REMOTE_USER}@{REMOTE_HOST}",
    ]


def _scp_base_command() -> list:
    """
    Build the base scp command.

    Returns:
        list: Base scp command arguments
    """
    return [
        "scp",
        "-i", REMOTE_SSH_KEY,
        "-P", str(REMOTE_PORT),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/home/{}/.ssh/known_hosts".format(REMOTE_USER),
    ]


def run_remote_kubectl(command: str) -> tuple:
    """
    Execute kubectl command remotely on the target VM via native ssh.

    Args:
        command: The kubectl command to execute

    Returns:
        tuple: (return_code, stdout, stderr)
    """
    remote_cmd = f"export KUBECONFIG={shlex.quote(REMOTE_KUBECONFIG)} && {command}"
    ssh_command = _ssh_base_command() + [remote_cmd]

    print(f"==> Executing remote command: {command}")

    try:
        result = subprocess.run(
            ssh_command,
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT_SECONDS
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: Command timed out after {SSH_TIMEOUT_SECONDS} seconds")
        if e.stdout:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout)
            print(f"  STDOUT(before timeout): {out.strip()}")
        if e.stderr:
            err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
            print(f"  STDERR(before timeout): {err.strip()}")
        return -1, "", f"Timeout after {SSH_TIMEOUT_SECONDS}s"
    except Exception as e:
        print(f"ERROR: Failed to execute command: {e}")
        return -1, "", str(e)


def copy_yaml_to_vm(yaml_path: str, remote_path: str) -> bool:
    """
    Copy the YAML file to the target VM using native scp.

    Args:
        yaml_path: Local path to the YAML file
        remote_path: Remote path on the VM

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"==> Copying YAML file to VM: {remote_path}")

    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        print(f"ERROR: Local YAML file not found: {yaml_file}")
        return False

    remote_target = f"{REMOTE_USER}@{REMOTE_HOST}:{remote_path}"
    scp_command = _scp_base_command() + [str(yaml_file), remote_target]

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
    kind_lower = kind.lower()
    command = f"kubectl delete {kind_lower} {shlex.quote(name)} -n {shlex.quote(namespace)} --ignore-not-found"

    print("==> Deleting existing chaos resource...")
    returncode, stdout, stderr = run_remote_kubectl(command)

    if returncode == 0:
        if stdout.strip():
            print(f"  {stdout.strip()}")
        else:
            print("  No existing resource found (or deleted successfully)")
        return True

    print("ERROR: Failed to delete resource")
    if stderr and stderr.strip():
        print(f"  STDERR: {stderr.strip()}")
    return False


def apply_chaos_yaml(yaml_path: str) -> bool:
    """
    Apply the chaos YAML file to create the experiment.

    Args:
        yaml_path: Path to the YAML file on the remote VM

    Returns:
        bool: True if successful
    """
    command = f"kubectl apply -f {shlex.quote(yaml_path)}"

    print("==> Applying chaos experiment...")
    returncode, stdout, stderr = run_remote_kubectl(command)

    if returncode == 0:
        if stdout.strip():
            print(f"  {stdout.strip()}")
        return True

    print("ERROR: Failed to apply chaos YAML")
    if stderr and stderr.strip():
        print(f"  STDERR: {stderr.strip()}")
    return False


def trigger_chaosmesh_experiment(yaml_file: str = None) -> bool:
    """
    Main function to trigger a ChaosMesh experiment.

    Args:
        yaml_file: Optional path to the YAML file (uses env var or default if not provided)

    Returns:
        bool: True if successful
    """
    _ensure_ssh_key_exists()

    yaml_path = yaml_file or CHAOSMESH_YAML_FILE

    print("=" * 60)
    print("ChaosMesh Experiment Trigger")
    print("=" * 60)
    print(f"Remote Host: {REMOTE_HOST}")
    print(f"Remote User: {REMOTE_USER}")
    print(f"Remote Port: {REMOTE_PORT}")
    print(f"Remote Kubeconfig: {REMOTE_KUBECONFIG}")
    print(f"SSH Key: {REMOTE_SSH_KEY}")
    print()

    # Step 1: Load and parse YAML file
    yaml_content = load_yaml_file(yaml_path)
    kind, name, namespace = extract_chaos_info(yaml_content)
    print()

    # Step 2: Copy YAML file to remote VM
    yaml_filename = Path(yaml_path).name
    remote_yaml_path = f"/tmp/{yaml_filename}"

    if not copy_yaml_to_vm(yaml_path, remote_yaml_path):
        print("ERROR: Failed to copy YAML file to remote VM")
        return False
    print()

    # Step 3: Delete existing chaos resource (if any)
    if not delete_chaos_resource(kind, name, namespace):
        print("WARN: Failed to delete existing resource, continuing anyway...")
    print()

    # Step 4: Apply the chaos YAML
    if not apply_chaos_yaml(remote_yaml_path):
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