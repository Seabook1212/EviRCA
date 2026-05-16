from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from run_semantic_mode import main as run_semantic_main


def main() -> None:
    sys.argv = [
        "run_semantic_mode.py",
        "--incident-id",
        "pod_network_loss_catalogue_001",
        "--backend-mode",
        "api",
        "--message",
        (
            "Analyze sock-shop from 2026-05-02T20:56:18Z to 2026-05-02T21:07:18Z. "
        ),
    ]
    run_semantic_main()


if __name__ == "__main__":
    main()
