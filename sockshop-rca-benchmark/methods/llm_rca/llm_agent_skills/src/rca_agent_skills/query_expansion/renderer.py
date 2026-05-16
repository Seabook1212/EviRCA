from __future__ import annotations

import re
from typing import Any


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_template(template: str, values: dict[str, Any]) -> str:
    safe_values = {k: ("" if v is None else v) for k, v in values.items()}
    return _PLACEHOLDER_RE.sub(lambda match: str(safe_values.get(match.group(1), match.group(0))), template)
