from __future__ import annotations
import json
import ast
from typing import Any, Dict


def parse_mtp_answers(raw: Any) -> Dict[str, Any]:
    """Parse heterogeneous MTP answers (json, python-literal, dict, str) into a dict.
    Returns empty dict on failure.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    s = str(raw)
    # try json
    try:
        return json.loads(s)
    except Exception:
        pass
    # try python literal
    try:
        val = ast.literal_eval(s)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def mtp_to_text(d: Dict[str, Any]) -> str:
    """Flatten parsed MTP dict into a readable single text (metier, talent, paradigme)."""
    if not isinstance(d, dict) or not d:
        return ""
    parts = []
    for k in ("metier", "talent", "paradigme"):
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            parts.extend([str(x) for x in v if str(x).strip()])
        elif isinstance(v, dict):
            for _, vv in v.items():
                if isinstance(vv, list):
                    parts.extend([str(x) for x in vv if str(x).strip()])
                elif vv is not None and str(vv).strip():
                    parts.append(str(vv))
        elif str(v).strip():
            parts.append(str(v))
    return "\n".join([p for p in parts if p.strip()])
