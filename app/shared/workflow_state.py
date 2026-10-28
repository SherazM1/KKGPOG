"""Small, Streamlit-independent helpers for input-dependent workflow results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import MutableMapping, Sequence
from typing import Any


def _content(value: Any) -> Any:
    if hasattr(value, "getvalue"):
        return {"name": getattr(value, "name", ""), "content": _content(value.getvalue())}
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _content(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_content(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported fingerprint input: {type(value).__name__}")


def fingerprint(**inputs: Any) -> str:
    """Hash content and explicit settings, never object identities or stream offsets."""
    payload = json.dumps(_content(inputs), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_revision(path: str | Path) -> bytes | None:
    source = Path(path)
    return source.read_bytes() if source.is_file() else None


def sync_inputs(state: MutableMapping, workflow: str, current: str,
                result_keys: Sequence[str]) -> bool:
    """Clear dependent results on change; retain a notice until the next build."""
    key = f"{workflow}:fingerprint"
    stale_key = f"{workflow}:stale"
    if state.get(key) != current:
        if any(state.get(result_key) is not None for result_key in result_keys):
            state[stale_key] = True
        for result_key in result_keys:
            state[result_key] = None
        state[key] = current
    return bool(state.get(stale_key, False))


def mark_current(state: MutableMapping, workflow: str) -> None:
    state[f"{workflow}:stale"] = False


def holiday_fingerprint(bos, assortment, image_root, configuration=None):
    return fingerprint(bos=bos, assortment=assortment, image_root=image_root,
                       configuration=configuration)


def sams_fingerprint(source, selected_pog, image_root, image_zip, catalog, mappings):
    return fingerprint(source=source, selected_pog=selected_pog, image_root=image_root,
                       image_zip=image_zip, catalog=catalog, mappings=mappings)
