"""Persistence helpers: arrays in ``.npz`` with JSON metadata embedded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["to_jsonable", "save_npz", "load_npz", "save_json", "load_json"]

_META_KEY = "__meta__"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and paths into JSON-friendly types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def save_npz(path: str | Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _META_KEY in arrays:
        raise ValueError(f"array name {_META_KEY!r} is reserved")
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload[_META_KEY] = np.array(json.dumps(to_jsonable(meta)))
    np.savez_compressed(path, **payload)
    return path


def load_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != _META_KEY}
        meta = json.loads(str(data[_META_KEY])) if _META_KEY in data.files else {}
    return arrays, meta


def save_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2) + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
