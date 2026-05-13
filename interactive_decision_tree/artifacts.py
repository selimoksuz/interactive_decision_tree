from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def load_tree_pickle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Tree pickle must contain a dictionary payload.")
    return payload


def save_tree_pickle(payload: dict[str, Any], path: str | Path) -> Path:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary.")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def load_tree_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Tree JSON must contain a dictionary payload.")
    return payload
