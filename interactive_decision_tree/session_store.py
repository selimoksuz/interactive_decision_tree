from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


DATA_ID_QUERY_PARAM = "data_id"
SESSION_DIR_ENV = "INTERACTIVE_TREE_SESSION_DIR"
SESSION_DIR_NAME = ".tree_sessions"
DATA_FILE_NAME = "data.pkl"
METADATA_FILE_NAME = "metadata.json"
_DATA_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_PROJECT_CHILD_NAME = "interactive_decision_tree"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _looks_like_project_root(path: Path) -> bool:
    return (
        (path / "interactive_decision_tree_app.py").exists()
        or (path / "BUSINESS_RELEASE_MANIFEST.json").exists()
        or ((path / "scripts" / "start_app.py").exists() and (path / "start_interactive_tree.sh").exists())
    )


def _candidate_project_roots() -> list[Path]:
    candidates: list[Path] = []

    def add(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        if resolved not in candidates:
            candidates.append(resolved)

    try:
        cwd = Path.cwd().resolve()
    except OSError:
        cwd = None

    if cwd is not None:
        for path in (cwd, *cwd.parents):
            add(path)
            add(path / _PROJECT_CHILD_NAME)

    root = project_root()
    add(root)
    for parent in root.parents:
        add(parent)
    return candidates


def discover_runtime_root() -> Path:
    for candidate in _candidate_project_roots():
        if _looks_like_project_root(candidate):
            return candidate
    return project_root()


def default_session_dir() -> Path:
    configured = os.environ.get(SESSION_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return discover_runtime_root() / SESSION_DIR_NAME


def normalize_data_id(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    text = str(value).strip()
    if _DATA_ID_RE.fullmatch(text):
        return text
    return None


def new_data_id() -> str:
    return uuid4().hex


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    columns = "\x1f".join(map(str, df.columns)).encode("utf-8", errors="replace")
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    except TypeError:
        row_hashes = df.astype(str).to_csv(index=True).encode("utf-8", errors="replace")
    return hashlib.sha256(columns + row_hashes).hexdigest()[:16]


def session_data_key(data_id: str, df: pd.DataFrame, metadata: dict[str, Any] | None = None) -> str:
    fingerprint = metadata.get("fingerprint") if isinstance(metadata, dict) else None
    if not fingerprint:
        fingerprint = dataframe_fingerprint(df)
    return f"session:{data_id}:{len(df)}:{len(df.columns)}:{fingerprint}"


def session_path(data_id: str, session_dir: Path | None = None) -> Path:
    safe_data_id = normalize_data_id(data_id)
    if safe_data_id is None:
        raise ValueError("Invalid data_id.")

    root = (session_dir or default_session_dir()).resolve()
    path = (root / safe_data_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Invalid data_id path.") from exc
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def save_dataframe_session(
    df: pd.DataFrame,
    *,
    data_id: str | None = None,
    source: str = "notebook",
    name: str | None = None,
    target: str | None = None,
    features: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    session_dir: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df must contain at least one row.")

    safe_data_id = normalize_data_id(data_id) if data_id is not None else new_data_id()
    if safe_data_id is None:
        raise ValueError("Invalid data_id.")

    if target is not None and target not in df.columns:
        raise ValueError(f"target column not found in DataFrame: {target}")
    if features is not None:
        missing = [feature for feature in features if feature not in df.columns]
        if missing:
            raise ValueError(f"feature column(s) not found in DataFrame: {missing}")

    path = session_path(safe_data_id, session_dir)
    path.mkdir(parents=True, exist_ok=True)

    fingerprint = dataframe_fingerprint(df)
    out_metadata: dict[str, Any] = {
        "data_id": safe_data_id,
        "source": source,
        "name": name or source,
        "target": target,
        "features": features,
        "rows": int(len(df)),
        "columns": [str(column) for column in df.columns],
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pandas_version": pd.__version__,
    }
    if metadata:
        out_metadata.update(_json_safe(metadata))

    data_tmp = path / f"{DATA_FILE_NAME}.tmp"
    meta_tmp = path / f"{METADATA_FILE_NAME}.tmp"
    df.to_pickle(data_tmp)
    meta_tmp.write_text(json.dumps(out_metadata, indent=2, default=str), encoding="utf-8")
    data_tmp.replace(path / DATA_FILE_NAME)
    meta_tmp.replace(path / METADATA_FILE_NAME)
    return safe_data_id, out_metadata


def load_dataframe_session(
    data_id: str,
    *,
    session_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = session_path(data_id, session_dir)
    data_path = path / DATA_FILE_NAME
    metadata_path = path / METADATA_FILE_NAME
    if not data_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Data session not found: {data_id}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    df = pd.read_pickle(data_path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Stored session is not a pandas DataFrame: {data_id}")
    return df, metadata


def try_load_dataframe_session(data_id: str | None) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    safe_data_id = normalize_data_id(data_id)
    if safe_data_id is None:
        return None
    try:
        return load_dataframe_session(safe_data_id)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None
