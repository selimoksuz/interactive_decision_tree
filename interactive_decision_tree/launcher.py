from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .session_store import (
    DATA_ID_QUERY_PARAM,
    SESSION_DIR_ENV,
    default_session_dir,
    project_root,
    save_dataframe_session,
)
from .sql_source import read_sql_dataframe


def _app_path() -> Path:
    import interactive_decision_tree_app

    return Path(interactive_decision_tree_app.__file__).resolve()


def _server_url(port: int) -> str:
    return f"http://localhost:{int(port)}"


def _append_query(base_url: str, query: str) -> str:
    parts = urllib.parse.urlsplit(base_url)
    path = parts.path or "/"
    existing_query = parts.query
    merged_query = f"{existing_query}&{query}" if existing_query else query
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, merged_query, parts.fragment))


def _launch_url(
    port: int,
    query: str,
    *,
    host: str = "localhost",
    scheme: str = "http",
    base_url: str | None = None,
) -> str:
    if base_url is None and "://" in host:
        base_url = host
    if base_url is None:
        base_url = f"{scheme}://{host}:{int(port)}/"
    return _append_query(base_url, query)


def _server_is_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(_server_url(port), timeout=1) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def _resolve_session_dir(session_dir: str | Path | None = None) -> Path:
    if session_dir is not None:
        return Path(session_dir).expanduser().resolve()
    return default_session_dir()


def _resolve_dataframe_column(df: pd.DataFrame, column: str | None) -> str | None:
    if column is None or column in df.columns:
        return column

    requested = str(column).casefold()
    matches = [existing for existing in df.columns if str(existing).casefold() == requested]
    if len(matches) == 1:
        return str(matches[0])
    return column


def _resolve_dataframe_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str] | None:
    if columns is None:
        return None
    return [_resolve_dataframe_column(df, column) or column for column in columns]


def _ensure_streamlit_server(port: int, session_dir: str | Path | None = None) -> None:
    if _server_is_ready(port):
        return

    resolved_session_dir = _resolve_session_dir(session_dir)
    env = os.environ.copy()
    env[SESSION_DIR_ENV] = str(resolved_session_dir)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_app_path()),
        "--server.port",
        str(int(port)),
        "--server.headless",
        "true",
    ]
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    subprocess.Popen(
        command,
        cwd=str(project_root()),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    deadline = time.time() + 25
    while time.time() < deadline:
        if _server_is_ready(port):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Streamlit server did not start on port {port}.")


def launch_tree(
    df: pd.DataFrame,
    target: str | None = None,
    features: list[str] | None = None,
    work_id: str | None = None,
    port: int = 8501,
    open_browser: bool = True,
    start_server: bool = True,
    session_name: str | None = None,
    session_dir: str | Path | None = None,
    host: str = "localhost",
    scheme: str = "http",
    base_url: str | None = None,
    _source: str = "notebook",
    _metadata: dict[str, Any] | None = None,
) -> str:
    resolved_session_dir = _resolve_session_dir(session_dir)
    resolved_target = _resolve_dataframe_column(df, target)
    resolved_features = _resolve_dataframe_columns(df, features)
    data_id, _ = save_dataframe_session(
        df,
        source=_source,
        name=session_name or "Notebook DataFrame",
        target=resolved_target,
        features=resolved_features,
        metadata=_metadata,
        session_dir=resolved_session_dir,
    )
    if start_server:
        _ensure_streamlit_server(port, resolved_session_dir)

    query = urllib.parse.urlencode(
        {
            DATA_ID_QUERY_PARAM: data_id,
            "work_id": work_id or uuid4().hex,
        }
    )
    url = _launch_url(port, query, host=host, scheme=scheme, base_url=base_url)
    if open_browser:
        webbrowser.open(url)
    return url


def launch_tree_sql(
    connection: Any,
    table: str | None = None,
    query: str | None = None,
    target: str | None = None,
    limit: int | None = None,
    sample_n: int | None = None,
    full_table: bool = False,
    **launch_kwargs: Any,
) -> str:
    df = read_sql_dataframe(
        connection,
        table=table,
        query=query,
        limit=limit,
        sample_n=sample_n,
        full_table=full_table,
    )
    source_name = table or "SQL query"
    metadata = {
        "sql": {
            "mode": "table" if table else "query",
            "table": table,
            "limit": limit,
            "sample_n": sample_n,
            "full_table": full_table,
        }
    }
    return launch_tree(
        df,
        target=target,
        session_name=launch_kwargs.pop("session_name", source_name),
        _source="sql",
        _metadata=metadata,
        **launch_kwargs,
    )
