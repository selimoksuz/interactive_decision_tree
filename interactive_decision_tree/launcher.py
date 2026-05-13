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


def _server_is_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(_server_url(port), timeout=1) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def _ensure_streamlit_server(port: int) -> None:
    if _server_is_ready(port):
        return

    env = os.environ.copy()
    env[SESSION_DIR_ENV] = str(default_session_dir())
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
    _source: str = "notebook",
    _metadata: dict[str, Any] | None = None,
) -> str:
    data_id, _ = save_dataframe_session(
        df,
        source=_source,
        name=session_name or "Notebook DataFrame",
        target=target,
        features=features,
        metadata=_metadata,
    )
    if start_server:
        _ensure_streamlit_server(port)

    query = urllib.parse.urlencode(
        {
            DATA_ID_QUERY_PARAM: data_id,
            "work_id": work_id or uuid4().hex,
        }
    )
    url = f"{_server_url(port)}/?{query}"
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
