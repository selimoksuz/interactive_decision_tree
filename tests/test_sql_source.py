from __future__ import annotations

import pandas as pd

from interactive_decision_tree import launch_tree_sql
from interactive_decision_tree.session_store import load_dataframe_session
from interactive_decision_tree.sql_source import read_sql_dataframe


def sqlite_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'sample.db'}"


def seed_table(url: str) -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
            "target": [0, 1, 0, 1, 0],
        }
    )
    from sqlalchemy import create_engine

    engine = create_engine(url)
    try:
        df.to_sql("sample", engine, index=False, if_exists="replace")
    finally:
        engine.dispose()


def test_sql_table_with_limit(tmp_path):
    url = sqlite_url(tmp_path)
    seed_table(url)

    df = read_sql_dataframe(url, table="sample", limit=2)

    assert len(df) == 2
    assert df["id"].tolist() == [1, 2]


def test_sql_query_mode(tmp_path):
    url = sqlite_url(tmp_path)
    seed_table(url)

    df = read_sql_dataframe(url, query="select * from sample where id >= 3 order by id")

    assert df["id"].tolist() == [3, 4, 5]


def test_sql_sample_n(tmp_path):
    url = sqlite_url(tmp_path)
    seed_table(url)

    df = read_sql_dataframe(url, table="sample", full_table=True, sample_n=2)

    assert len(df) == 2
    assert set(df.columns) == {"id", "value", "target"}


def test_sql_full_table(tmp_path):
    url = sqlite_url(tmp_path)
    seed_table(url)

    df = read_sql_dataframe(url, table="sample", full_table=True)

    assert len(df) == 5


def test_launch_tree_sql_query_resolves_target_case_insensitively(tmp_path):
    url = sqlite_url(tmp_path)
    seed_table(url)
    session_dir = tmp_path / "sessions"

    ui_url = launch_tree_sql(
        url,
        query="select id, value, target as risk_flag from sample order by id",
        target="RISK_FLAG",
        full_table=True,
        start_server=False,
        open_browser=False,
        session_dir=session_dir,
    )

    from urllib.parse import parse_qs, urlparse

    data_id = parse_qs(urlparse(ui_url).query)["data_id"][0]
    _, metadata = load_dataframe_session(data_id, session_dir=session_dir)
    assert metadata["target"] == "risk_flag"
