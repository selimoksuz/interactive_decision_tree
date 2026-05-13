from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_SQL_LIMIT = 10_000


def _require_sqlalchemy() -> Any:
    try:
        import sqlalchemy as sa
    except ImportError as exc:
        raise ImportError(
            "SQL support requires SQLAlchemy. Install it with `pip install SQLAlchemy`."
        ) from exc
    return sa


def _engine_from_connection(connection: Any) -> tuple[Any, bool]:
    sa = _require_sqlalchemy()
    if isinstance(connection, str):
        return sa.create_engine(connection), True
    return connection, False


def _split_schema_table(table: str) -> tuple[str | None, str]:
    parts = [part.strip() for part in table.split(".") if part.strip()]
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, table.strip()


def read_sql_dataframe(
    connection: Any,
    *,
    table: str | None = None,
    query: str | None = None,
    limit: int | None = None,
    sample_n: int | None = None,
    full_table: bool = False,
    random_state: int = 7,
) -> pd.DataFrame:
    if bool(table) == bool(query):
        raise ValueError("Provide exactly one of table or query.")

    effective_limit = None if full_table else (limit if limit is not None else DEFAULT_SQL_LIMIT)
    if effective_limit is not None and effective_limit <= 0:
        effective_limit = None
    if sample_n is not None and sample_n <= 0:
        sample_n = None

    engine, should_dispose = _engine_from_connection(connection)
    sa = _require_sqlalchemy()
    try:
        with engine.connect() as conn:
            if table:
                schema, table_name = _split_schema_table(table)
                if not table_name:
                    raise ValueError("Table name cannot be empty.")
                metadata = sa.MetaData()
                table_obj = sa.Table(table_name, metadata, autoload_with=conn, schema=schema)
                statement = sa.select(table_obj)
                if effective_limit is not None:
                    statement = statement.limit(int(effective_limit))
                df = pd.read_sql(statement, conn)
            else:
                query_text = str(query or "").strip()
                if not query_text:
                    raise ValueError("Query cannot be empty.")
                df = pd.read_sql_query(sa.text(query_text), conn)
                if not full_table and limit is not None and int(limit) > 0:
                    df = df.head(int(limit))
    finally:
        if should_dispose:
            engine.dispose()

    if sample_n is not None and int(sample_n) < len(df):
        df = df.sample(n=int(sample_n), random_state=random_state).sort_index()
    return df
