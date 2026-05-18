from __future__ import annotations

from decimal import Decimal
from numbers import Number
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


def _strip_query_terminator(query: str) -> str:
    return query.strip().rstrip(";").strip()


def _limited_query_for_dialect(query: str, dialect_name: str, limit: int) -> tuple[str, dict[str, int]]:
    clean_query = _strip_query_terminator(query)
    if not clean_query:
        raise ValueError("Query cannot be empty.")
    if dialect_name.lower().startswith("oracle"):
        return f"select * from ({clean_query}) idt_query_limit where rownum <= :idt_limit", {"idt_limit": int(limit)}
    return f"select * from ({clean_query}) as idt_query_limit limit :idt_limit", {"idt_limit": int(limit)}


def _object_series_is_numeric(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.head(1000)
    return all(isinstance(value, (Number, Decimal)) and not isinstance(value, bool) for value in sample)


def normalize_sql_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in normalized.columns:
        series = normalized[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        if _object_series_is_numeric(series):
            normalized[column] = pd.to_numeric(series, errors="coerce")
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        unique_count = int(non_null.nunique(dropna=True))
        unique_ratio = unique_count / max(1, len(non_null))
        if unique_count <= 5_000 and unique_ratio <= 0.2:
            normalized[column] = series.astype("category")
    return normalized


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
                query_text = _strip_query_terminator(str(query or ""))
                if not query_text:
                    raise ValueError("Query cannot be empty.")
                if effective_limit is not None:
                    limited_query, params = _limited_query_for_dialect(
                        query_text,
                        str(getattr(conn.dialect, "name", "")),
                        int(effective_limit),
                    )
                    df = pd.read_sql_query(sa.text(limited_query), conn, params=params)
                else:
                    df = pd.read_sql_query(sa.text(query_text), conn)
    finally:
        if should_dispose:
            engine.dispose()

    if sample_n is not None and int(sample_n) < len(df):
        df = df.sample(n=int(sample_n), random_state=random_state).sort_index()
    return normalize_sql_dataframe(df)
