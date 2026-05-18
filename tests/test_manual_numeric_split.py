from __future__ import annotations

import pandas as pd

from interactive_decision_tree_app import (
    candidate_branch_detail_rows,
    candidate_split_summary_rows,
    manual_numeric_branch_rows,
    parse_threshold_text,
    score_numeric_manual_bins,
)


def test_parse_threshold_text_accepts_commas_semicolons_and_newlines():
    assert parse_threshold_text("40, 20;30\n50") == [20.0, 30.0, 40.0, 50.0]


def test_manual_numeric_branch_rows_shows_each_manual_bin_count():
    df = pd.DataFrame({"x": [10, 20, 25, 35, 45, None]})

    rows = manual_numeric_branch_rows(df, df.index.tolist(), "x", [20, 30, 40])

    assert rows == [
        {"branch": "<= 20", "rows": 2},
        {"branch": "> 20 and <= 30", "rows": 1},
        {"branch": "> 30 and <= 40", "rows": 1},
        {"branch": "> 40 or missing", "rows": 2},
    ]


def test_candidate_branch_detail_rows_expands_manual_split_branches():
    df = pd.DataFrame(
        {
            "x": [10, 20, 25, 35, 45, 55],
            "risk_flag": ["low", "low", "low", "high", "high", "high"],
        }
    )
    candidate = score_numeric_manual_bins(
        df=df,
        target="risk_flag",
        row_idx=df.index.tolist(),
        feature="x",
        thresholds=[20, 40],
        min_leaf=1,
    )
    assert candidate is not None

    rows = candidate_branch_detail_rows(df, "risk_flag", df.index.tolist(), candidate)

    assert [row["condition"] for row in rows] == ["<= 20", "> 20 and <= 40", "> 40"]
    assert [row["rows"] for row in rows] == [2, 2, 2]
    assert [row["positive_class_count"] for row in rows] == [0, 1, 2]
    assert all("branch_impurity" in row for row in rows)


def test_candidate_split_summary_rows_matches_manual_preview_columns():
    df = pd.DataFrame(
        {
            "x": [10, 20, 25, 35, 45, 55],
            "risk_flag": ["low", "low", "low", "high", "high", "high"],
        }
    )
    candidate = score_numeric_manual_bins(
        df=df,
        target="risk_flag",
        row_idx=df.index.tolist(),
        feature="x",
        thresholds=[20, 40],
        min_leaf=1,
    )
    assert candidate is not None

    rows = candidate_split_summary_rows(df, "risk_flag", df.index.tolist(), candidate)

    assert len(rows) == 1
    assert rows[0]["split_type"] == "numeric_manual_bins"
    assert rows[0]["split"] == "x manual bins: 20, 40"
    assert rows[0]["branches"] == 3
    assert "information_gain" in rows[0]
    assert "weighted_tree_delta" in rows[0]
    assert "child_weighted_impurity" in rows[0]
