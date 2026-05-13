from __future__ import annotations

import pandas as pd

from interactive_decision_tree_app import manual_numeric_branch_rows, parse_threshold_text


def test_parse_threshold_text_accepts_commas_semicolons_and_newlines():
    assert parse_threshold_text("40, 20;30\n50") == [20.0, 30.0, 40.0, 50.0]


def test_manual_numeric_branch_rows_shows_each_manual_bin_count():
    df = pd.DataFrame({"x": [10, 20, 25, 35, 45, None]})

    rows = manual_numeric_branch_rows(df, df.index.tolist(), "x", [20, 30, 40])

    assert rows == [
        {"branch": "<= 20", "rows": 2},
        {"branch": "> 20 and <= 30", "rows": 1},
        {"branch": "> 30 and <= 40", "rows": 1},
        {"branch": "> 40", "rows": 2},
    ]
