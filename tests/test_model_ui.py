from __future__ import annotations

import pandas as pd

from interactive_decision_tree.model_ui import (
    default_id_column,
    find_row_position_by_id_value,
    what_if_id_columns,
)


def test_what_if_id_lookup_helpers_find_string_and_numeric_ids():
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "account_no": [1001, 1002, 1003],
            "target": [0, 1, 0],
        }
    )

    columns = what_if_id_columns(df, "target")

    assert columns == ["customer_id", "account_no"]
    assert default_id_column(columns) == "customer_id"
    assert find_row_position_by_id_value(df, "customer_id", "C002") == 1
    assert find_row_position_by_id_value(df, "account_no", "1003") == 2
    assert find_row_position_by_id_value(df, "customer_id", "missing") is None
