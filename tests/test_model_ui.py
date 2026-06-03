from __future__ import annotations

import pandas as pd

from interactive_decision_tree.model_ui import (
    default_id_column,
    find_row_position_by_id_value,
    shap_default_stratify_columns,
    shap_score_band_labels,
    shap_strata_labels,
    stratified_sample_by_labels,
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


def test_shap_score_band_labels_create_ordered_score_bands():
    scores = pd.Series(range(100), index=range(100), dtype=float)

    bands = shap_score_band_labels(scores, bins=10)

    assert bands.nunique() == 10
    assert bands.iloc[0] == "score_01"
    assert bands.iloc[-1] == "score_10"


def test_shap_stratified_sample_balances_score_target_and_segments():
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(80)],
            "segment": ["A", "B"] * 40,
            "risk_flag": ["good", "bad"] * 40,
            "model_feature": range(80),
        }
    )
    scores = pd.Series([i / 79 for i in range(80)], index=df.index)
    strata = shap_strata_labels(
        df,
        scores,
        target="risk_flag",
        extra_columns=["segment"],
        score_bins=4,
    )

    sampled = stratified_sample_by_labels(df, strata, n=16, random_state=11)
    sampled_bands = shap_score_band_labels(scores.reindex(sampled.index), bins=4)

    assert len(sampled) == 16
    assert set(sampled["risk_flag"]) == {"good", "bad"}
    assert set(sampled["segment"]) == {"A", "B"}
    assert sampled_bands.nunique() >= 3


def test_shap_default_stratify_columns_prefers_business_columns_outside_model_features():
    df = pd.DataFrame(
        {
            "risk_flag": [0, 1],
            "segment": ["A", "B"],
            "product": ["p1", "p2"],
            "income": [10, 20],
        }
    )

    assert shap_default_stratify_columns(df, "risk_flag", ["income"]) == ["segment", "product"]
