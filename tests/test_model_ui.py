from __future__ import annotations

import pandas as pd

from interactive_decision_tree.model_ui import (
    default_id_column,
    find_row_position_by_id_value,
    shap_default_stratify_columns,
    shap_interaction_color_options,
    shap_score_band_labels,
    shap_stratify_column_labels,
    shap_strata_labels,
    stratified_sample_by_labels,
    what_if_candidate_values,
    what_if_id_columns,
    what_if_local_sensitivity,
)


class LocalSensitivityModel:
    feature_names_in_ = ["strong_num", "cat_driver", "weak_num"]
    classes_ = [0, 1]

    def predict_proba(self, frame: pd.DataFrame):
        score = (
            0.05
            + 0.08 * pd.to_numeric(frame["strong_num"], errors="coerce").fillna(0)
            + 0.45 * frame["cat_driver"].astype(str).eq("B").astype(float)
            + 0.01 * pd.to_numeric(frame["weak_num"], errors="coerce").fillna(0)
        ).clip(0, 1)
        return pd.concat([1 - score, score], axis=1).to_numpy(dtype=float)


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


def test_what_if_candidate_values_exclude_current_value():
    values = what_if_candidate_values(pd.Series([1, 2, 3, 4, 5]), current_value=3, max_values=4)

    assert len(values) <= 4
    assert 3 not in values


def test_what_if_local_sensitivity_orders_numeric_and_categorical_drivers():
    df = pd.DataFrame(
        {
            "strong_num": [0, 2, 4, 6, 8, 10],
            "cat_driver": ["A", "B", "A", "C", "B", "A"],
            "weak_num": [0, 1, 2, 3, 4, 5],
        }
    )

    ranking = what_if_local_sensitivity(
        model=LocalSensitivityModel(),
        df=df,
        row_position=0,
        feature_names=["strong_num", "cat_driver", "weak_num"],
        positive_class=1,
    )

    assert ranking["feature"].tolist() == ["strong_num", "cat_driver", "weak_num"]
    assert ranking.iloc[0]["sensitivity"] > ranking.iloc[1]["sensitivity"] > ranking.iloc[2]["sensitivity"]
    assert ranking.iloc[1]["candidate_value"] == "B"


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


def test_shap_interaction_color_options_exclude_main_feature():
    options = shap_interaction_color_options(["income", "age", "segment"], "income")

    assert options == ["auto", "age", "segment"]


def test_shap_strata_labels_keep_numeric_target_as_class_label():
    df = pd.DataFrame(
        {
            "risk_flag": [0, 1, 0, 1],
            "application_date": pd.to_datetime(["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15"]),
        }
    )
    scores = pd.Series([0.1, 0.2, 0.8, 0.9], index=df.index)

    strata = shap_strata_labels(
        df,
        scores,
        target="risk_flag",
        extra_columns=["application_date"],
        score_bins=2,
        include_target=True,
    )

    assert any("|risk_flag=0|" in value for value in strata)
    assert any("|risk_flag=1|" in value for value in strata)
    assert any("application_date=2026-01" in value for value in strata)
    assert any("application_date=2026-02" in value for value in strata)


def test_shap_stratify_column_labels_bin_numeric_and_month_bucket_dates():
    numeric = pd.Series([10, 20, 30, 40], name="score")
    date = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-20", "2026-02-01", None]), name="as_of_date")

    numeric_labels = shap_stratify_column_labels(numeric, prefix="score", bins=2)
    date_labels = shap_stratify_column_labels(date, prefix="as_of_date")

    assert set(numeric_labels) == {"score_score_01", "score_score_02"}
    assert date_labels.tolist() == [
        "as_of_date=2026-01",
        "as_of_date=2026-01",
        "as_of_date=2026-02",
        "as_of_date=__MISSING__",
    ]
