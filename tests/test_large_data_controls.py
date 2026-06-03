from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st

from interactive_decision_tree.session_store import save_dataframe_session
from interactive_decision_tree_app import (
    DEFAULT_DEMO_ROWS,
    PREDICTION_THRESHOLD_F1,
    PREDICTION_THRESHOLD_MANUAL,
    PREDICTION_THRESHOLD_MANUAL_KEY,
    PREDICTION_THRESHOLD_MODE_KEY,
    analysis_row_idx,
    apply_feature_manager_edits,
    best_f1_threshold_from_scores,
    build_binary_validation_state,
    build_optimal_tree,
    cached_ranking_ready_message,
    candidate_cache_key,
    get_cached_candidates,
    candidate_splits,
    candidate_validation_stats,
    candidate_validation_stats_from_state,
    checkpoint_ui_state,
    choose_generalized_split_candidate,
    default_model_feature_selection,
    demo_data_key,
    evaluation_model_metrics,
    filter_feature_options,
    init_tree,
    leaf_performance_rows,
    make_demo_data,
    model_metrics,
    model_performance_wide_table,
    restore_checkpoint_dataframe,
    restore_checkpoint_ui_state,
    restore_tree_state_from_checkpoint,
    score_split,
    normalize_feature_selection,
    session_auto_tree_needs_generalization_restore,
    split_branch_indices,
    split_ranking_scope_caption,
    split_node,
    store_cached_candidates,
    train_test_split_indices,
    tree_export,
    tree_predictions,
    update_feature_selection_for_filtered,
    undo_last_split,
    validate_test_dataframe,
    AUTO_TREE_GENERALIZATION_VERSION,
)


def test_demo_data_default_is_large_enough_for_train_test_validation():
    demo = make_demo_data()
    assert len(demo) == DEFAULT_DEMO_ROWS
    assert DEFAULT_DEMO_ROWS >= 5_000
    assert demo["customer_id"].iloc[0] == "CUST000001"
    assert demo["customer_id"].is_unique
    assert demo[["age", "income", "tenure_months", "segment"]].isna().any().any()
    assert (demo["income"] == -999).any()
    assert (demo["region"] == "UNKNOWN").any()
    assert (demo["product"] == "NO_INFO").any()
    default_features = [column for column in demo.columns if column != "risk_flag"]
    assert "customer_id" not in default_model_feature_selection(default_features)
    assert demo_data_key(demo).startswith("demo:5000:9:")


def test_split_ranking_scope_caption_distinguishes_leaf_from_active_train():
    message = split_ranking_scope_caption(
        candidate_feature_count=6,
        total_feature_count=6,
        ranked_feature_limit=3,
        selected_node_rows=500,
        active_train_rows=5_000,
    )

    assert "500 row(s) in the selected leaf" in message
    assert "5,000 active train row(s)" in message
    assert "full active train data" in message
    assert "top 3 variable(s)" in message


def test_cached_ranking_message_distinguishes_leaf_from_active_train():
    message = cached_ranking_ready_message(
        candidate_count=164,
        analyzed_rows=500,
        selected_node_rows=500,
        active_train_rows=5_000,
    )

    assert message == "Cached ranking ready: 164 candidate(s), 500 selected-leaf row(s) out of 5,000 active train row(s)."


def test_data_setup_feature_search_supports_contains_multiple_terms_and_wildcards():
    features = ["age", "income", "avg_balance_3m", "risk_score", "segment"]

    assert filter_feature_options(features, "bal") == ["avg_balance_3m"]
    assert filter_feature_options(features, "age,score") == ["age", "risk_score"]
    assert filter_feature_options(features, "inc%") == ["income"]
    assert filter_feature_options(features, "*ment") == ["segment"]


def test_data_setup_feature_selection_actions_preserve_feature_order():
    features = ["age", "income", "segment", "score"]
    selected = ["income"]
    filtered = ["age", "score"]

    assert update_feature_selection_for_filtered(features, selected, filtered, True) == [
        "age",
        "income",
        "score",
    ]
    assert update_feature_selection_for_filtered(features, ["age", "income", "score"], filtered, False) == [
        "income"
    ]
    assert normalize_feature_selection(features, ["income", "missing", "age"]) == ["income", "age"]


def test_data_setup_feature_manager_edits_preserve_hidden_selection():
    features = ["age", "income", "segment", "score"]
    edited = pd.DataFrame({"include": [True, False], "variable": ["age", "score"]})

    assert apply_feature_manager_edits(features, ["income", "score"], edited) == ["age", "income"]


def test_analysis_row_idx_samples_stably():
    row_idx = list(range(1_000))

    first = analysis_row_idx(row_idx, max_rows=100, random_state=42)
    second = analysis_row_idx(row_idx, max_rows=100, random_state=42)

    assert first == second
    assert len(first) == 100
    assert first == sorted(first)
    assert set(first).issubset(row_idx)


def test_analysis_row_idx_keeps_small_inputs_unchanged():
    row_idx = [4, 2, 9]

    assert analysis_row_idx(row_idx, max_rows=10) == row_idx


def test_analysis_row_idx_stratifies_by_target_when_sampling():
    df = pd.DataFrame(
        {
            "target": ["bad"] * 80 + ["good"] * 20,
        }
    )

    sampled = analysis_row_idx(df.index.tolist(), max_rows=20, random_state=7, df=df, target="target")
    counts = df.loc[sampled, "target"].value_counts().to_dict()

    assert counts == {"bad": 16, "good": 4}


def test_analysis_row_idx_stratifies_by_selected_categorical_columns():
    df = pd.DataFrame(
        {
            "segment": ["A"] * 80 + ["B"] * 20,
            "target": ["bad", "good"] * 50,
        }
    )

    sampled = analysis_row_idx(
        df.index.tolist(),
        max_rows=20,
        random_state=7,
        df=df,
        stratify_columns=["segment"],
    )
    counts = df.loc[sampled, "segment"].value_counts().to_dict()

    assert counts == {"A": 16, "B": 4}


def test_analysis_row_idx_bins_numeric_stratify_columns():
    df = pd.DataFrame({"score": range(100)})

    sampled = analysis_row_idx(
        df.index.tolist(),
        max_rows=20,
        random_state=7,
        df=df,
        stratify_columns=["score"],
        stratify_numeric_bins=4,
    )
    bins = pd.qcut(df["score"], q=4, duplicates="drop")
    sampled_counts = bins.loc[sampled].value_counts().sort_index().tolist()

    assert sampled_counts == [5, 5, 5, 5]


def test_train_test_split_indices_stratifies_target():
    df = pd.DataFrame(
        {
            "x": range(100),
            "risk_flag": ["high"] * 30 + ["low"] * 70,
        }
    )

    train_idx, test_idx = train_test_split_indices(
        df,
        target="risk_flag",
        test_fraction=0.2,
        random_state=11,
        stratify=True,
    )

    assert len(train_idx) == 80
    assert len(test_idx) == 20
    assert df.loc[test_idx, "risk_flag"].value_counts().to_dict() == {"low": 14, "high": 6}


def test_train_test_split_indices_uses_selected_stratify_columns():
    df = pd.DataFrame(
        {
            "segment": ["A"] * 80 + ["B"] * 20,
            "risk_flag": ["high", "low"] * 50,
        }
    )

    train_idx, test_idx = train_test_split_indices(
        df,
        target="risk_flag",
        test_fraction=0.2,
        random_state=11,
        stratify=True,
        stratify_columns=["segment"],
    )

    assert len(train_idx) == 80
    assert len(test_idx) == 20
    assert df.loc[test_idx, "segment"].value_counts().to_dict() == {"A": 16, "B": 4}


def test_validate_test_dataframe_requires_target_and_features():
    test_df = pd.DataFrame({"age": [30], "risk_flag": [1]})

    assert validate_test_dataframe(test_df, "risk_flag", ["age", "income"]) == ["income"]


def test_model_performance_wide_table_pivots_train_test_columns():
    metrics = pd.DataFrame(
        [
            {"dataset": "Train", "metric": "rows", "value": 80},
            {"dataset": "Train", "metric": "accuracy", "value": 0.8},
            {"dataset": "Test", "metric": "rows", "value": 20},
            {"dataset": "Test", "metric": "accuracy", "value": 0.7},
        ]
    )

    wide = model_performance_wide_table(metrics)

    assert wide.columns.tolist() == ["metric", "Train", "Test"]
    assert wide.to_dict("records") == [
        {"metric": "rows", "Train": 80, "Test": 20},
        {"metric": "accuracy", "Train": 0.8, "Test": 0.7},
    ]


def test_checkpoint_ui_state_excludes_button_widget_values():
    st.session_state.clear()
    st.session_state["group_merge_category_groups::demo::risk_flag::5::product"] = [0, 1]
    st.session_state["group_merge_button_category_groups::demo::risk_flag::5::product"] = True

    saved = checkpoint_ui_state()

    assert "group_merge_category_groups::demo::risk_flag::5::product" in saved
    assert "group_merge_button_category_groups::demo::risk_flag::5::product" not in saved


def test_restore_checkpoint_ui_state_skips_stale_button_widget_values():
    st.session_state.clear()
    restore_checkpoint_ui_state(
        {
            "ui_state": {
                "group_merge_category_groups::demo::risk_flag::5::product": [0, 1],
                "group_merge_button_category_groups::demo::risk_flag::5::product": True,
            }
        }
    )

    assert st.session_state["group_merge_category_groups::demo::risk_flag::5::product"] == [0, 1]
    assert "group_merge_button_category_groups::demo::risk_flag::5::product" not in st.session_state


def test_candidate_cache_key_changes_with_variable_set():
    base = {
        "data_key": "data",
        "target": "risk_flag",
        "node_id": 0,
        "row_count": 1_000_000,
        "parameters": {"min_leaf": 20, "max_thresholds": 100},
        "max_rows": 50_000,
    }

    all_features = candidate_cache_key(features=["age", "income", "segment"], **base)
    reduced_features = candidate_cache_key(features=["age", "segment"], **base)

    assert all_features != reduced_features


def test_undo_preserves_cached_ranking_for_restored_leaf():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "risk_flag": ["low", "low", "high", "high"],
        }
    )
    init_tree(train)
    candidate = score_split(
        df=train,
        target="risk_flag",
        row_idx=train.index.tolist(),
        feature="x",
        split_type="numeric_le",
        value=2.5,
        min_leaf=1,
    )
    assert candidate is not None
    parameters = {
        "min_leaf": 1,
        "max_thresholds": 3,
        "max_numeric_bins": 2,
        "max_categories": 3,
        "max_category_groups": 2,
    }
    cache_key = candidate_cache_key(
        data_key="data",
        target="risk_flag",
        node_id=0,
        features=["x"],
        row_count=len(train),
        parameters=parameters,
        max_rows=len(train),
    )
    store_cached_candidates(
        "data",
        "risk_flag",
        0,
        cache_key,
        [candidate],
        analyzed_rows=len(train),
        full_rows=len(train),
    )

    split_node(train, 0, candidate, select_first_child=False)
    assert undo_last_split() is True

    assert get_cached_candidates("data", "risk_flag", 0, cache_key) == [candidate]


def test_candidate_splits_parallel_matches_serial():
    df = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45, 50, 55],
            "segment": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "risk_flag": ["low", "low", "low", "high", "high", "high", "high", "low"],
        }
    )
    kwargs = {
        "df": df,
        "target": "risk_flag",
        "features": ["age", "segment"],
        "row_idx": df.index.tolist(),
        "min_leaf": 1,
        "max_thresholds": 4,
        "max_categories": 4,
        "max_numeric_bins": 3,
        "max_category_groups": 3,
    }

    serial = candidate_splits(**kwargs, parallel_workers=1)
    parallel = candidate_splits(**kwargs, parallel_workers=2)

    assert [(item.feature, item.label, item.information_gain) for item in parallel] == [
        (item.feature, item.label, item.information_gain) for item in serial
    ]


def test_numeric_split_labels_missing_in_greater_branch():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, None, 4.0],
            "risk_flag": ["low", "low", "high", "high"],
        }
    )

    candidate = score_split(
        df=df,
        target="risk_flag",
        row_idx=df.index.tolist(),
        feature="x",
        split_type="numeric_le",
        value=2.5,
        min_leaf=1,
    )

    assert candidate is not None
    assert candidate.branch_labels == ("<= 2.5", "> 2.5 or missing")


def test_numeric_split_missing_policy_can_route_left_right_or_separate():
    df = pd.DataFrame(
        {
            "x": [1.0, None, 4.0, 5.0],
            "risk_flag": ["low", "high", "high", "high"],
        }
    )

    left = score_split(
        df=df,
        target="risk_flag",
        row_idx=df.index.tolist(),
        feature="x",
        split_type="numeric_le",
        value=2.5,
        min_leaf=1,
        missing_policy="left",
    )
    right = score_split(
        df=df,
        target="risk_flag",
        row_idx=df.index.tolist(),
        feature="x",
        split_type="numeric_le",
        value=2.5,
        min_leaf=1,
        missing_policy="right",
    )
    separate = score_split(
        df=df,
        target="risk_flag",
        row_idx=df.index.tolist(),
        feature="x",
        split_type="numeric_le",
        value=2.5,
        min_leaf=1,
        missing_policy="separate",
    )

    assert left is not None
    assert right is not None
    assert separate is not None
    assert left.branch_labels == ("<= 2.5 or missing", "> 2.5")
    assert right.branch_labels == ("<= 2.5", "> 2.5 or missing")
    assert separate.branch_labels == ("<= 2.5", "> 2.5", "missing")
    assert separate.missing_policy == "separate"
    assert split_branch_indices(df, df.index.tolist(), separate)[-1] == ("missing", [1])


def test_candidate_splits_scores_all_numeric_missing_policies():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, None, 4.0, 5.0, None],
            "risk_flag": ["low", "low", "high", "high", "high", "low"],
        }
    )

    candidates = candidate_splits(
        df=df,
        target="risk_flag",
        row_idx=df.index.tolist(),
        features=["x"],
        max_thresholds=3,
        max_categories=3,
        max_numeric_bins=3,
        max_category_groups=2,
        min_leaf=1,
        parallel_workers=1,
    )
    policies = {candidate.missing_policy for candidate in candidates if candidate.split_type == "numeric_le"}

    assert {"left", "right", "separate"}.issubset(policies)


def test_numeric_multiway_max_bins_counts_separate_missing_branch():
    df = pd.DataFrame(
        {
            "x": [float(i) for i in range(1, 81)] + [None] * 10,
            "risk_flag": (["low", "high"] * 40) + (["low", "high"] * 5),
        }
    )

    capped = candidate_splits(
        df=df,
        target="risk_flag",
        row_idx=df.index.tolist(),
        features=["x"],
        max_thresholds=3,
        max_categories=3,
        max_numeric_bins=4,
        max_category_groups=2,
        min_leaf=1,
        parallel_workers=1,
    )
    capped_multiway = [candidate for candidate in capped if candidate.split_type == "numeric_bins"]

    assert capped_multiway
    assert max(candidate.branch_count for candidate in capped_multiway) <= 4

    uncapped = candidate_splits(
        df=df,
        target="risk_flag",
        row_idx=df.index.tolist(),
        features=["x"],
        max_thresholds=3,
        max_categories=3,
        max_numeric_bins=5,
        max_category_groups=2,
        min_leaf=1,
        parallel_workers=1,
    )

    assert any(
        candidate.split_type == "numeric_bins"
        and candidate.missing_policy == "separate"
        and candidate.branch_count == 5
        for candidate in uncapped
    )


def test_evaluation_model_metrics_scores_test_dataframe():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "risk_flag": ["low", "low", "high", "high"],
        }
    )
    test = pd.DataFrame(
        {
            "x": [1.5, 3.5],
            "risk_flag": ["low", "high"],
        }
    )
    init_tree(train)
    candidate = score_split(
        df=train,
        target="risk_flag",
        row_idx=train.index.tolist(),
        feature="x",
        split_type="numeric_le",
        value=2.5,
        min_leaf=1,
    )
    assert candidate is not None
    split_node(train, 0, candidate, select_first_child=False)

    metrics = evaluation_model_metrics(train, test, "risk_flag", "Test")

    accuracy = metrics.loc[metrics["metric"] == "accuracy", "value"].iloc[0]
    assert float(accuracy) == 1.0


def test_model_metrics_includes_train_rows_and_scored_rows():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "risk_flag": ["low", "low", "high", "high"],
        }
    )
    init_tree(train)

    metrics = model_metrics(train, "risk_flag")

    values = metrics.set_index("metric")["value"].to_dict()
    assert values["rows"] == 4
    assert values["scored_rows"] == 4


def test_manual_prediction_threshold_changes_binary_leaf_prediction():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "risk_flag": ["high", "high", "high", "low", "low"],
        }
    )
    init_tree(train)
    st.session_state[PREDICTION_THRESHOLD_MODE_KEY] = PREDICTION_THRESHOLD_MANUAL
    st.session_state[PREDICTION_THRESHOLD_MANUAL_KEY] = 0.7

    predictions = tree_predictions(train, "risk_flag")

    assert set(predictions["positive_rate"]) == {0.6}
    assert set(predictions["prediction"]) == {"low"}


def test_best_f1_threshold_from_scores_selects_cutoff():
    summary = best_f1_threshold_from_scores(
        pd.Series([0, 1, 1, 1]),
        pd.Series([0.2, 0.4, 0.8, 0.9]),
    )

    assert summary["threshold"] == pytest.approx(0.4)
    assert summary["f1"] == pytest.approx(1.0)
    assert summary["precision"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(1.0)


def test_tree_export_records_prediction_threshold():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "risk_flag": ["high", "high", "high", "low", "low"],
        }
    )
    init_tree(train)
    st.session_state[PREDICTION_THRESHOLD_MODE_KEY] = PREDICTION_THRESHOLD_MANUAL
    st.session_state[PREDICTION_THRESHOLD_MANUAL_KEY] = 0.7

    payload = tree_export(train, "risk_flag", ["x"], parameters={})

    assert payload["prediction_threshold_mode"] == PREDICTION_THRESHOLD_MANUAL
    assert payload["prediction_threshold"] == pytest.approx(0.7)
    assert payload["tree"]["target_summary"]["prediction"] == "low"
    assert payload["tree"]["target_summary"]["default_rate"] == pytest.approx(0.6)


def test_model_performance_train_and_test_are_scored_independently():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "risk_flag": ["low", "low", "high", "high"],
        }
    )
    test = pd.DataFrame(
        {
            "x": [1.5, 3.5],
            "risk_flag": ["high", "high"],
        }
    )
    init_tree(train)
    candidate = score_split(
        df=train,
        target="risk_flag",
        row_idx=train.index.tolist(),
        feature="x",
        split_type="numeric_le",
        value=2.5,
        min_leaf=1,
    )
    assert candidate is not None
    split_node(train, 0, candidate, select_first_child=False)

    train_metrics = model_metrics(train, "risk_flag")
    train_metrics.insert(0, "dataset", "Train")
    test_metrics = evaluation_model_metrics(train, test, "risk_flag", "Test")
    wide = model_performance_wide_table(pd.concat([train_metrics, test_metrics], ignore_index=True))

    values = wide.set_index("metric")
    assert values.loc["rows", "Train"] == 4
    assert values.loc["rows", "Test"] == 2
    assert float(values.loc["accuracy", "Train"]) == 1.0
    assert float(values.loc["accuracy", "Test"]) == 0.5


def test_leaf_performance_rows_measure_test_when_eval_dataframe_is_present():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "risk_flag": ["low", "low", "high", "high"],
        }
    )
    test = pd.DataFrame(
        {
            "x": [1.5, 3.5],
            "risk_flag": ["high", "high"],
        }
    )
    init_tree(train)
    candidate = score_split(
        df=train,
        target="risk_flag",
        row_idx=train.index.tolist(),
        feature="x",
        split_type="numeric_le",
        value=2.5,
        min_leaf=1,
    )
    assert candidate is not None
    split_node(train, 0, candidate, select_first_child=False)

    rows = leaf_performance_rows(
        train,
        "risk_flag",
        "data",
        eval_df=test,
        dataset_name="Test",
    )

    by_leaf = {row["leaf"]: row for row in rows}
    assert by_leaf[1]["dataset"] == "Test"
    assert by_leaf[1]["n"] == 1
    assert by_leaf[1]["predict"] == "low"
    assert by_leaf[1]["default_rate"] == 1.0
    assert by_leaf[1]["measurement_default_rate"] == 1.0
    assert by_leaf[1]["train_default_rate"] == 0.0
    assert by_leaf[1]["prediction_threshold"] == pytest.approx(0.5)
    assert by_leaf[1]["prediction_margin"] == pytest.approx(-0.5)
    assert by_leaf[2]["n"] == 1
    assert by_leaf[2]["predict"] == "high"


def test_max_f1_prediction_threshold_assigns_positive_at_cutoff():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "risk_flag": ["high", "low", "high", "high"],
        }
    )
    init_tree(train)
    candidate = score_split(
        df=train,
        target="risk_flag",
        row_idx=train.index.tolist(),
        feature="x",
        split_type="numeric_le",
        value=2.5,
        min_leaf=1,
    )
    assert candidate is not None
    split_node(train, 0, candidate, select_first_child=False)
    st.session_state[PREDICTION_THRESHOLD_MODE_KEY] = PREDICTION_THRESHOLD_F1

    rows = leaf_performance_rows(train, "risk_flag", "data")
    by_leaf = {row["leaf"]: row for row in rows}

    assert by_leaf[1]["train_default_rate"] == pytest.approx(0.5)
    assert by_leaf[1]["prediction_threshold"] == pytest.approx(0.5)
    assert by_leaf[1]["prediction_margin"] == pytest.approx(0.0)
    assert by_leaf[1]["predict"] == "high"
    assert by_leaf[2]["train_default_rate"] == pytest.approx(1.0)
    assert by_leaf[2]["predict"] == "high"


def test_candidate_validation_stats_blocks_test_gini_drop():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "risk_flag": ["low", "low", "high", "high"],
        }
    )
    test = pd.DataFrame(
        {
            "x": [1.5, 3.5],
            "risk_flag": ["high", "low"],
        }
    )
    init_tree(train)
    candidate = score_split(
        df=train,
        target="risk_flag",
        row_idx=train.index.tolist(),
        feature="x",
        split_type="numeric_le",
        value=2.5,
        min_leaf=1,
    )
    assert candidate is not None

    stats = candidate_validation_stats(
        train_df=train,
        test_df=test,
        target="risk_flag",
        node_id=0,
        candidate=candidate,
        max_gini_gap=0.1,
    )

    assert stats is not None
    assert stats["test_gini_delta"] < 0
    assert stats["validation_safe"] is False


def test_fast_candidate_validation_matches_prediction_path():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "risk_flag": ["low", "low", "high", "high", "high", "low"],
        }
    )
    test = pd.DataFrame(
        {
            "x": [1.5, 2.5, 3.5, 4.5, 5.5],
            "risk_flag": ["low", "high", "high", "low", "high"],
        }
    )
    init_tree(train)
    candidate = score_split(
        df=train,
        target="risk_flag",
        row_idx=train.index.tolist(),
        feature="x",
        split_type="numeric_le",
        value=3.0,
        min_leaf=1,
    )
    assert candidate is not None

    slow_stats = candidate_validation_stats(
        train_df=train,
        test_df=test,
        target="risk_flag",
        node_id=0,
        candidate=candidate,
        max_gini_gap=0.1,
    )
    state = build_binary_validation_state(
        train,
        test,
        "risk_flag",
        {0: test.index.tolist()},
    )
    fast_stats = candidate_validation_stats_from_state(
        state=state,
        node_id=0,
        candidate=candidate,
        max_gini_gap=0.1,
    )

    assert fast_stats is not None
    assert slow_stats is not None
    for key in [
        "train_gini_before",
        "test_gini_before",
        "train_gini_after",
        "test_gini_after",
        "train_gini_delta",
        "test_gini_delta",
        "gini_gap_before",
        "gini_gap_after",
    ]:
        assert fast_stats[key] == pytest.approx(slow_stats[key])
    assert fast_stats["validation_safe"] == slow_stats["validation_safe"]


def test_build_optimal_tree_penalizes_but_allows_validation_unsafe_split():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "risk_flag": ["low", "low", "high", "high"],
        }
    )
    test = pd.DataFrame(
        {
            "x": [1.0, 1.5, 2.0, 3.0, 3.5, 4.0],
            "risk_flag": ["low", "low", "high", "low", "high", "high"],
        }
    )

    split_count = build_optimal_tree(
        df=train,
        target="risk_flag",
        features=["x"],
        ranked_feature_limit=1,
        test_df=test,
        min_leaf=1,
        max_thresholds=3,
        max_categories=3,
        max_numeric_bins=2,
        max_category_groups=2,
        max_depth=1,
        max_leaves=2,
        min_information_gain=0.0,
        candidate_rows=len(train),
        parallel_workers=1,
        max_validation_gini_gap=0.1,
    )

    assert split_count == 1
    assert st.session_state.tree[0]["split"] is not None
    assert st.session_state.auto_tree_diagnostics["stop_reason"] == "max_leaves"
    assert st.session_state.auto_tree_diagnostics["leaf_count"] == 2


def test_build_optimal_tree_stops_when_validation_score_would_fall():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "risk_flag": ["low", "low", "high", "high"],
        }
    )
    test = pd.DataFrame(
        {
            "x": [1.5, 3.5],
            "risk_flag": ["high", "low"],
        }
    )

    split_count = build_optimal_tree(
        df=train,
        target="risk_flag",
        features=["x"],
        ranked_feature_limit=1,
        test_df=test,
        min_leaf=1,
        max_thresholds=3,
        max_categories=3,
        max_numeric_bins=2,
        max_category_groups=2,
        max_depth=1,
        max_leaves=2,
        min_information_gain=0.0,
        candidate_rows=len(train),
        parallel_workers=1,
        max_validation_gini_gap=0.1,
    )

    assert split_count == 0
    assert st.session_state.tree[0]["split"] is None
    assert st.session_state.auto_tree_diagnostics["stop_reason"] == "no_validation_improvement"
    assert st.session_state.auto_tree_diagnostics["leaf_count"] == 1


def test_build_optimal_tree_records_zero_split_diagnostics_when_gain_is_too_low():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x": list(range(100)),
            "risk_flag": ["high" if i % 2 else "low" for i in range(100)],
        }
    )

    split_count = build_optimal_tree(
        df=train,
        target="risk_flag",
        features=["x"],
        ranked_feature_limit=1,
        test_df=None,
        min_leaf=1,
        max_thresholds=5,
        max_categories=3,
        max_numeric_bins=2,
        max_category_groups=2,
        max_depth=1,
        max_leaves=2,
        min_information_gain=1.0,
        candidate_rows=len(train),
        parallel_workers=1,
        max_validation_gini_gap=0.1,
    )

    diagnostics = st.session_state.auto_tree_diagnostics
    assert split_count == 0
    assert diagnostics["candidate_count"] > 0
    assert diagnostics["best_candidate"]["feature"] == "x"
    assert diagnostics["best_candidate"]["information_gain"] < 1.0


def test_build_optimal_tree_reranks_limited_features_per_leaf():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "x_root": [0, 0, 0, 0, 1, 1, 1, 1],
            "y_child": [0, 0, 1, 1, 0, 0, 1, 1],
            "risk_flag": ["low", "low", "high", "high", "high", "high", "high", "high"],
        }
    )

    split_count = build_optimal_tree(
        df=train,
        target="risk_flag",
        features=["x_root", "y_child"],
        ranked_feature_limit=1,
        test_df=None,
        min_leaf=1,
        max_thresholds=3,
        max_categories=3,
        max_numeric_bins=2,
        max_category_groups=2,
        max_depth=2,
        max_leaves=4,
        min_information_gain=0.0,
        candidate_rows=len(train),
        parallel_workers=1,
        max_validation_gini_gap=0.1,
    )

    assert split_count == 2
    assert st.session_state.tree[0]["split"]["feature"] == "x_root"
    assert any(
        node.get("split", {}).get("feature") == "y_child"
        for node in st.session_state.tree.values()
        if node["id"] != 0
    )


def test_build_optimal_tree_can_continue_from_existing_tree():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "gate": [0, 0, 0, 0, 1, 1, 1, 1],
            "x": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "risk_flag": ["low", "low", "high", "high", "low", "low", "high", "high"],
        }
    )
    init_tree(train)
    manual_candidate = score_split(
        df=train,
        target="risk_flag",
        row_idx=train.index.tolist(),
        feature="gate",
        split_type="numeric_le",
        value=0.5,
        min_leaf=1,
    )
    assert manual_candidate is not None
    split_node(train, 0, manual_candidate, select_first_child=False)

    split_count = build_optimal_tree(
        df=train,
        target="risk_flag",
        features=["x"],
        ranked_feature_limit=1,
        test_df=None,
        min_leaf=1,
        max_thresholds=3,
        max_categories=3,
        max_numeric_bins=2,
        max_category_groups=2,
        max_depth=2,
        max_leaves=4,
        min_information_gain=0.0,
        candidate_rows=len(train),
        parallel_workers=1,
        max_validation_gini_gap=0.1,
        reset_tree=False,
    )

    assert split_count == 2
    assert st.session_state.tree[0]["split"]["feature"] == "gate"
    assert {
        node["split"]["feature"]
        for node in st.session_state.tree.values()
        if node["id"] != 0 and node.get("split") is not None
    } == {"x"}

    assert undo_last_split() is True
    assert st.session_state.tree[0]["split"]["feature"] == "gate"
    assert all(node.get("split") is None for node in st.session_state.tree.values() if node["id"] != 0)
    assert st.session_state.split_history == [0]


def test_build_optimal_tree_reset_true_rebuilds_from_root():
    st.session_state.clear()
    train = pd.DataFrame(
        {
            "gate": [0, 0, 0, 0, 1, 1, 1, 1],
            "x": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "risk_flag": ["low", "low", "high", "high", "low", "low", "high", "high"],
        }
    )
    init_tree(train)
    manual_candidate = score_split(
        df=train,
        target="risk_flag",
        row_idx=train.index.tolist(),
        feature="gate",
        split_type="numeric_le",
        value=0.5,
        min_leaf=1,
    )
    assert manual_candidate is not None
    split_node(train, 0, manual_candidate, select_first_child=False)

    split_count = build_optimal_tree(
        df=train,
        target="risk_flag",
        features=["x"],
        ranked_feature_limit=1,
        test_df=None,
        min_leaf=1,
        max_thresholds=3,
        max_categories=3,
        max_numeric_bins=2,
        max_category_groups=2,
        max_depth=1,
        max_leaves=2,
        min_information_gain=0.0,
        candidate_rows=len(train),
        parallel_workers=1,
        max_validation_gini_gap=0.1,
    )

    assert split_count == 1
    assert st.session_state.tree[0]["split"]["feature"] == "x"
    assert len(st.session_state.tree) == 3
    assert st.session_state.auto_tree_diagnostics["stop_reason"] == "max_leaves"
    assert st.session_state.auto_tree_diagnostics["leaf_count"] == 2

    assert undo_last_split() is True
    assert st.session_state.tree[0]["split"] is None
    assert st.session_state.split_history == []


def test_restore_checkpoint_dataframe_uses_session_snapshot_without_embedded_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERACTIVE_TREE_SESSION_DIR", str(tmp_path))
    df = pd.DataFrame({"age": [30, 40], "risk_flag": [0, 1]})
    data_id, _ = save_dataframe_session(df, target="risk_flag", features=["age"], name="upload")
    checkpoint = {
        "data": {
            "source": "uploaded",
            "name": "upload.csv",
            "data_id": data_id,
            "data_key": "uploaded-key",
            "frame_json_omitted": True,
        }
    }

    restored = restore_checkpoint_dataframe(checkpoint)

    assert restored is not None
    restored_df, restored_name, restored_key = restored
    pd.testing.assert_frame_equal(restored_df, df)
    assert restored_name == "upload.csv"
    assert restored_key == "uploaded-key"


def test_restore_tree_state_ignores_checkpoint_without_tree_state_key():
    st.session_state.clear()
    df = pd.DataFrame({"x": [1.0, 2.0], "risk_flag": ["low", "high"]})
    checkpoint = {
        "tree_schema_version": 4,
        "tree_state": {
            "state_key": None,
            "tree": {},
        },
    }

    restored = restore_tree_state_from_checkpoint(checkpoint, ("data", "risk_flag", 4), df)

    assert restored is False
    assert "tree" not in st.session_state


def test_restore_tree_state_rejects_stale_auto_tree_with_validation():
    st.session_state.clear()
    df = pd.DataFrame({"x": [1.0, 2.0], "risk_flag": ["low", "high"]})
    checkpoint = {
        "tree_schema_version": 4,
        "data": {"metadata": {"validation": {"mode": "split_train_data"}}},
        "tree_state": {
            "state_key": ["data", "risk_flag", 4],
            "auto_tree_message": "Optimal tree rebuilt from root with 1 split(s). Reached 2/2 leaves.",
            "tree": {
                "0": {
                    "id": 0,
                    "depth": 0,
                    "path": "root",
                    "row_idx": [0, 1],
                    "split": {
                        "feature": "x",
                        "split_type": "numeric_le",
                        "value": 1.5,
                        "label": "x <= 1.5",
                        "branch_count": 2,
                        "branch_labels": ["<= 1.5", "> 1.5"],
                        "information_gain": 1.0,
                        "weighted_entropy": 0.0,
                    },
                    "children": [{"id": 1, "label": "<= 1.5"}, {"id": 2, "label": "> 1.5"}],
                    "left": 1,
                    "right": 2,
                },
                "1": {"id": 1, "depth": 1, "path": "root -> x <= 1.5", "row_idx": [0], "split": None, "children": [], "left": None, "right": None},
                "2": {"id": 2, "depth": 1, "path": "root -> x > 1.5", "row_idx": [1], "split": None, "children": [], "left": None, "right": None},
            },
        },
    }

    restored = restore_tree_state_from_checkpoint(checkpoint, ("data", "risk_flag", 4), df)

    assert restored is False
    assert "tree" not in st.session_state
    assert "older generalization logic" in st.session_state.auto_tree_message


def test_restore_tree_state_accepts_current_auto_tree_generalization_version():
    st.session_state.clear()
    df = pd.DataFrame({"x": [1.0, 2.0], "risk_flag": ["low", "high"]})
    checkpoint = {
        "tree_schema_version": 4,
        "data": {"metadata": {"validation": {"mode": "split_train_data"}}},
        "tree_state": {
            "state_key": ["data", "risk_flag", 4],
            "next_node_id": 3,
            "current_node_id": 0,
            "split_history": [0],
            "split_action_history": [[0]],
            "auto_tree_message": "Optimal tree rebuilt from root with 1 split(s). Reached 2/2 leaves.",
            "auto_tree_diagnostics": {"generalization_version": AUTO_TREE_GENERALIZATION_VERSION},
            "tree": {
                "0": {
                    "id": 0,
                    "depth": 0,
                    "path": "root",
                    "row_idx": [0, 1],
                    "split": {
                        "feature": "x",
                        "split_type": "numeric_le",
                        "value": 1.5,
                        "label": "x <= 1.5",
                        "branch_count": 2,
                        "branch_labels": ["<= 1.5", "> 1.5"],
                        "information_gain": 1.0,
                        "weighted_entropy": 0.0,
                    },
                    "children": [{"id": 1, "label": "<= 1.5"}, {"id": 2, "label": "> 1.5"}],
                    "left": 1,
                    "right": 2,
                },
                "1": {"id": 1, "depth": 1, "path": "root -> x <= 1.5", "row_idx": [0], "split": None, "children": [], "left": None, "right": None},
                "2": {"id": 2, "depth": 1, "path": "root -> x > 1.5", "row_idx": [1], "split": None, "children": [], "left": None, "right": None},
            },
        },
    }

    restored = restore_tree_state_from_checkpoint(checkpoint, ("data", "risk_flag", 4), df)

    assert restored is True
    assert st.session_state.tree[0]["split"]["feature"] == "x"
    assert st.session_state.auto_tree_diagnostics["generalization_version"] == AUTO_TREE_GENERALIZATION_VERSION


def test_generalized_split_selection_penalizes_large_validation_gap():
    balanced_choice = {
        "candidate": "balanced",
        "test_gini_after": 0.673,
        "gini_gap_after": 0.002,
        "validation_score": 0.673,
        "raw_score": (0.1,),
    }
    leaky_choice = {
        "candidate": "leaky",
        "test_gini_after": 0.680,
        "gini_gap_after": 0.035,
        "validation_score": 0.680,
        "raw_score": (0.2,),
    }

    selected = choose_generalized_split_candidate([balanced_choice, leaky_choice])

    assert selected is balanced_choice


def test_session_auto_tree_restore_only_flags_stale_optimal_tree():
    st.session_state.clear()
    st.session_state.tree = {
        0: {"id": 0, "split": {"feature": "x"}},
        1: {"id": 1, "split": None},
    }
    st.session_state.auto_tree_message = "Optimal tree rebuilt from root with 1 split(s)."
    st.session_state.auto_tree_diagnostics = {"generalization_version": AUTO_TREE_GENERALIZATION_VERSION - 1}

    assert session_auto_tree_needs_generalization_restore() is True

    st.session_state.auto_tree_diagnostics = {"generalization_version": AUTO_TREE_GENERALIZATION_VERSION}
    assert session_auto_tree_needs_generalization_restore() is False

    st.session_state.auto_tree_message = ""
    st.session_state.auto_tree_diagnostics = {}
    assert session_auto_tree_needs_generalization_restore() is False
