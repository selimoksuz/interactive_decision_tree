from __future__ import annotations

import pandas as pd
import streamlit as st

from interactive_decision_tree.session_store import save_dataframe_session
from interactive_decision_tree_app import (
    DEFAULT_DEMO_ROWS,
    analysis_row_idx,
    build_optimal_tree,
    candidate_cache_key,
    candidate_splits,
    candidate_validation_stats,
    checkpoint_ui_state,
    evaluation_model_metrics,
    init_tree,
    leaf_performance_rows,
    make_demo_data,
    model_metrics,
    model_performance_wide_table,
    restore_checkpoint_dataframe,
    restore_checkpoint_ui_state,
    score_split,
    split_node,
    train_test_split_indices,
    validate_test_dataframe,
)


def test_demo_data_default_is_large_enough_for_train_test_validation():
    assert len(make_demo_data()) == DEFAULT_DEMO_ROWS
    assert DEFAULT_DEMO_ROWS >= 5_000


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
    assert by_leaf[2]["n"] == 1
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


def test_build_optimal_tree_skips_validation_unsafe_split():
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
        max_validation_gini_gap_increase=0.0,
    )

    assert split_count == 0
    assert st.session_state.tree[0]["split"] is None


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
