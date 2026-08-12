from __future__ import annotations

import copy
import io
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import interactive_decision_tree.woe_binning as woe_binning
from interactive_decision_tree.woe_binning import (
    WOE_MAX_BINS,
    WoeBuildConfig,
    apply_bin_table_edits,
    apply_numeric_cutpoints,
    bin_display_label,
    build_initial_spec,
    evaluate_spec,
    merge_selected_bins,
    numeric_bin_label,
    parse_special_values,
    set_assigned_woe_from_table,
)
from interactive_decision_tree.woe_export import (
    build_project_export,
    dc_corp_woe_artifacts,
    project_excel_bytes,
    project_json_bytes,
    project_pickle_bytes,
)
from interactive_decision_tree.woe_ui import (
    apply_current_bin_changes,
    current_bin_editor_columns,
    filter_variable_options,
    metrics_frame,
    normalize_variable_selection,
    scoped_woe_project,
    update_variable_selection_for_filtered,
    variable_editor_order,
)


def woe_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [22, 24, 27, 33, 38, 45, 51, 59, None, -999],
            "segment": ["A", "A", "B", "B", "C", "C", "D", "D", "", "SPECIAL"],
            "target": [1, 1, 0, 1, 0, 0, 0, 0, 1, 1],
        }
    )


def test_woe_variable_filter_supports_contains_multiple_terms_and_wildcards():
    variables = ["age", "income", "avg_balance_3m", "risk_score", "segment"]

    assert filter_variable_options(variables, "bal") == ["avg_balance_3m"]
    assert filter_variable_options(variables, "age,score") == ["age", "risk_score"]
    assert filter_variable_options(variables, "inc%") == ["income"]
    assert filter_variable_options(variables, "*ment") == ["segment"]


def test_woe_metrics_frame_uses_arrow_safe_display_strings():
    frame = metrics_frame({"export_iv": 0.1234567, "is_monotonic": True, "engine_used": "optbinning"})

    values = dict(frame.to_dict("split")["data"])
    assert str(frame["value"].dtype) == "string"
    assert values["export_iv"] == "0.123457"
    assert values["is_monotonic"] == "Yes"
    assert values["engine_used"] == "optbinning"


def test_woe_variable_selection_actions_preserve_order():
    variables = ["age", "income", "segment", "score"]

    assert update_variable_selection_for_filtered(variables, ["income"], ["age", "score"], True) == [
        "age",
        "income",
        "score",
    ]
    assert update_variable_selection_for_filtered(variables, ["age", "income", "score"], ["age", "score"], False) == [
        "income"
    ]
    assert normalize_variable_selection(variables, ["income", "missing", "age"]) == ["income", "age"]


def test_scoped_woe_project_limits_visible_variables_without_copying_states():
    age_state = {"name": "age"}
    segment_state = {"name": "segment"}
    project = {"project_key": "unit", "variables": {"age": age_state, "segment": segment_state}}

    scoped = scoped_woe_project(project, ["segment"])

    assert set(scoped["variables"]) == {"segment"}
    assert scoped["variables"]["segment"] is segment_state
    assert set(project["variables"]) == {"age", "segment"}


def test_numeric_woe_supports_special_missing_and_manual_woe():
    df = woe_df()
    config = WoeBuildConfig(max_bins=3, special_values=("-999",), engine="fallback")
    spec = build_initial_spec(df, "target", "age", 1, config)

    assert any(bin_spec["kind"] == "missing" for bin_spec in spec["bins"])
    assert any(bin_spec["kind"] == "special" for bin_spec in spec["bins"])

    report = evaluate_spec(df, "target", spec, 1)
    assert {"event_count", "non_event_count", "calculated_woe", "export_woe"}.issubset(report["table"].columns)
    assert report["metrics"]["bin_count"] >= 3

    edited = report["table"].copy()
    edited.loc[edited["kind"] == "special", "assigned_woe"] = -1.25
    updated = set_assigned_woe_from_table(spec, edited)
    updated_report = evaluate_spec(df, "target", updated, 1)
    special_export = updated_report["table"].loc[updated_report["table"]["kind"] == "special", "export_woe"].iloc[0]
    assert special_export == -1.25
    assert updated_report["metrics"]["manual_woe_bins"] == 1


def test_woe_evaluation_adds_binomial_and_hhi_tests():
    df = woe_df()
    spec = build_initial_spec(
        df,
        "target",
        "age",
        1,
        WoeBuildConfig(max_bins=3, binomial_confidence_level=0.90, engine="fallback"),
    )

    report = evaluate_spec(df, "target", spec, 1)
    table = report["table"]
    metrics = report["metrics"]

    assert {
        "expected_event_rate",
        "expected_event_count",
        "event_count_delta",
        "binomial_p_value",
        "binomial_adjusted_p_value",
        "binomial_significant",
        "binomial_pass",
        "binomial_result",
        "binomial_one_tail_p_value",
        "binomial_one_tail_adjusted_p_value",
        "binomial_one_tail_pass",
        "binomial_one_tail_result",
        "binomial_ci_lower",
        "binomial_ci_upper",
        "bucket_weight",
        "variable_avg_value",
        "variable_avg_event_rate",
        "hhi_contribution",
        "bucket_hhi",
    }.issubset(table.columns)
    assert table["bucket_weight"].sum() == pytest.approx(1.0)
    assert table["bucket_weight"].equals(table["all_concentration"])
    assert metrics["hhi_total"] == pytest.approx(float((table["bucket_weight"] ** 2).sum()))
    assert metrics["average_bucket_hhi"] == pytest.approx(
        metrics["hhi_total"] / metrics["binomial_family_size"]
    )
    assert 0.0 <= metrics["hhi_total"] <= 1.0
    assert metrics["hhi_concentration"] == woe_binning.hhi_concentration_label(metrics["hhi_total"])
    assert 0.0 <= metrics["binomial_signal_share"] <= 1.0
    assert metrics["binomial_family_size"] == int((table["count"] > 0).sum())
    assert metrics["binomial_confidence_level"] == pytest.approx(0.90)
    assert metrics["binomial_family_alpha"] == pytest.approx(0.10)
    assert metrics["binomial_alternative"] == "two-sided"
    assert metrics["binomial_test_method"] == "central_exact_binomial"
    assert metrics["binomial_one_tail_alternative"] == "greater"
    assert metrics["binomial_one_tail_test_method"] == "exact_binomial_upper_tail"
    assert metrics["binomial_multiple_testing"] == "bonferroni"
    assert metrics["variable_avg_event_rate"] == pytest.approx(df["target"].mean())
    assert metrics["variable_avg_value"] == pytest.approx(df["age"].mean())
    normal_rows = table[table["kind"] == "normal"]
    assert normal_rows["variable_avg_value"].notna().all()
    assert set(table["binomial_result"]) <= {"Pass", "Reject", "No data"}
    assert set(table["binomial_one_tail_result"]) <= {"Pass", "Reject", "No data"}
    assert table["bucket_hhi"].equals(table["hhi_contribution"])

    first = table.iloc[0]
    expected_two_tail = min(
        1.0,
        2.0
        * min(
            stats.binom.cdf(first["event_count"], first["count"], metrics["variable_avg_event_rate"]),
            stats.binom.sf(first["event_count"] - 1, first["count"], metrics["variable_avg_event_rate"]),
        ),
    )
    expected_one_tail = stats.binom.sf(
        first["event_count"] - 1,
        first["count"],
        metrics["variable_avg_event_rate"],
    )
    assert first["binomial_p_value"] == pytest.approx(expected_two_tail)
    assert first["binomial_one_tail_p_value"] == pytest.approx(expected_one_tail)

    visible_columns = current_bin_editor_columns("numeric")
    assert visible_columns[:4] == ["merge", "label", "lower", "upper"]
    assert {
        "count",
        "event_count",
        "non_event_count",
        "bucket_weight",
        "event_rate",
        "variable_avg_value",
        "binomial_result",
        "binomial_one_tail_result",
        "bucket_hhi",
    } <= set(visible_columns)
    assert {"binomial_p_value", "binomial_ci_lower", "binomial_ci_upper"}.isdisjoint(visible_columns)


def test_woe_max_bins_supports_more_than_twenty_normal_bins():
    df = pd.DataFrame({"feature": np.arange(500), "target": np.tile([0, 1], 250)})

    spec = build_initial_spec(
        df,
        "target",
        "feature",
        1,
        WoeBuildConfig(max_bins=25, min_bin_size=0.01, engine="fallback"),
    )

    assert WOE_MAX_BINS == 100
    assert sum(bin_spec["kind"] == "normal" for bin_spec in spec["bins"]) == 25


def test_auto_optbinning_raises_when_engine_unavailable(monkeypatch):
    def fail_loader():
        raise RuntimeError("missing optbinning package")

    monkeypatch.setattr(woe_binning, "_load_optimal_binning_class", fail_loader)

    with pytest.raises(RuntimeError, match="missing optbinning package"):
        build_initial_spec(woe_df(), "target", "age", 1, WoeBuildConfig(engine="auto"))


def test_forced_optbinning_raises_when_engine_unavailable(monkeypatch):
    def fail_loader():
        raise RuntimeError("missing optbinning package")

    monkeypatch.setattr(woe_binning, "_load_optimal_binning_class", fail_loader)

    with pytest.raises(RuntimeError, match="missing optbinning package"):
        build_initial_spec(woe_df(), "target", "age", 1, WoeBuildConfig(engine="optbinning"))


def test_forced_optbinning_uses_optbinning_when_available():
    pytest.importorskip("optbinning")
    status = woe_binning.optbinning_status()
    if not status["available"]:
        pytest.skip(str(status["error"]))

    rng = np.random.default_rng(7)
    x = rng.normal(size=400)
    p = 1 / (1 + np.exp(-2 * x))
    df = pd.DataFrame({"score": x, "target": rng.binomial(1, p)})

    spec = build_initial_spec(df, "target", "score", 1, WoeBuildConfig(engine="optbinning"))

    assert spec["config"]["engine_used"] == "optbinning"
    assert spec["config"]["engine_fallback_reason"] is None


def test_forced_optbinning_groups_categorical_by_event_profile_when_available():
    pytest.importorskip("optbinning")
    status = woe_binning.optbinning_status()
    if not status["available"]:
        pytest.skip(str(status["error"]))

    df = pd.DataFrame(
        {
            "segment": list("AAAAABBBBBCCCCCDDDDDEEEEEFFFFFGGGGGHHHHH"),
            "target": [
                0, 0, 0, 0, 1,
                0, 0, 1, 1, 1,
                0, 1, 1, 1, 1,
                0, 0, 0, 1, 1,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 0,
                1, 1, 0, 0, 1,
                0, 1, 0, 1, 0,
            ],
        }
    )

    spec = build_initial_spec(df, "target", "segment", 1, WoeBuildConfig(max_bins=4, engine="optbinning"))
    groups = [bin_spec["values"] for bin_spec in spec["bins"] if bin_spec["kind"] == "normal"]

    assert spec["config"]["engine_used"] == "optbinning"
    assert len(groups) == 4
    assert any(set(group) == {"F", "A"} for group in groups)
    assert any(set(group) == {"C", "E"} for group in groups)


def test_apply_numeric_cutpoints_rebuilds_normal_bins():
    df = woe_df()
    spec = build_initial_spec(df, "target", "age", 1, WoeBuildConfig(engine="fallback"))
    updated = apply_numeric_cutpoints(spec, [30, 50])
    normal_labels = [bin_spec["label"] for bin_spec in updated["bins"] if bin_spec["kind"] == "normal"]

    assert normal_labels == ["(-inf, 30]", "(30, 50]", "(50, inf)"]


def test_numeric_bin_label_avoids_scientific_notation():
    assert numeric_bin_label(33280.0, 43650.0) == "(33280, 43650]"


def test_numeric_table_reformats_stale_scientific_interval_label():
    df = pd.DataFrame(
        {
            "income": [20_000.0, 35_000.0, 50_000.0, 70_000.0],
            "target": [1, 0, 0, 1],
        }
    )
    spec = build_initial_spec(df, "target", "income", 1, WoeBuildConfig(engine="fallback"))
    spec = apply_numeric_cutpoints(spec, [30000.0, 44090.0, 60000.0])
    normal = [bin_spec for bin_spec in spec["bins"] if bin_spec["kind"] == "normal"]
    normal[1]["lower"] = 33520.0
    normal[1]["upper"] = 44090.0
    normal[1]["label"] = "(3.352e+04, 4.409e+04]"

    table = evaluate_spec(df, "target", spec, 1)["table"]
    middle_label = table.loc[table["bin_id"] == normal[1]["bin_id"], "label"].iloc[0]

    assert bin_display_label(spec, normal[1]) == "(33520, 44090]"
    assert middle_label == "(33520, 44090]"
    assert "e+" not in middle_label


def test_numeric_table_edits_can_change_shared_boundary_from_lower_or_upper():
    df = woe_df()
    spec = build_initial_spec(df, "target", "age", 1, WoeBuildConfig(engine="fallback"))
    spec = apply_numeric_cutpoints(spec, [30, 50])
    table = evaluate_spec(df, "target", spec, 1)["table"]

    edited = table.copy()
    edited.loc[edited["bin_id"] == "b002", "lower"] = 35
    updated = apply_bin_table_edits(spec, edited)
    labels = [bin_spec["label"] for bin_spec in updated["bins"] if bin_spec["kind"] == "normal"]

    assert labels == ["(-inf, 35]", "(35, 50]", "(50, inf)"]


def test_numeric_table_edits_reject_mismatched_adjacent_boundaries():
    df = woe_df()
    spec = build_initial_spec(df, "target", "age", 1, WoeBuildConfig(engine="fallback"))
    spec = apply_numeric_cutpoints(spec, [30, 50])
    table = evaluate_spec(df, "target", spec, 1)["table"]

    edited = table.copy()
    edited.loc[edited["bin_id"] == "b001", "upper"] = 34
    edited.loc[edited["bin_id"] == "b002", "lower"] = 35

    try:
        apply_bin_table_edits(spec, edited)
    except ValueError as exc:
        assert "boundary mismatch" in str(exc)
    else:
        raise AssertionError("Expected mismatched numeric boundary to be rejected")


def test_merge_selected_bins_numeric_can_merge_previous_with_current():
    df = woe_df()
    spec = build_initial_spec(df, "target", "age", 1, WoeBuildConfig(engine="fallback"))
    spec = apply_numeric_cutpoints(spec, [30, 50])

    updated = merge_selected_bins(spec, ["b001", "b002"])
    labels = [bin_spec["label"] for bin_spec in updated["bins"] if bin_spec["kind"] == "normal"]

    assert labels == ["(-inf, 50]", "(50, inf)"]


def test_apply_current_bin_changes_combines_table_edits_and_merge():
    df = woe_df()
    spec = build_initial_spec(df, "target", "age", 1, WoeBuildConfig(engine="fallback"))
    spec = apply_numeric_cutpoints(spec, [30, 50])
    table = evaluate_spec(df, "target", spec, 1)["table"]
    edited = table.copy()
    edited.loc[edited["bin_id"] == "b001", "assigned_woe"] = -0.5
    state = {"name": "age", "current_spec": spec}

    actions = apply_current_bin_changes(
        state,
        edited,
        ["b002", "b003"],
        {"kind": "numeric", "changed": False, "cutpoints": [30, 50]},
    )

    normal = [bin_spec for bin_spec in state["current_spec"]["bins"] if bin_spec["kind"] == "normal"]
    assert actions == ["apply_table_edits", "merge_selected_bins"]
    assert len(normal) == 2
    assert normal[0]["assigned_woe"] == -0.5
    assert state["status"] == "edited"
    assert state["edits"][-1]["action"] == "apply_current_bin_changes"


def test_merge_selected_bins_categorical_can_merge_non_adjacent_groups():
    df = woe_df()
    spec = build_initial_spec(df, "target", "segment", 1, WoeBuildConfig(max_bins=4, engine="fallback"))

    updated = merge_selected_bins(spec, ["b001", "b003"])
    normal_values = [bin_spec["values"] for bin_spec in updated["bins"] if bin_spec["kind"] == "normal"]

    assert normal_values[0]
    assert len(normal_values) == 3


def test_categorical_merge_fast_profile_matches_full_scan_evaluation():
    df = pd.DataFrame(
        {
            "segment": ["A", "B", "C", "D", "E", "F", None, ""] * 20,
            "target": [1, 1, 0, 0, 1, 0, 1, 0] * 20,
        }
    )
    spec = build_initial_spec(df, "target", "segment", 1, WoeBuildConfig(max_bins=4, engine="fallback"))
    merged = merge_selected_bins(spec, ["b001", "b003"])
    scan_spec = copy.deepcopy(merged)
    scan_spec.pop("evaluation_profile", None)

    fast = evaluate_spec(df, "target", merged, 1)
    scanned = evaluate_spec(df, "target", scan_spec, 1)

    pd.testing.assert_frame_equal(
        fast["table"].reset_index(drop=True),
        scanned["table"].reset_index(drop=True),
        check_dtype=False,
    )
    assert fast["metrics"]["export_iv"] == scanned["metrics"]["export_iv"]
    assert fast["metrics"]["export_gini"] == scanned["metrics"]["export_gini"]


def test_project_export_contains_integrated_mapping():
    df = woe_df()
    age_spec = build_initial_spec(df, "target", "age", 1, WoeBuildConfig(engine="fallback"))
    segment_spec = build_initial_spec(df, "target", "segment", 1, WoeBuildConfig(engine="fallback"))
    project = {
        "data_key": "unit-test",
        "variables": {
            "age": {"name": "age", "status": "approved", "original_spec": age_spec, "current_spec": age_spec},
            "segment": {
                "name": "segment",
                "status": "approved",
                "original_spec": segment_spec,
                "current_spec": segment_spec,
            },
        },
    }

    payload = build_project_export(project, df, None, "target", 1)
    encoded = project_json_bytes(payload)
    decoded = json.loads(encoded.decode("utf-8"))

    assert decoded["format"] == "interactive_woe_mapping"
    assert decoded["variable_count"] == 2
    assert {variable["name"] for variable in decoded["variables"]} == {"age", "segment"}
    age_payload = next(variable for variable in decoded["variables"] if variable["name"] == "age")
    assert "hhi_total" in age_payload["metrics"]
    assert "binomial_signal_share" in age_payload["metrics"]
    assert age_payload["metrics"]["binomial_confidence_level"] == pytest.approx(0.95)
    assert age_payload["metrics"]["binomial_alternative"] == "two-sided"
    assert age_payload["metrics"]["binomial_one_tail_alternative"] == "greater"
    assert "bucket_weight" in age_payload["bins"][0]
    assert "variable_avg_event_rate" in age_payload["bins"][0]
    assert "variable_avg_value" in age_payload["bins"][0]
    assert "binomial_pass" in age_payload["bins"][0]
    assert "binomial_one_tail_result" in age_payload["bins"][0]
    assert "bucket_hhi" in age_payload["bins"][0]
    assert "binomial_p_value" in age_payload["bins"][0]
    assert "hhi_contribution" in age_payload["bins"][0]


def test_project_export_filters_by_export_decision_not_manual_status():
    df = woe_df()
    age_spec = build_initial_spec(df, "target", "age", 1, WoeBuildConfig(engine="fallback"))
    segment_spec = build_initial_spec(df, "target", "segment", 1, WoeBuildConfig(engine="fallback"))
    project = {
        "data_key": "unit-test",
        "variables": {
            "age": {
                "name": "age",
                "status": "edited",
                "export_decision": "include",
                "original_spec": age_spec,
                "current_spec": age_spec,
            },
            "segment": {
                "name": "segment",
                "status": "edited",
                "export_decision": "exclude",
                "original_spec": segment_spec,
                "current_spec": segment_spec,
            },
        },
    }

    included = build_project_export(project, df, None, "target", 1, included_only=True)
    excluded = build_project_export(project, df, None, "target", 1, excluded_only=True)
    all_variables = build_project_export(project, df, None, "target", 1)

    assert [variable["name"] for variable in included["variables"]] == ["age"]
    assert [variable["name"] for variable in excluded["variables"]] == ["segment"]
    assert {variable["name"] for variable in all_variables["variables"]} == {"age", "segment"}
    assert included["variables"][0]["mapping_state"] == "auto"
    assert included["variables"][0]["export_decision"] == "include"


def test_excel_export_respects_selected_export_scope():
    df = woe_df()
    age_spec = build_initial_spec(df, "target", "age", 1, WoeBuildConfig(engine="fallback"))
    segment_spec = build_initial_spec(df, "target", "segment", 1, WoeBuildConfig(engine="fallback"))
    project = {
        "data_key": "unit-test",
        "variables": {
            "age": {
                "name": "age",
                "status": "edited",
                "export_decision": "include",
                "original_spec": age_spec,
                "current_spec": age_spec,
                "edits": [{"action": "keep_age"}],
            },
            "segment": {
                "name": "segment",
                "status": "edited",
                "export_decision": "exclude",
                "original_spec": segment_spec,
                "current_spec": segment_spec,
                "edits": [{"action": "exclude_segment"}],
            },
        },
    }

    workbook = project_excel_bytes(project, df, None, "target", 1, included_only=True)
    metrics = pd.read_excel(io.BytesIO(workbook), sheet_name="Variable Metrics")
    details = pd.read_excel(io.BytesIO(workbook), sheet_name="Bin Details")
    edits = pd.read_excel(io.BytesIO(workbook), sheet_name="Manual Edits")

    assert metrics["variable"].tolist() == ["age"]
    assert set(details["variable"]) == {"age"}
    assert edits["variable"].tolist() == ["age"]


def test_pickle_export_contains_scoped_dc_corp_pipe_cache_payload():
    df = woe_df()
    age_spec = build_initial_spec(df, "target", "age", 1, WoeBuildConfig(engine="fallback"))
    segment_spec = build_initial_spec(df, "target", "segment", 1, WoeBuildConfig(engine="fallback"))
    project = {
        "data_key": "unit-test",
        "variables": {
            "age": {
                "name": "age",
                "status": "edited",
                "export_decision": "include",
                "original_spec": age_spec,
                "current_spec": age_spec,
            },
            "segment": {
                "name": "segment",
                "status": "edited",
                "export_decision": "exclude",
                "original_spec": segment_spec,
                "current_spec": segment_spec,
            },
        },
    }
    payload = build_project_export(project, df, None, "target", 1, included_only=True)

    cache = pickle.loads(project_pickle_bytes(payload))
    dc_payload = dc_corp_woe_artifacts(payload)

    assert cache["format"] == "interactive_woe_cache"
    assert cache["variables"] == ["age"]
    assert cache["woe_mapping"] == dc_payload["woe_mapping"]
    assert set(cache["dc_corp_pipe"]["woe_mapping"]["variables"]) == {"age"}
    assert set(cache["dc_corp_pipe"]["woe_values"]) == {"age"}
    age_entry = cache["dc_corp_pipe"]["woe_mapping"]["variables"]["age"]
    assert age_entry["type"] == "numeric"
    assert age_entry["bins"]
    assert all(isinstance(key, int) for key in cache["woe_values"]["age"]["woe_map"])


def test_variable_editor_order_sorts_by_current_gini_descending():
    df = pd.DataFrame(
        {
            "strong": [1, 1, 1, 1, 9, 9, 9, 9],
            "weak": [1, 9, 1, 9, 1, 9, 1, 9],
            "target": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )
    strong_spec = build_initial_spec(df, "target", "strong", 1, WoeBuildConfig(engine="fallback"))
    weak_spec = build_initial_spec(df, "target", "weak", 1, WoeBuildConfig(engine="fallback"))
    project = {
        "data_key": "unit-test",
        "variables": {
            "weak": {"name": "weak", "original_spec": weak_spec, "current_spec": weak_spec},
            "strong": {"name": "strong", "original_spec": strong_spec, "current_spec": strong_spec},
        },
    }

    assert variable_editor_order(project, df, None, "target", 1) == ["strong", "weak"]


def test_parse_special_values_accepts_commas_and_newlines():
    assert parse_special_values("-999, -1\nUNKNOWN") == ["-999", "-1", "UNKNOWN"]
