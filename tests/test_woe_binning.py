from __future__ import annotations

import json

import pandas as pd

from interactive_decision_tree.woe_binning import (
    WoeBuildConfig,
    apply_bin_table_edits,
    apply_numeric_cutpoints,
    build_initial_spec,
    evaluate_spec,
    merge_selected_bins,
    numeric_bin_label,
    parse_special_values,
    set_assigned_woe_from_table,
)
from interactive_decision_tree.woe_export import build_project_export, project_json_bytes


def woe_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [22, 24, 27, 33, 38, 45, 51, 59, None, -999],
            "segment": ["A", "A", "B", "B", "C", "C", "D", "D", "", "SPECIAL"],
            "target": [1, 1, 0, 1, 0, 0, 0, 0, 1, 1],
        }
    )


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


def test_apply_numeric_cutpoints_rebuilds_normal_bins():
    df = woe_df()
    spec = build_initial_spec(df, "target", "age", 1, WoeBuildConfig(engine="fallback"))
    updated = apply_numeric_cutpoints(spec, [30, 50])
    normal_labels = [bin_spec["label"] for bin_spec in updated["bins"] if bin_spec["kind"] == "normal"]

    assert normal_labels == ["(-inf, 30]", "(30, 50]", "(50, inf)"]


def test_numeric_bin_label_avoids_scientific_notation():
    assert numeric_bin_label(33280.0, 43650.0) == "(33280, 43650]"


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


def test_merge_selected_bins_categorical_can_merge_non_adjacent_groups():
    df = woe_df()
    spec = build_initial_spec(df, "target", "segment", 1, WoeBuildConfig(max_bins=4, engine="fallback"))

    updated = merge_selected_bins(spec, ["b001", "b003"])
    normal_values = [bin_spec["values"] for bin_spec in updated["bins"] if bin_spec["kind"] == "normal"]

    assert normal_values[0]
    assert len(normal_values) == 3


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


def test_parse_special_values_accepts_commas_and_newlines():
    assert parse_special_values("-999, -1\nUNKNOWN") == ["-999", "-1", "UNKNOWN"]
