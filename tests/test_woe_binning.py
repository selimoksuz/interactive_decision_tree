from __future__ import annotations

import json

import pandas as pd

from interactive_decision_tree.woe_binning import (
    WoeBuildConfig,
    apply_numeric_cutpoints,
    build_initial_spec,
    evaluate_spec,
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
