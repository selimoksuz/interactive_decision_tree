from __future__ import annotations

import io
import json
import pickle
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .woe_binning import evaluate_spec, json_safe_value


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [safe_json(item) for item in value]
    return json_safe_value(value)


def state_mapping_state(state: dict[str, Any]) -> str:
    current_spec = state.get("current_spec")
    original_spec = state.get("original_spec")
    if isinstance(current_spec, dict) and isinstance(original_spec, dict) and current_spec == original_spec:
        return "auto"
    return "edited"


def state_export_decision(state: dict[str, Any]) -> str:
    decision = str(state.get("export_decision", "")).lower()
    if decision in {"include", "exclude"}:
        return decision
    if str(state.get("status", "")).lower() == "rejected":
        return "exclude"
    return "include"


def scoped_variable_states(
    project: dict[str, Any],
    *,
    approved_only: bool = False,
    include_statuses: set[str] | None = None,
    exclude_statuses: set[str] | None = None,
    included_only: bool = False,
    excluded_only: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    states: list[tuple[str, dict[str, Any]]] = []
    for variable, state in sorted(project.get("variables", {}).items()):
        status = str(state.get("status", "auto"))
        included = state_export_decision(state) == "include"
        if approved_only and status != "approved":
            continue
        if include_statuses is not None and status not in include_statuses:
            continue
        if exclude_statuses is not None and status in exclude_statuses:
            continue
        if included_only and not included:
            continue
        if excluded_only and included:
            continue
        states.append((variable, state))
    return states


def variable_state_rows(
    project: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
    *,
    approved_only: bool = False,
    include_statuses: set[str] | None = None,
    exclude_statuses: set[str] | None = None,
    included_only: bool = False,
    excluded_only: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable, state in scoped_variable_states(
        project,
        approved_only=approved_only,
        include_statuses=include_statuses,
        exclude_statuses=exclude_statuses,
        included_only=included_only,
        excluded_only=excluded_only,
    ):
        current_spec = state.get("current_spec")
        original_spec = state.get("original_spec")
        if not isinstance(current_spec, dict) or not isinstance(original_spec, dict):
            continue
        current_train = evaluate_spec(train_df, target, current_spec, positive_class, "Train")["metrics"]
        original_train = evaluate_spec(train_df, target, original_spec, positive_class, "Train")["metrics"]
        row = {
            "variable": variable,
            "type": current_spec.get("feature_kind"),
            "mapping_state": state_mapping_state(state),
            "export_decision": state_export_decision(state),
            "engine": current_train.get("engine_used"),
            "original_iv": original_train.get("export_iv"),
            "current_iv": current_train.get("export_iv"),
            "iv_delta": _delta(current_train.get("export_iv"), original_train.get("export_iv")),
            "original_gini": original_train.get("export_gini"),
            "current_gini": current_train.get("export_gini"),
            "gini_delta": _delta(current_train.get("export_gini"), original_train.get("export_gini")),
            "bins": current_train.get("bin_count"),
            "manual_woe_bins": current_train.get("manual_woe_bins"),
            "monotonic": current_train.get("is_monotonic"),
            "monotonic_direction": current_train.get("monotonic_direction"),
            "monotonic_violations": current_train.get("monotonic_violation_count"),
            "hhi_total": current_train.get("hhi_total"),
            "average_bucket_hhi": current_train.get("average_bucket_hhi"),
            "variable_avg_value": current_train.get("variable_avg_value"),
            "normalized_hhi": current_train.get("normalized_hhi"),
            "hhi_concentration": current_train.get("hhi_concentration"),
            "max_bin_concentration": current_train.get("max_bin_concentration"),
            "max_bucket_weight": current_train.get("max_bucket_weight"),
            "variable_avg_event_rate": current_train.get("variable_avg_event_rate"),
            "binomial_confidence_level": current_train.get("binomial_confidence_level"),
            "binomial_alternative": current_train.get("binomial_alternative"),
            "binomial_test_method": current_train.get("binomial_test_method"),
            "binomial_one_tail_alternative": current_train.get("binomial_one_tail_alternative"),
            "binomial_one_tail_test_method": current_train.get("binomial_one_tail_test_method"),
            "binomial_multiple_testing": current_train.get("binomial_multiple_testing"),
            "binomial_pass_bins": current_train.get("binomial_pass_bins"),
            "binomial_reject_bins": current_train.get("binomial_reject_bins"),
            "binomial_one_tail_pass_bins": current_train.get("binomial_one_tail_pass_bins"),
            "binomial_one_tail_reject_bins": current_train.get("binomial_one_tail_reject_bins"),
            "binomial_significant_bins": current_train.get("binomial_significant_bins"),
            "binomial_signal_rate": current_train.get("binomial_signal_rate"),
            "binomial_signal_share": current_train.get("binomial_signal_share"),
        }
        if test_df is not None:
            current_test = evaluate_spec(test_df, target, current_spec, positive_class, "Test")["metrics"]
            row["test_iv"] = current_test.get("export_iv")
            row["test_gini"] = current_test.get("export_gini")
            row["test_hhi_total"] = current_test.get("hhi_total")
            row["test_binomial_signal_share"] = current_test.get("binomial_signal_share")
        rows.append(row)
    return pd.DataFrame(rows)


def _delta(current: Any, original: Any) -> float | None:
    try:
        if current is None or original is None:
            return None
        return float(current) - float(original)
    except (TypeError, ValueError):
        return None


def bin_detail_rows(
    project: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
    *,
    approved_only: bool = False,
    include_statuses: set[str] | None = None,
    exclude_statuses: set[str] | None = None,
    included_only: bool = False,
    excluded_only: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, state in scoped_variable_states(
        project,
        approved_only=approved_only,
        include_statuses=include_statuses,
        exclude_statuses=exclude_statuses,
        included_only=included_only,
        excluded_only=excluded_only,
    ):
        current_spec = state.get("current_spec")
        if not isinstance(current_spec, dict):
            continue
        train_table = evaluate_spec(train_df, target, current_spec, positive_class, "Train")["table"]
        frames.append(train_table)
        if test_df is not None:
            frames.append(evaluate_spec(test_df, target, current_spec, positive_class, "Test")["table"])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def export_variable_payload(
    state: dict[str, Any],
    train_df: pd.DataFrame,
    target: str,
    positive_class: Any,
) -> dict[str, Any] | None:
    current_spec = state.get("current_spec")
    original_spec = state.get("original_spec")
    if not isinstance(current_spec, dict) or not isinstance(original_spec, dict):
        return None
    evaluation = evaluate_spec(train_df, target, current_spec, positive_class, "Train")
    metrics = evaluation["metrics"]
    table = evaluation["table"]
    bins: list[dict[str, Any]] = []
    spec_by_id = {str(bin_spec.get("bin_id")): bin_spec for bin_spec in current_spec.get("bins", [])}
    for row in table.to_dict("records"):
        bin_spec = spec_by_id.get(str(row.get("bin_id")), {})
        bins.append(
            {
                "bin_id": row.get("bin_id"),
                "kind": row.get("kind"),
                "label": row.get("label"),
                "lower": safe_json(bin_spec.get("lower")),
                "upper": safe_json(bin_spec.get("upper")),
                "values": safe_json(bin_spec.get("values", [])),
                "calculated_woe": safe_json(row.get("calculated_woe")),
                "assigned_woe": safe_json(row.get("assigned_woe")),
                "export_woe": safe_json(row.get("export_woe")),
                "event_count": safe_json(row.get("event_count")),
                "non_event_count": safe_json(row.get("non_event_count")),
                "count": safe_json(row.get("count")),
                "event_rate": safe_json(row.get("event_rate")),
                "variable_avg_value": safe_json(row.get("variable_avg_value")),
                "variable_avg_event_rate": safe_json(row.get("variable_avg_event_rate")),
                "expected_event_rate": safe_json(row.get("expected_event_rate")),
                "expected_event_count": safe_json(row.get("expected_event_count")),
                "event_count_delta": safe_json(row.get("event_count_delta")),
                "binomial_p_value": safe_json(row.get("binomial_p_value")),
                "binomial_adjusted_p_value": safe_json(row.get("binomial_adjusted_p_value")),
                "binomial_significant": safe_json(row.get("binomial_significant")),
                "binomial_pass": safe_json(row.get("binomial_pass")),
                "binomial_result": safe_json(row.get("binomial_result")),
                "binomial_two_tail_pass": safe_json(row.get("binomial_two_tail_pass")),
                "binomial_two_tail_result": safe_json(row.get("binomial_two_tail_result")),
                "binomial_one_tail_p_value": safe_json(row.get("binomial_one_tail_p_value")),
                "binomial_one_tail_adjusted_p_value": safe_json(
                    row.get("binomial_one_tail_adjusted_p_value")
                ),
                "binomial_one_tail_pass": safe_json(row.get("binomial_one_tail_pass")),
                "binomial_one_tail_result": safe_json(row.get("binomial_one_tail_result")),
                "binomial_ci_lower": safe_json(row.get("binomial_ci_lower")),
                "binomial_ci_upper": safe_json(row.get("binomial_ci_upper")),
                "bucket_weight": safe_json(row.get("bucket_weight")),
                "all_concentration": safe_json(row.get("all_concentration")),
                "event_concentration": safe_json(row.get("event_concentration")),
                "non_event_concentration": safe_json(row.get("non_event_concentration")),
                "hhi_contribution": safe_json(row.get("hhi_contribution")),
                "bucket_hhi": safe_json(row.get("bucket_hhi")),
                "calculated_iv": safe_json(row.get("calculated_iv")),
                "export_iv": safe_json(row.get("export_iv")),
                "protected": safe_json(row.get("protected")),
                "note": safe_json(row.get("note")),
            }
        )
    return {
        "name": state.get("name") or current_spec.get("feature"),
        "type": current_spec.get("feature_kind"),
        "mapping_state": state_mapping_state(state),
        "export_decision": state_export_decision(state),
        "status": state.get("status", "auto"),
        "metrics": safe_json(metrics),
        "config": safe_json(current_spec.get("config", {})),
        "bins": bins,
    }


def build_project_export(
    project: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
    approved_only: bool = False,
    include_statuses: set[str] | None = None,
    exclude_statuses: set[str] | None = None,
    included_only: bool = False,
    excluded_only: bool = False,
) -> dict[str, Any]:
    variables: list[dict[str, Any]] = []
    for _, state in scoped_variable_states(
        project,
        approved_only=approved_only,
        include_statuses=include_statuses,
        exclude_statuses=exclude_statuses,
        included_only=included_only,
        excluded_only=excluded_only,
    ):
        payload = export_variable_payload(state, train_df, target, positive_class)
        if payload is not None:
            variables.append(payload)
    summary = variable_state_rows(
        project,
        train_df,
        test_df,
        target,
        positive_class,
        approved_only=approved_only,
        include_statuses=include_statuses,
        exclude_statuses=exclude_statuses,
        included_only=included_only,
        excluded_only=excluded_only,
    )
    return {
        "format": "interactive_woe_mapping",
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": str(target),
        "positive_class": safe_json(positive_class),
        "data_key": project.get("data_key"),
        "variable_count": len(variables),
        "variables": variables,
        "summary": safe_json(summary.to_dict("records")),
    }


def project_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(safe_json(payload), indent=2, ensure_ascii=False).encode("utf-8")


def _normal_dc_bins(variable: dict[str, Any]) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for bin_payload in variable.get("bins", []):
        if bin_payload.get("kind") != "normal":
            continue
        left = bin_payload.get("lower")
        right = bin_payload.get("upper")
        if left is None and right is None:
            left = float("-inf")
            right = float("inf")
        bins.append({"left": left, "right": right, "woe": float(bin_payload.get("export_woe", 0.0) or 0.0)})
    return bins


def _numeric_dc_entry(variable: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    entry: dict[str, Any] = {"type": "numeric", "bins": _normal_dc_bins(variable)}
    values_entry: dict[str, Any] = {
        "type": "numeric",
        "bins": _numeric_edges(variable),
        "woe_map": {index: row["woe"] for index, row in enumerate(entry["bins"])},
        "iv": variable.get("metrics", {}).get("export_iv"),
        "stats": _dc_stats(variable),
    }
    special_woe: dict[Any, float] = {}
    special_values: list[Any] = []
    for bin_payload in variable.get("bins", []):
        woe_value = float(bin_payload.get("export_woe", 0.0) or 0.0)
        if bin_payload.get("kind") == "missing":
            entry["missing_woe"] = woe_value
            values_entry["missing_woe"] = woe_value
            values_entry["missing_label"] = "MISSING"
        elif bin_payload.get("kind") == "special":
            for value in bin_payload.get("values", []):
                special_values.append(value)
                special_woe[value] = woe_value
    if special_woe:
        entry["special_woe"] = special_woe
        entry["special_values"] = special_values
        values_entry["special_woe"] = special_woe
        values_entry["special_values"] = special_values
    return entry, values_entry


def _numeric_edges(variable: dict[str, Any]) -> list[float]:
    normal_bins = [item for item in variable.get("bins", []) if item.get("kind") == "normal"]
    if not normal_bins:
        return [float("-inf"), float("inf")]
    edges: list[float] = [float("-inf")]
    for bin_payload in normal_bins[:-1]:
        upper = bin_payload.get("upper")
        if upper is not None:
            edges.append(float(upper))
    edges.append(float("inf"))
    return edges


def _dc_stats(variable: dict[str, Any]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for bin_payload in variable.get("bins", []):
        stats.append(
            {
                "label": bin_payload.get("label"),
                "bin_left": bin_payload.get("lower"),
                "bin_right": bin_payload.get("upper"),
                "woe": bin_payload.get("export_woe"),
                "iv_contrib": bin_payload.get("export_iv"),
                "event_count": bin_payload.get("event_count"),
                "nonevent_count": bin_payload.get("non_event_count"),
                "total_count": bin_payload.get("count"),
                "event_rate": bin_payload.get("event_rate"),
                "variable_avg_value": bin_payload.get("variable_avg_value"),
                "variable_avg_event_rate": bin_payload.get("variable_avg_event_rate"),
                "expected_event_rate": bin_payload.get("expected_event_rate"),
                "expected_event_count": bin_payload.get("expected_event_count"),
                "event_count_delta": bin_payload.get("event_count_delta"),
                "binomial_p_value": bin_payload.get("binomial_p_value"),
                "binomial_adjusted_p_value": bin_payload.get("binomial_adjusted_p_value"),
                "binomial_significant": bin_payload.get("binomial_significant"),
                "binomial_pass": bin_payload.get("binomial_pass"),
                "binomial_result": bin_payload.get("binomial_result"),
                "binomial_one_tail_p_value": bin_payload.get("binomial_one_tail_p_value"),
                "binomial_one_tail_adjusted_p_value": bin_payload.get(
                    "binomial_one_tail_adjusted_p_value"
                ),
                "binomial_one_tail_pass": bin_payload.get("binomial_one_tail_pass"),
                "binomial_one_tail_result": bin_payload.get("binomial_one_tail_result"),
                "binomial_ci_lower": bin_payload.get("binomial_ci_lower"),
                "binomial_ci_upper": bin_payload.get("binomial_ci_upper"),
                "bucket_weight": bin_payload.get("bucket_weight"),
                "hhi_contribution": bin_payload.get("hhi_contribution"),
                "bucket_hhi": bin_payload.get("bucket_hhi"),
                "members": bin_payload.get("values", []),
            }
        )
    return stats


def _categorical_dc_entry(variable: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    woe_map: dict[str, float] = {}
    entry: dict[str, Any] = {"type": "categorical", "groups": groups}
    for bin_payload in variable.get("bins", []):
        woe_value = float(bin_payload.get("export_woe", 0.0) or 0.0)
        kind = bin_payload.get("kind")
        if kind == "missing":
            entry["missing_woe"] = woe_value
            entry["missing_label"] = "__MISSING__"
            woe_map["__MISSING__"] = woe_value
            woe_map[""] = woe_value
            continue
        if kind not in {"normal", "special"}:
            continue
        members = [str(value) for value in bin_payload.get("values", [])]
        groups.append(
            {
                "label": str(bin_payload.get("label", "")),
                "members": members,
                "woe": woe_value,
                "event_count": bin_payload.get("event_count"),
                "nonevent_count": bin_payload.get("non_event_count"),
                "total_count": bin_payload.get("count"),
                "event_rate": bin_payload.get("event_rate"),
                "variable_avg_value": bin_payload.get("variable_avg_value"),
                "variable_avg_event_rate": bin_payload.get("variable_avg_event_rate"),
                "expected_event_rate": bin_payload.get("expected_event_rate"),
                "expected_event_count": bin_payload.get("expected_event_count"),
                "event_count_delta": bin_payload.get("event_count_delta"),
                "binomial_p_value": bin_payload.get("binomial_p_value"),
                "binomial_adjusted_p_value": bin_payload.get("binomial_adjusted_p_value"),
                "binomial_significant": bin_payload.get("binomial_significant"),
                "binomial_pass": bin_payload.get("binomial_pass"),
                "binomial_result": bin_payload.get("binomial_result"),
                "binomial_one_tail_p_value": bin_payload.get("binomial_one_tail_p_value"),
                "binomial_one_tail_adjusted_p_value": bin_payload.get(
                    "binomial_one_tail_adjusted_p_value"
                ),
                "binomial_one_tail_pass": bin_payload.get("binomial_one_tail_pass"),
                "binomial_one_tail_result": bin_payload.get("binomial_one_tail_result"),
                "binomial_ci_lower": bin_payload.get("binomial_ci_lower"),
                "binomial_ci_upper": bin_payload.get("binomial_ci_upper"),
                "bucket_weight": bin_payload.get("bucket_weight"),
                "hhi_contribution": bin_payload.get("hhi_contribution"),
                "bucket_hhi": bin_payload.get("bucket_hhi"),
                "iv_contrib": bin_payload.get("export_iv"),
            }
        )
        for value in members:
            woe_map[str(value)] = woe_value
    entry["woe_map"] = woe_map
    entry["default_woe"] = 0.0
    values_entry = {
        **entry,
        "iv": variable.get("metrics", {}).get("export_iv"),
        "stats": groups,
        "categories": list(woe_map),
    }
    return entry, values_entry


def dc_corp_woe_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    mapping = {"variables": {}}
    woe_values: dict[str, Any] = {}
    for variable in payload.get("variables", []):
        name = str(variable.get("name"))
        if variable.get("type") == "numeric":
            entry, values_entry = _numeric_dc_entry(variable)
        else:
            entry, values_entry = _categorical_dc_entry(variable)
        mapping["variables"][name] = entry
        woe_values[name] = values_entry
    return {"woe_mapping": mapping, "woe_values": woe_values}


def project_pickle_bytes(payload: dict[str, Any]) -> bytes:
    dc_artifacts = dc_corp_woe_artifacts(payload)
    cache_payload = {
        "format": "interactive_woe_cache",
        "schema_version": 1,
        "created_at": payload.get("created_at"),
        "target": payload.get("target"),
        "positive_class": payload.get("positive_class"),
        "data_key": payload.get("data_key"),
        "variable_count": payload.get("variable_count"),
        "variables": [variable.get("name") for variable in payload.get("variables", [])],
        "summary": payload.get("summary", []),
        "interactive_woe_mapping": payload,
        "dc_corp_pipe": dc_artifacts,
        "woe_mapping": dc_artifacts["woe_mapping"],
        "woe_values": dc_artifacts["woe_values"],
    }
    return pickle.dumps(cache_payload, protocol=pickle.HIGHEST_PROTOCOL)


def project_excel_bytes(
    project: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
    *,
    approved_only: bool = False,
    include_statuses: set[str] | None = None,
    exclude_statuses: set[str] | None = None,
    included_only: bool = False,
    excluded_only: bool = False,
) -> bytes:
    output = io.BytesIO()
    summary = variable_state_rows(
        project,
        train_df,
        test_df,
        target,
        positive_class,
        approved_only=approved_only,
        include_statuses=include_statuses,
        exclude_statuses=exclude_statuses,
        included_only=included_only,
        excluded_only=excluded_only,
    )
    bins = bin_detail_rows(
        project,
        train_df,
        test_df,
        target,
        positive_class,
        approved_only=approved_only,
        include_statuses=include_statuses,
        exclude_statuses=exclude_statuses,
        included_only=included_only,
        excluded_only=excluded_only,
    )
    export_payload = build_project_export(
        project,
        train_df,
        test_df,
        target,
        positive_class,
        approved_only=approved_only,
        include_statuses=include_statuses,
        exclude_statuses=exclude_statuses,
        included_only=included_only,
        excluded_only=excluded_only,
    )
    contract = pd.DataFrame(
        [
            {"field": "format", "value": export_payload["format"]},
            {"field": "schema_version", "value": export_payload["schema_version"]},
            {"field": "target", "value": export_payload["target"]},
            {"field": "positive_class", "value": str(export_payload["positive_class"])},
            {"field": "variable_count", "value": export_payload["variable_count"]},
            {"field": "data_key", "value": str(export_payload.get("data_key"))},
        ]
    )
    edits = []
    for variable, state in scoped_variable_states(
        project,
        approved_only=approved_only,
        include_statuses=include_statuses,
        exclude_statuses=exclude_statuses,
        included_only=included_only,
        excluded_only=excluded_only,
    ):
        for edit in state.get("edits", []):
            edits.append({"variable": variable, **safe_json(edit)})
    edits_frame = pd.DataFrame(edits)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        contract.to_excel(writer, sheet_name="Export Contract", index=False)
        summary.to_excel(writer, sheet_name="Variable Metrics", index=False)
        bins.to_excel(writer, sheet_name="Bin Details", index=False)
        edits_frame.to_excel(writer, sheet_name="Manual Edits", index=False)
    return output.getvalue()


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def variable_sql_case(variable: dict[str, Any]) -> str:
    name = str(variable["name"])
    lines = [f"CASE"]
    for bin_payload in variable.get("bins", []):
        kind = bin_payload.get("kind")
        woe = bin_payload.get("export_woe")
        if kind == "missing":
            lines.append(f"  WHEN {name} IS NULL THEN {woe}")
        elif kind == "special":
            values = ", ".join(sql_literal(value) for value in bin_payload.get("values", []))
            if values:
                lines.append(f"  WHEN {name} IN ({values}) THEN {woe}")
        elif variable.get("type") == "numeric":
            lower = bin_payload.get("lower")
            upper = bin_payload.get("upper")
            if lower is None and upper is None:
                lines.append(f"  WHEN {name} IS NOT NULL THEN {woe}")
            elif lower is None:
                lines.append(f"  WHEN {name} <= {upper} THEN {woe}")
            elif upper is None:
                lines.append(f"  WHEN {name} > {lower} THEN {woe}")
            else:
                lines.append(f"  WHEN {name} > {lower} AND {name} <= {upper} THEN {woe}")
        else:
            values = ", ".join(sql_literal(value) for value in bin_payload.get("values", []))
            if values:
                lines.append(f"  WHEN {name} IN ({values}) THEN {woe}")
    lines.append("  ELSE 0.0")
    lines.append(f"END AS {name}_WOE")
    return "\n".join(lines)


def project_sql_text(payload: dict[str, Any]) -> str:
    cases = [variable_sql_case(variable) for variable in payload.get("variables", [])]
    return ",\n\n".join(cases)
