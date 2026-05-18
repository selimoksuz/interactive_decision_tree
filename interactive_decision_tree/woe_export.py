from __future__ import annotations

import io
import json
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


def variable_state_rows(
    project: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable, state in sorted(project.get("variables", {}).items()):
        current_spec = state.get("current_spec")
        original_spec = state.get("original_spec")
        if not isinstance(current_spec, dict) or not isinstance(original_spec, dict):
            continue
        current_train = evaluate_spec(train_df, target, current_spec, positive_class, "Train")["metrics"]
        original_train = evaluate_spec(train_df, target, original_spec, positive_class, "Train")["metrics"]
        row = {
            "variable": variable,
            "type": current_spec.get("feature_kind"),
            "status": state.get("status", "auto"),
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
        }
        if test_df is not None:
            current_test = evaluate_spec(test_df, target, current_spec, positive_class, "Test")["metrics"]
            row["test_iv"] = current_test.get("export_iv")
            row["test_gini"] = current_test.get("export_gini")
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
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for state in sorted(project.get("variables", {}).values(), key=lambda item: item.get("name", "")):
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
                "all_concentration": safe_json(row.get("all_concentration")),
                "event_concentration": safe_json(row.get("event_concentration")),
                "non_event_concentration": safe_json(row.get("non_event_concentration")),
                "calculated_iv": safe_json(row.get("calculated_iv")),
                "export_iv": safe_json(row.get("export_iv")),
                "protected": safe_json(row.get("protected")),
                "note": safe_json(row.get("note")),
            }
        )
    return {
        "name": state.get("name") or current_spec.get("feature"),
        "type": current_spec.get("feature_kind"),
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
) -> dict[str, Any]:
    variables: list[dict[str, Any]] = []
    for state in sorted(project.get("variables", {}).values(), key=lambda item: item.get("name", "")):
        if approved_only and state.get("status") != "approved":
            continue
        payload = export_variable_payload(state, train_df, target, positive_class)
        if payload is not None:
            variables.append(payload)
    summary = variable_state_rows(project, train_df, test_df, target, positive_class)
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


def project_excel_bytes(
    project: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
) -> bytes:
    output = io.BytesIO()
    summary = variable_state_rows(project, train_df, test_df, target, positive_class)
    bins = bin_detail_rows(project, train_df, test_df, target, positive_class)
    export_payload = build_project_export(project, train_df, test_df, target, positive_class)
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
    for variable, state in sorted(project.get("variables", {}).items()):
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


def project_python_transformer_text(payload: dict[str, Any]) -> str:
    mapping_json = json.dumps(safe_json(payload), indent=2, ensure_ascii=False)
    return f'''from __future__ import annotations

import pandas as pd


WOE_MAPPING = {mapping_json}


def _is_missing(value):
    return pd.isna(value) or (isinstance(value, str) and value.strip() == "")


def _map_numeric(value, bins):
    if _is_missing(value):
        for bin_payload in bins:
            if bin_payload.get("kind") == "missing":
                return bin_payload.get("export_woe", 0.0)
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    for bin_payload in bins:
        if bin_payload.get("kind") == "special":
            try:
                if numeric in [float(item) for item in bin_payload.get("values", [])]:
                    return bin_payload.get("export_woe", 0.0)
            except (TypeError, ValueError):
                pass
        if bin_payload.get("kind") != "normal":
            continue
        lower = bin_payload.get("lower")
        upper = bin_payload.get("upper")
        if lower is not None and numeric <= float(lower):
            continue
        if upper is not None and numeric > float(upper):
            continue
        return bin_payload.get("export_woe", 0.0)
    return 0.0


def _map_categorical(value, bins):
    if _is_missing(value):
        for bin_payload in bins:
            if bin_payload.get("kind") == "missing":
                return bin_payload.get("export_woe", 0.0)
        return 0.0
    text = str(value)
    for bin_payload in bins:
        if text in {{str(item) for item in bin_payload.get("values", [])}}:
            return bin_payload.get("export_woe", 0.0)
    return 0.0


def transform_woe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for variable in WOE_MAPPING.get("variables", []):
        name = variable["name"]
        bins = variable.get("bins", [])
        if variable.get("type") == "numeric":
            out[f"{{name}}_WOE"] = out[name].map(lambda value: _map_numeric(value, bins))
        else:
            out[f"{{name}}_WOE"] = out[name].map(lambda value: _map_categorical(value, bins))
    return out
'''
