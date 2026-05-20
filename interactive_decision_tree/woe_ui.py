from __future__ import annotations

import fnmatch
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from .woe_binning import (
    WoeBuildConfig,
    apply_bin_table_edits,
    apply_categorical_groups,
    apply_numeric_cutpoints,
    bin_display_label,
    build_initial_spec,
    categorical_groups_from_spec,
    copy_spec,
    cutpoints_from_spec,
    evaluate_original_current,
    evaluate_spec,
    merge_selected_bins,
    missing_mask,
    normal_bins,
    parse_category_groups,
    parse_cutpoints,
    parse_special_values,
    special_mask,
)
from .woe_export import (
    build_project_export,
    project_excel_bytes,
    project_json_bytes,
    project_pickle_bytes,
    project_sql_text,
    state_export_decision,
    state_mapping_state,
    variable_state_rows,
)


WOE_PROJECTS_KEY = "_interactive_tree_woe_projects"
WOE_ACTIVE_VARIABLE_KEY = "_interactive_tree_woe_active_variable"
WOE_SUMMARY_CACHE_KEY = "_interactive_tree_woe_summary_cache"
WOE_REPORT_CACHE_KEY = "_interactive_tree_woe_report_cache"
WOE_VARIABLE_FILTER_MAX_VISIBLE = 250


def session_cache(key: str) -> dict[Any, Any]:
    cache = st.session_state.setdefault(key, {})
    if not isinstance(cache, dict):
        st.session_state[key] = {}
        cache = st.session_state[key]
    return cache


def bounded_cache_set(cache: dict[Any, Any], key: Any, value: Any, max_items: int = 32) -> None:
    if len(cache) >= max_items and key not in cache:
        cache.pop(next(iter(cache)), None)
    cache[key] = value


def spec_signature(spec: dict[str, Any] | None) -> str:
    if not isinstance(spec, dict):
        return ""
    raw = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def project_signature(project: dict[str, Any]) -> str:
    payload = []
    for variable, state in sorted(project.get("variables", {}).items()):
        payload.append(
            {
                "variable": str(variable),
                "status": str(state.get("status", "")),
                "export_decision": state_export_decision(state),
                "updated_at": str(state.get("updated_at", "")),
                "original": spec_signature(state.get("original_spec")),
                "current": spec_signature(state.get("current_spec")),
            }
        )
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def data_signature(df: pd.DataFrame, test_df: pd.DataFrame | None, target: str, positive_class: Any) -> tuple[Any, ...]:
    return (
        id(df),
        int(len(df)),
        id(test_df) if test_df is not None else None,
        int(len(test_df)) if test_df is not None else 0,
        str(target),
        str(positive_class),
    )


def woe_project_key(data_key: str, target: str, positive_class: Any) -> str:
    raw = f"{data_key}|{target}|{positive_class}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def get_projects() -> dict[str, Any]:
    if WOE_PROJECTS_KEY not in st.session_state:
        st.session_state[WOE_PROJECTS_KEY] = {}
    return st.session_state[WOE_PROJECTS_KEY]


def get_project(data_key: str, target: str, positive_class: Any) -> dict[str, Any]:
    key = woe_project_key(data_key, target, positive_class)
    projects = get_projects()
    if key not in projects:
        projects[key] = {
            "project_key": key,
            "data_key": data_key,
            "target": target,
            "positive_class": str(positive_class),
            "variables": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    return projects[key]


def add_edit(state: dict[str, Any], action: str, detail: dict[str, Any] | None = None) -> None:
    state.setdefault("edits", []).append(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **(detail or {}),
        }
    )
    state["updated_at"] = datetime.now(timezone.utc).isoformat()


MAPPING_STATE_LABELS = {
    "auto": "Auto binning",
    "edited": "Manual revision",
}

EXPORT_DECISION_LABELS = {
    "include": "Included",
    "exclude": "Excluded",
}


def mark_mapping_edited(state: dict[str, Any], action: str, detail: dict[str, Any] | None = None) -> None:
    state["status"] = "edited"
    add_edit(state, action, detail)


def set_export_decision(state: dict[str, Any], decision: str) -> None:
    state["export_decision"] = decision
    add_edit(state, "set_export_decision", {"export_decision": decision})


def metrics_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    keys = [
        "export_iv",
        "export_gini",
        "calculated_iv",
        "calculated_gini",
        "bin_count",
        "manual_woe_bins",
        "is_monotonic",
        "monotonic_direction",
        "monotonic_violation_count",
        "engine_used",
    ]
    return pd.DataFrame([{"metric": key, "value": metrics.get(key)} for key in keys])


def woe_column_config() -> dict[str, Any]:
    return {
        "bin_order": st.column_config.NumberColumn("order", format="%d"),
        "count": st.column_config.NumberColumn("count", format="%d"),
        "event_count": st.column_config.NumberColumn("event", format="%d"),
        "non_event_count": st.column_config.NumberColumn("non_event", format="%d"),
        "event_rate": st.column_config.NumberColumn("event_rate", format="%.6f"),
        "all_concentration": st.column_config.NumberColumn("all_pct", format="%.6f"),
        "event_concentration": st.column_config.NumberColumn("event_pct", format="%.6f"),
        "non_event_concentration": st.column_config.NumberColumn("non_event_pct", format="%.6f"),
        "calculated_woe": st.column_config.NumberColumn("calculated_woe", format="%.6f"),
        "assigned_woe": st.column_config.NumberColumn("assigned_woe", format="%.6f"),
        "lower": st.column_config.NumberColumn("lower", format="%.12g"),
        "upper": st.column_config.NumberColumn("upper", format="%.12g"),
        "export_woe": st.column_config.NumberColumn("export_woe", format="%.6f"),
        "calculated_iv": st.column_config.NumberColumn("calculated_iv", format="%.6f"),
        "export_iv": st.column_config.NumberColumn("export_iv", format="%.6f"),
        "protected": st.column_config.CheckboxColumn("protected"),
    }


def editable_bin_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "bin_id",
        "bin_order",
        "kind",
        "label",
        "lower",
        "upper",
        "values",
        "count",
        "event_count",
        "non_event_count",
        "event_rate",
        "all_concentration",
        "event_concentration",
        "non_event_concentration",
        "calculated_woe",
        "assigned_woe",
        "export_woe",
        "calculated_iv",
        "export_iv",
        "protected",
        "note",
    ]
    return table[[column for column in columns if column in table.columns]].copy()


def cutpoints_text(spec: dict[str, Any]) -> str:
    return ", ".join(f"{value:.12g}" for value in cutpoints_from_spec(spec))


def category_groups_text(spec: dict[str, Any]) -> str:
    return "\n".join(", ".join(group) for group in categorical_groups_from_spec(spec))


def run_initial_binning(
    project: dict[str, Any],
    df: pd.DataFrame,
    target: str,
    variables: list[str],
    positive_class: Any,
    config: WoeBuildConfig,
    replace_existing: bool,
) -> None:
    progress = st.progress(0.0, text="Running WOE binning")
    total = max(1, len(variables))
    for index, variable in enumerate(variables, start=1):
        if not replace_existing and variable in project.get("variables", {}):
            progress.progress(index / total, text=f"Skipped existing variable {variable}")
            continue
        spec = build_initial_spec(df, target, variable, positive_class, config)
        project.setdefault("variables", {})[variable] = {
            "name": variable,
            "status": "auto",
            "export_decision": "include",
            "original_spec": copy_spec(spec),
            "current_spec": copy_spec(spec),
            "edits": [
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "action": "initial_binning",
                    "engine_used": spec.get("config", {}).get("engine_used", "fallback"),
                }
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        progress.progress(index / total, text=f"Binned {variable}")
    progress.empty()


def normalize_variable_selection(variables: list[str], selected: Any) -> list[str]:
    variable_set = set(map(str, variables))
    if selected is None:
        return []
    if isinstance(selected, str):
        selected_iterable = [selected]
    else:
        try:
            selected_iterable = list(selected)
        except TypeError:
            selected_iterable = []
    return [str(variable) for variable in selected_iterable if str(variable) in variable_set]


def filter_variable_options(variables: list[str], query: str | None) -> list[str]:
    text = str(query or "").strip().lower()
    if not text:
        return list(variables)
    tokens = [
        token.strip().lower().replace("%", "*")
        for chunk in text.replace("\n", ",").split(",")
        for token in [chunk]
        if token.strip()
    ]
    if not tokens:
        return list(variables)

    def matches(variable: str) -> bool:
        lowered = str(variable).lower()
        return any(
            fnmatch.fnmatch(lowered, token) if "*" in token or "?" in token else token in lowered
            for token in tokens
        )

    return [variable for variable in variables if matches(variable)]


def ordered_variable_selection(variables: list[str], selected: set[str]) -> list[str]:
    return [variable for variable in variables if variable in selected]


def update_variable_selection_for_filtered(
    variables: list[str],
    current_selection: list[str],
    filtered_variables: list[str],
    include: bool,
) -> list[str]:
    selected = set(normalize_variable_selection(variables, current_selection))
    filtered = set(map(str, filtered_variables))
    if include:
        selected.update(filtered)
    else:
        selected.difference_update(filtered)
    return ordered_variable_selection(variables, selected)


def render_woe_variable_selector(variables: list[str]) -> list[str]:
    default_variables = variables[: min(20, len(variables))]
    selected_key = "woe_selected_variables"
    remembered_variables = normalize_variable_selection(
        variables,
        st.session_state.get(selected_key, default_variables),
    )
    if not remembered_variables:
        remembered_variables = default_variables
    st.session_state[selected_key] = remembered_variables

    st.sidebar.markdown("**WOE variables**")
    selected_variables = normalize_variable_selection(
        variables,
        st.session_state.get(selected_key, remembered_variables),
    )
    with st.sidebar.popover("Dropdown Filter Panel"):
        query = st.text_input(
            "Search variable",
            placeholder="Search",
            key="woe_variable_search",
            label_visibility="collapsed",
            help="Case-insensitive contains search. Use comma/newline for multiple terms; * or % works as wildcard.",
        )
        filtered_variables = filter_variable_options(variables, query)
        filtered_signature = hashlib.sha256(
            json.dumps(filtered_variables, default=str).encode("utf-8")
        ).hexdigest()[:10]
        selection_signature = hashlib.sha256(
            json.dumps(selected_variables, default=str).encode("utf-8")
        ).hexdigest()[:10]
        st.caption(
            f"{len(selected_variables):,} selected | {len(filtered_variables):,} matching | {len(variables):,} total"
        )

        selected_set = set(selected_variables)
        all_filtered_selected = bool(filtered_variables) and all(
            variable in selected_set for variable in filtered_variables
        )
        select_all_value = st.checkbox(
            "(Select All)",
            value=all_filtered_selected,
            key=f"woe_variable_filter_all::{filtered_signature}::{selection_signature}",
            disabled=not filtered_variables,
        )
        if filtered_variables and select_all_value != all_filtered_selected:
            selected_variables = update_variable_selection_for_filtered(
                variables,
                selected_variables,
                filtered_variables,
                select_all_value,
            )
            st.session_state[selected_key] = selected_variables
            selected_set = set(selected_variables)

        list_container = st.container(height=260, border=True)
        visible_variables = filtered_variables[:WOE_VARIABLE_FILTER_MAX_VISIBLE]
        if len(filtered_variables) > len(visible_variables):
            list_container.caption(
                f"Showing first {len(visible_variables):,} matching variable(s). Use search to narrow the list."
            )
        next_selected = set(selected_variables)
        changed = False
        for variable in visible_variables:
            was_selected = variable in selected_set
            is_selected = list_container.checkbox(
                str(variable),
                value=was_selected,
                key=f"woe_variable_filter_item::{filtered_signature}::{selection_signature}::{variable}",
            )
            if is_selected != was_selected:
                changed = True
                if is_selected:
                    next_selected.add(variable)
                else:
                    next_selected.discard(variable)
        if changed:
            selected_variables = ordered_variable_selection(variables, next_selected)
            st.session_state[selected_key] = selected_variables
        if not filtered_variables:
            st.caption("No matching variables.")
        st.caption("Selections are used when you press Run initial WOE binning.")

    selected_variables = normalize_variable_selection(variables, st.session_state.get(selected_key, selected_variables))
    st.sidebar.caption(f"{len(selected_variables):,} selected | {len(variables):,} available")
    return selected_variables


def build_config_from_sidebar(features: list[str]) -> tuple[list[str], WoeBuildConfig, bool]:
    selected_variables = render_woe_variable_selector(features)
    max_bins = st.sidebar.number_input("Max bins", min_value=2, max_value=20, value=6, step=1, key="woe_max_bins")
    min_bin_size = st.sidebar.slider(
        "Min bin size",
        min_value=0.01,
        max_value=0.30,
        value=0.05,
        step=0.01,
        key="woe_min_bin_size",
    )
    monotonic_trend = st.sidebar.selectbox(
        "Monotonic trend",
        ["auto", "none", "ascending", "descending"],
        index=0,
        key="woe_monotonic_trend",
    )
    engine = st.sidebar.selectbox("Binning engine", ["auto", "fallback", "optbinning"], index=0, key="woe_engine")
    missing_separate = st.sidebar.checkbox("Missing as separate bin", value=True, key="woe_missing_separate")
    blank_as_missing = st.sidebar.checkbox("Blank string as missing", value=True, key="woe_blank_as_missing")
    replace_existing = st.sidebar.checkbox(
        "Overwrite existing mappings on rerun",
        value=False,
        key="woe_replace_existing",
        help="When enabled, running initial WOE binning again replaces the stored auto and current mapping for selected variables.",
    )
    config = WoeBuildConfig(
        max_bins=int(max_bins),
        min_bin_size=float(min_bin_size),
        monotonic_trend=str(monotonic_trend),
        missing_separate=bool(missing_separate),
        blank_as_missing=bool(blank_as_missing),
        engine=str(engine),
    )
    return list(selected_variables), config, bool(replace_existing)


def cached_variable_state_rows(
    project: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
) -> pd.DataFrame:
    cache = session_cache(WOE_SUMMARY_CACHE_KEY)
    key = (
        str(project.get("project_key")),
        data_signature(df, test_df, target, positive_class),
        project_signature(project),
    )
    if key not in cache:
        bounded_cache_set(
            cache,
            key,
            variable_state_rows(project, df, test_df, target, positive_class),
            max_items=16,
        )
    return cache[key].copy()


def cached_evaluate_original_current(
    project: dict[str, Any],
    state: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
) -> dict[str, Any]:
    cache = session_cache(WOE_REPORT_CACHE_KEY)
    key = (
        str(project.get("project_key")),
        str(state.get("name")),
        data_signature(df, test_df, target, positive_class),
        str(state.get("status", "")),
        str(state.get("export_decision", "")),
        str(state.get("updated_at", "")),
        spec_signature(state.get("original_spec")),
        spec_signature(state.get("current_spec")),
    )
    if key not in cache:
        bounded_cache_set(
            cache,
            key,
            evaluate_original_current(
                df,
                test_df,
                target,
                state["original_spec"],
                state["current_spec"],
                positive_class,
            ),
            max_items=32,
        )
    return cache[key]


def cached_special_missing_preview(
    state: dict[str, Any],
    df: pd.DataFrame,
    blank_as_missing: bool,
    special_values: list[str],
) -> pd.DataFrame:
    cache = session_cache(WOE_REPORT_CACHE_KEY)
    key = (
        "special_missing_preview",
        id(df),
        int(len(df)),
        str(state.get("name")),
        bool(blank_as_missing),
        tuple(str(value) for value in special_values),
    )
    if key not in cache:
        feature_series = df[state["name"]]
        missing_preview = missing_mask(feature_series, bool(blank_as_missing))
        special_preview = special_mask(feature_series, special_values) & ~missing_preview
        preview = pd.DataFrame(
            [
                {"bucket": "missing", "rows": int(missing_preview.sum()), "share": float(missing_preview.mean())},
                {"bucket": "special", "rows": int(special_preview.sum()), "share": float(special_preview.mean())},
                {
                    "bucket": "normal",
                    "rows": int((~missing_preview & ~special_preview).sum()),
                    "share": float((~missing_preview & ~special_preview).mean()),
                },
            ]
        )
        bounded_cache_set(cache, key, preview, max_items=32)
    return cache[key].copy()


def render_catalog(
    project: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
) -> pd.DataFrame:
    summary = cached_variable_state_rows(project, df, test_df, target, positive_class)
    st.subheader("WOE variable catalog")
    if summary.empty:
        st.info("Run initial WOE binning to create variable mappings.")
        return summary
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "original_iv": st.column_config.NumberColumn(format="%.6f"),
            "current_iv": st.column_config.NumberColumn(format="%.6f"),
            "iv_delta": st.column_config.NumberColumn(format="%.6f"),
            "original_gini": st.column_config.NumberColumn(format="%.6f"),
            "current_gini": st.column_config.NumberColumn(format="%.6f"),
            "gini_delta": st.column_config.NumberColumn(format="%.6f"),
            "test_iv": st.column_config.NumberColumn(format="%.6f"),
            "test_gini": st.column_config.NumberColumn(format="%.6f"),
        },
    )
    return summary


def render_project_exports(
    project: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
) -> None:
    if not project.get("variables"):
        return
    with st.expander("Project exports", expanded=False):
        scope_options = {
            "included": {
                "label": "Included variables",
                "included_only": True,
                "excluded_only": False,
            },
            "excluded": {
                "label": "Excluded variables",
                "included_only": False,
                "excluded_only": True,
            },
            "all": {
                "label": "All variables",
                "included_only": False,
                "excluded_only": False,
            },
        }
        scope_key = st.selectbox(
            "WOE export scope",
            options=list(scope_options),
            index=0,
            format_func=lambda key: scope_options[str(key)]["label"],
            help="Applies to every WOE export download shown below.",
        )
        prepare_exports = st.checkbox(
            "Prepare export downloads",
            value=False,
            key=f"woe_prepare_exports::{project.get('project_key')}",
            help="Large data exports recalculate bin metrics, so prepare them only when you are ready to download.",
        )
        if not prepare_exports:
            st.caption("Downloads are not prepared on every UI rerun. Enable this only at export time.")
            return
        scope = scope_options[str(scope_key)]
        export_payload = build_project_export(
            project,
            df,
            test_df,
            target,
            positive_class,
            included_only=bool(scope["included_only"]),
            excluded_only=bool(scope["excluded_only"]),
        )
        if not export_payload.get("variables"):
            st.warning("Selected export scope contains no variables.")
        export_cols = st.columns(4)
        export_cols[0].download_button(
            "Download WOE JSON",
            data=project_json_bytes(export_payload),
            file_name="interactive_woe_mapping.json",
            mime="application/json",
            width="stretch",
        )
        export_cols[1].download_button(
            "Download WOE Excel",
            data=project_excel_bytes(
                project,
                df,
                test_df,
                target,
                positive_class,
                included_only=bool(scope["included_only"]),
                excluded_only=bool(scope["excluded_only"]),
            ),
            file_name="interactive_woe_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
        export_cols[2].download_button(
            "Download WOE cache PKL",
            data=project_pickle_bytes(export_payload),
            file_name="interactive_woe_cache.pkl",
            mime="application/octet-stream",
            width="stretch",
        )
        export_cols[3].download_button(
            "Download SQL CASE",
            data=project_sql_text(export_payload),
            file_name="woe_transform.sql",
            mime="text/plain",
            width="stretch",
        )


def _sort_metric(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) else None


def variable_editor_order(
    project: dict[str, Any],
    summary: pd.DataFrame | None = None,
    *legacy_args: Any,
) -> list[str]:
    variables = sorted(project.get("variables", {}))
    if not variables:
        return []
    if legacy_args:
        df = summary
        if isinstance(df, pd.DataFrame) and len(legacy_args) >= 3:
            summary = variable_state_rows(project, df, legacy_args[0], legacy_args[1], legacy_args[2])
        else:
            summary = None
    if summary is None or summary.empty or "variable" not in summary.columns:
        return variables
    metric_by_variable = {str(row["variable"]): row for row in summary.to_dict("records")}

    def sort_key(variable: str) -> tuple[int, float, float, str]:
        row = metric_by_variable.get(variable, {})
        gini = _sort_metric(row.get("current_gini"))
        iv = _sort_metric(row.get("current_iv"))
        if gini is not None:
            return (0, -gini, -(iv or 0.0), variable)
        if iv is not None:
            return (1, 0.0, -iv, variable)
        return (2, 0.0, 0.0, variable)

    return sorted(variables, key=sort_key)


def render_special_missing_editor(
    state: dict[str, Any],
    df: pd.DataFrame,
    target: str,
    positive_class: Any,
) -> None:
    current_spec = state["current_spec"]
    config = dict(current_spec.get("config", {}))
    st.markdown("**Special / missing policy**")
    st.caption(
        "SAS-style flow: keep true missing separate first, define business special codes per variable "
        "(-999, UNKNOWN, NO_INFO), then decide whether each bucket gets calculated WOE or assigned WOE."
    )
    special_values_text = st.text_area(
        "Special values",
        value=", ".join(str(value) for value in config.get("special_values", [])),
        key=f"woe_special_values::{state['name']}",
        help="Comma or newline separated. Examples: -999, -1, UNKNOWN, N/A.",
    )
    cols = st.columns(3)
    missing_separate = cols[0].checkbox(
        "Missing separate",
        value=bool(config.get("missing_separate", True)),
        key=f"woe_missing_separate::{state['name']}",
    )
    blank_as_missing = cols[1].checkbox(
        "Blank as missing",
        value=bool(config.get("blank_as_missing", True)),
        key=f"woe_blank_as_missing::{state['name']}",
    )
    protected_special = cols[2].checkbox(
        "Protect special bin",
        value=bool(config.get("protected_special", True)),
        key=f"woe_protected_special::{state['name']}",
    )
    special_values = parse_special_values(special_values_text)
    preview = cached_special_missing_preview(state, df, bool(blank_as_missing), special_values)
    st.dataframe(
        preview,
        hide_index=True,
        width="stretch",
        column_config={"share": st.column_config.NumberColumn(format="%.6f")},
    )
    if st.button("Apply special / missing policy", width="stretch", key=f"woe_apply_special::{state['name']}"):
        new_config = WoeBuildConfig(
            max_bins=int(config.get("max_bins", 6)),
            min_bin_size=float(config.get("min_bin_size", 0.05)),
            monotonic_trend=str(config.get("monotonic_trend", "auto")),
            missing_separate=bool(missing_separate),
            blank_as_missing=bool(blank_as_missing),
            special_values=tuple(special_values),
            protected_special=bool(protected_special),
            engine=str(config.get("engine", "auto")),
        )
        state["current_spec"] = build_initial_spec(df, target, state["name"], positive_class, new_config)
        mark_mapping_edited(
            state,
            "apply_special_missing_policy",
            {
                "special_values": list(new_config.special_values),
                "missing_separate": new_config.missing_separate,
                "blank_as_missing": new_config.blank_as_missing,
            },
        )
        st.rerun()


def render_manual_structure_editor(state: dict[str, Any]) -> None:
    current_spec = state["current_spec"]
    normal = normal_bins(current_spec)
    merge_options = [str(bin_spec.get("bin_id")) for bin_spec in normal]
    if len(merge_options) >= 2:
        selected_bins = st.multiselect(
            "Bins to merge",
            options=merge_options,
            format_func=lambda bin_id: next(
                bin_display_label(current_spec, bin_spec)
                for bin_spec in normal
                if str(bin_spec.get("bin_id")) == str(bin_id)
            ),
            key=f"woe_merge_bins::{state['name']}",
            help=(
                "Numeric variables require adjacent bins. Categorical variables can merge any selected groups."
            ),
        )
        if st.button(
            "Merge selected bins",
            width="stretch",
            key=f"woe_merge_selected::{state['name']}",
            disabled=len(selected_bins) < 2,
        ):
            try:
                state["current_spec"] = merge_selected_bins(current_spec, selected_bins)
            except ValueError as exc:
                st.error(str(exc))
            else:
                mark_mapping_edited(state, "merge_selected_bins", {"bin_ids": selected_bins})
                st.rerun()

    if current_spec.get("feature_kind") == "numeric":
        st.markdown("**Numeric cutpoints**")
        cutpoint_key = f"woe_cutpoints::{state['name']}"
        cutpoint_text = st.text_area(
            "Cutpoints",
            value=st.session_state.get(cutpoint_key, cutpoints_text(current_spec)),
            key=cutpoint_key,
            help="Comma or newline separated numeric cutpoints.",
        )
        col1, _ = st.columns(2)
        if col1.button("Apply cutpoints", width="stretch", key=f"woe_apply_cutpoints::{state['name']}"):
            state["current_spec"] = apply_numeric_cutpoints(current_spec, parse_cutpoints(cutpoint_text))
            mark_mapping_edited(state, "apply_cutpoints", {"cutpoints": parse_cutpoints(cutpoint_text)})
            st.rerun()
    else:
        st.markdown("**Categorical groups**")
        group_key = f"woe_category_groups::{state['name']}"
        groups_text = st.text_area(
            "One bin per line, values comma separated",
            value=st.session_state.get(group_key, category_groups_text(current_spec)),
            key=group_key,
            height=180,
        )
        if st.button("Apply categorical groups", width="stretch", key=f"woe_apply_groups::{state['name']}"):
            state["current_spec"] = apply_categorical_groups(current_spec, parse_category_groups(groups_text))
            mark_mapping_edited(
                state,
                "apply_categorical_groups",
                {"group_count": len(parse_category_groups(groups_text))},
            )
            st.rerun()


def render_variable_editor(
    project: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
    summary: pd.DataFrame | None = None,
) -> None:
    variables = variable_editor_order(project, summary)
    if not variables:
        return

    remembered = st.session_state.get(WOE_ACTIVE_VARIABLE_KEY)
    default_index = variables.index(remembered) if remembered in variables else 0
    variable = st.selectbox("Variable editor", variables, index=default_index, key=WOE_ACTIVE_VARIABLE_KEY)
    state = project["variables"][variable]
    mapping_state = state_mapping_state(state)
    export_decision = state_export_decision(state)
    state_cols = st.columns(4)
    state_cols[0].metric("Mapping state", MAPPING_STATE_LABELS[mapping_state])
    state_cols[1].metric("Export", EXPORT_DECISION_LABELS[export_decision])
    if state_cols[2].button("Reset to auto mapping", width="stretch", key=f"woe_reset_original::{variable}"):
        state["current_spec"] = copy_spec(state["original_spec"])
        state["status"] = "auto"
        add_edit(state, "reset_current_to_original")
        st.rerun()
    export_button = "Exclude from export" if export_decision == "include" else "Include in export"
    next_decision = "exclude" if export_decision == "include" else "include"
    if state_cols[3].button(export_button, width="stretch", key=f"woe_export_decision::{variable}"):
        set_export_decision(state, next_decision)
        st.rerun()

    reports = cached_evaluate_original_current(project, state, df, test_df, target, positive_class)
    current_train = reports["current_train"]
    original_train = reports["original_train"]

    metric_cols = st.columns(5)
    metric_cols[0].metric("Current IV", f"{current_train['metrics']['export_iv']:.6f}")
    current_gini = current_train["metrics"].get("export_gini")
    metric_cols[1].metric("Current Gini", "n/a" if current_gini is None else f"{current_gini:.6f}")
    metric_cols[2].metric("Bins", str(current_train["metrics"].get("bin_count")))
    metric_cols[3].metric("Manual WOE bins", str(current_train["metrics"].get("manual_woe_bins")))
    metric_cols[4].metric("Monotonic", str(current_train["metrics"].get("monotonic_direction")))

    bins_tab, structure_tab, special_tab, compare_tab = st.tabs(
        ["Current bins", "Bin structure", "Special / missing", "Original comparison"]
    )

    with bins_tab:
        chart_frame = current_train["table"][current_train["table"]["kind"] == "normal"][
            ["label", "event_rate", "export_woe"]
        ].set_index("label")
        if not chart_frame.empty:
            st.line_chart(chart_frame)
        editor_key = f"woe_bin_editor::{project['project_key']}::{variable}"
        disabled_columns = [
            "bin_id",
            "bin_order",
            "kind",
            "label",
            "values",
            "count",
            "event_count",
            "non_event_count",
            "event_rate",
            "all_concentration",
            "event_concentration",
            "non_event_concentration",
            "calculated_woe",
            "export_woe",
            "calculated_iv",
            "export_iv",
            "protected",
        ]
        if state["current_spec"].get("feature_kind") != "numeric":
            disabled_columns.extend(["lower", "upper"])
        edited = st.data_editor(
            editable_bin_table(current_train["table"]),
            hide_index=True,
            width="stretch",
            key=editor_key,
            disabled=disabled_columns,
            column_config=woe_column_config(),
        )
        if st.button("Apply table edits", width="stretch", key=f"woe_apply_table_edits::{variable}"):
            try:
                state["current_spec"] = apply_bin_table_edits(state["current_spec"], edited)
            except ValueError as exc:
                st.error(str(exc))
            else:
                mark_mapping_edited(state, "apply_table_edits")
                st.rerun()

    with structure_tab:
        render_manual_structure_editor(state)

    with special_tab:
        render_special_missing_editor(state, df, target, positive_class)

    with compare_tab:
        compare_cols = st.columns(2)
        compare_cols[0].markdown("**Original train metrics**")
        compare_cols[0].dataframe(metrics_frame(original_train["metrics"]), hide_index=True, width="stretch")
        compare_cols[1].markdown("**Current train metrics**")
        compare_cols[1].dataframe(metrics_frame(current_train["metrics"]), hide_index=True, width="stretch")
        if reports.get("current_test") is not None:
            test_cols = st.columns(2)
            test_cols[0].markdown("**Original test metrics**")
            test_cols[0].dataframe(
                metrics_frame(reports["original_test"]["metrics"]), hide_index=True, width="stretch"
            )
            test_cols[1].markdown("**Current test metrics**")
            test_cols[1].dataframe(metrics_frame(reports["current_test"]["metrics"]), hide_index=True, width="stretch")
        st.markdown("**Original bins**")
        st.dataframe(editable_bin_table(original_train["table"]), hide_index=True, width="stretch")


def render_woe_workspace(
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    features: list[str],
    positive_class: Any,
    data_key: str,
) -> None:
    st.subheader("WOE Binning")
    st.caption("Variable generation workspace: each variable is edited independently, then exported as one mapping.")
    if positive_class is None:
        st.error("WOE Binning requires a binary target and a positive class.")
        return
    if not features:
        st.info("No active variables are selected in Data Setup.")
        return

    project = get_project(data_key, target, positive_class)
    selected_variables, config, replace_existing = build_config_from_sidebar(features)
    run_clicked = st.sidebar.button(
        "Run initial WOE binning",
        width="stretch",
        type="primary",
        disabled=not selected_variables,
    )
    if run_clicked:
        run_initial_binning(project, df, target, selected_variables, positive_class, config, replace_existing)
        st.rerun()

    summary = render_catalog(project, df, test_df, target, positive_class)
    if not summary.empty:
        render_project_exports(project, df, test_df, target, positive_class)
        render_variable_editor(project, df, test_df, target, positive_class, summary)
