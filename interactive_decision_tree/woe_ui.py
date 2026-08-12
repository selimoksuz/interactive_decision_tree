from __future__ import annotations

import fnmatch
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from .woe_binning import (
    WOE_BINOMIAL_CONFIDENCE_LEVEL,
    WOE_MAX_BINS,
    WoeBuildConfig,
    apply_bin_table_edits,
    apply_categorical_groups,
    apply_numeric_cutpoints,
    assigned_woe_value,
    build_initial_spec,
    categorical_groups_from_spec,
    copy_spec,
    cutpoints_from_spec,
    evaluate_original_current,
    evaluate_spec,
    merge_selected_bins,
    missing_mask,
    optbinning_status,
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
    scoped_variable_states,
    state_export_decision,
    state_mapping_state,
    variable_state_rows,
)

WOE_PROJECTS_KEY = "_interactive_tree_woe_projects"
WOE_CHECKPOINT_DIRTY_KEY = "_interactive_tree_woe_checkpoint_dirty"
WOE_DETAIL_OPEN_PREFIX = "woe_detail_open::"
WOE_ACTIVE_VARIABLE_KEY = "_interactive_tree_woe_active_variable"
WOE_REPORT_CACHE_KEY = "_interactive_tree_woe_report_cache"
WOE_VARIABLE_ROW_CACHE_KEY = "_interactive_tree_woe_variable_row_cache"
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
    signature_spec = {key: value for key, value in spec.items() if key != "evaluation_profile"}
    raw = json.dumps(signature_spec, sort_keys=True, default=str)
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
    st.session_state[WOE_CHECKPOINT_DIRTY_KEY] = True


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
        "hhi_total",
        "hhi_concentration",
        "max_bucket_weight",
        "binomial_significant_bins",
        "binomial_signal_share",
        "engine_used",
    ]
    return pd.DataFrame([{"metric": key, "value": metrics.get(key)} for key in keys])


def woe_column_config(
    confidence_level: float = WOE_BINOMIAL_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    confidence_label = f"{float(confidence_level):.0%} simultaneous CI"
    return {
        "merge": st.column_config.CheckboxColumn("Merge"),
        "bin_order": st.column_config.NumberColumn("Order", format="%d"),
        "count": st.column_config.NumberColumn("Count", format="%d"),
        "event_count": st.column_config.NumberColumn("Event Count", format="%d"),
        "non_event_count": st.column_config.NumberColumn("Non-event Count", format="%d"),
        "bucket_weight": st.column_config.NumberColumn("Bucket Weight", format="%.6f"),
        "event_rate": st.column_config.NumberColumn("Event Rate", format="%.6f"),
        "event_concentration": st.column_config.NumberColumn("Event Share", format="%.6f"),
        "non_event_concentration": st.column_config.NumberColumn("Non-event Share", format="%.6f"),
        "binomial_adjusted_p_value": st.column_config.NumberColumn("Adjusted p-value", format="%.6f"),
        "binomial_result": st.column_config.TextColumn("Binomial Result"),
        "binomial_ci_lower": st.column_config.NumberColumn(f"{confidence_label} Lower", format="%.6f"),
        "binomial_ci_upper": st.column_config.NumberColumn(f"{confidence_label} Upper", format="%.6f"),
        "calculated_woe": st.column_config.NumberColumn("Calculated WOE", format="%.6f"),
        "assigned_woe": st.column_config.NumberColumn("Assigned WOE", format="%.6f"),
        "lower": st.column_config.NumberColumn("Lower", format="%.12g"),
        "upper": st.column_config.NumberColumn("Upper", format="%.12g"),
        "export_woe": st.column_config.NumberColumn("Export WOE", format="%.6f"),
        "calculated_iv": st.column_config.NumberColumn("Calculated IV", format="%.6f"),
        "export_iv": st.column_config.NumberColumn("Export IV", format="%.6f"),
        "protected": st.column_config.CheckboxColumn("Protected"),
    }


def editable_bin_table(table: pd.DataFrame, *, include_merge: bool = False) -> pd.DataFrame:
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
        "bucket_weight",
        "event_rate",
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
    out = table[[column for column in columns if column in table.columns]].copy()
    if include_merge:
        out.insert(0, "merge", False)
    return out


def binomial_test_frame(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "label",
        "bucket_weight",
        "event_rate",
        "binomial_adjusted_p_value",
        "binomial_result",
        "binomial_ci_lower",
        "binomial_ci_upper",
    ]
    return table[[column for column in columns if column in table.columns]].copy()


def selected_normal_merge_bin_ids(edited_table: pd.DataFrame) -> list[str]:
    if edited_table.empty or not {"merge", "bin_id", "kind"}.issubset(edited_table.columns):
        return []
    selected = edited_table["merge"].fillna(False).astype(bool)
    normal = edited_table["kind"].astype(str) == "normal"
    return edited_table.loc[selected & normal, "bin_id"].astype(str).tolist()


def cutpoints_text(spec: dict[str, Any]) -> str:
    return ", ".join(f"{value:.12g}" for value in cutpoints_from_spec(spec))


def category_groups_text(spec: dict[str, Any]) -> str:
    return "\n".join(", ".join(group) for group in categorical_groups_from_spec(spec))


def sync_text_widget_to_spec(widget_key: str, spec: dict[str, Any], value: str) -> None:
    signature_key = f"{widget_key}::spec_signature"
    signature = spec_signature(spec)
    if st.session_state.get(signature_key) != signature:
        st.session_state[widget_key] = value
        st.session_state[signature_key] = signature


def woe_initial_run_status_key(data_key: str, target: str) -> str:
    return f"woe_initial_run_status::{data_key}::{target}"


def render_woe_initial_run_status(status: Any) -> None:
    if not isinstance(status, dict) or not status.get("state"):
        return
    state = str(status.get("state"))
    message = str(status.get("message") or "")
    if state == "done":
        st.success(message or "Done")
    elif state == "failed":
        st.error(message or "Failed")
    else:
        st.info(message or "Running")


def mark_woe_refresh_done(data_key: str, target: str) -> None:
    status_key = woe_initial_run_status_key(data_key, target)
    status = st.session_state.get(status_key)
    if not isinstance(status, dict) or status.get("state") != "refreshing":
        return
    processed = int(status.get("processed") or 0)
    total = int(status.get("total") or processed)
    st.session_state[status_key] = {
        **status,
        "state": "done",
        "message": f"Done: {processed:,} binned, {total:,} selected.",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def run_initial_binning(
    project: dict[str, Any],
    df: pd.DataFrame,
    target: str,
    variables: list[str],
    positive_class: Any,
    config: WoeBuildConfig,
) -> dict[str, int]:
    progress = st.progress(0.0, text="Running WOE binning")
    total = max(1, len(variables))
    processed = 0
    for index, variable in enumerate(variables, start=1):
        spec = build_initial_spec(df, target, variable, positive_class, config)
        processed += 1
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
                    "engine_used": spec.get("config", {}).get("engine_used", "unknown"),
                }
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        progress.progress(index / total, text=f"Binned {variable}")
    progress.progress(
        1.0,
        text=f"WOE binning finished for {len(variables):,} selected variable(s).",
    )
    st.session_state[WOE_CHECKPOINT_DIRTY_KEY] = True
    return {"processed": processed, "total": len(variables)}


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


def woe_filter_all_key(scope_signature: str, filtered_signature: str) -> str:
    return f"woe_variable_filter_all::{scope_signature}::{filtered_signature}"


def woe_filter_item_key(scope_signature: str, filtered_signature: str, variable: str) -> str:
    return f"woe_variable_filter_item::{scope_signature}::{filtered_signature}::{variable}"


def set_widget_state(key: str, value: Any) -> None:
    if st.session_state.get(key) != value:
        st.session_state[key] = value


def apply_woe_filtered_selection(
    selected_key: str,
    variables: list[str],
    filtered_variables: list[str],
    visible_variables: list[str],
    checkbox_key: str,
    scope_signature: str,
    filtered_signature: str,
) -> None:
    selected_variables = update_variable_selection_for_filtered(
        variables,
        st.session_state.get(selected_key, []),
        filtered_variables,
        bool(st.session_state.get(checkbox_key)),
    )
    st.session_state[selected_key] = selected_variables
    selected = set(selected_variables)
    for variable in visible_variables:
        st.session_state[woe_filter_item_key(scope_signature, filtered_signature, variable)] = variable in selected


def apply_woe_item_selection(
    selected_key: str,
    variables: list[str],
    variable: str,
    checkbox_key: str,
    select_all_key: str,
    filtered_variables: list[str],
) -> None:
    selected = set(normalize_variable_selection(variables, st.session_state.get(selected_key, [])))
    if bool(st.session_state.get(checkbox_key)):
        selected.add(str(variable))
    else:
        selected.discard(str(variable))
    selected_variables = ordered_variable_selection(variables, selected)
    st.session_state[selected_key] = selected_variables
    if filtered_variables:
        selected_set = set(selected_variables)
        st.session_state[select_all_key] = all(variable in selected_set for variable in filtered_variables)


def woe_variable_selection_key(data_key: str, target: str) -> str:
    return f"woe_selected_variables::{data_key}::{target}"


def current_woe_variable_selection(variables: list[str], *, data_key: str, target: str) -> list[str]:
    default_variables = variables[: min(20, len(variables))]
    selected_key = woe_variable_selection_key(data_key, target)
    if selected_key not in st.session_state:
        st.session_state[selected_key] = default_variables
    else:
        st.session_state[selected_key] = normalize_variable_selection(
            variables,
            st.session_state.get(selected_key),
        )
    return normalize_variable_selection(
        variables,
        st.session_state.get(selected_key, default_variables),
    )


def scoped_woe_project(project: dict[str, Any], selected_variables: list[str]) -> dict[str, Any]:
    selected = {str(variable) for variable in selected_variables}
    scoped_project = dict(project)
    scoped_project["variables"] = {
        variable: state
        for variable, state in project.get("variables", {}).items()
        if str(variable) in selected
    }
    return scoped_project


def woe_detail_open_key(data_key: str, target: str) -> str:
    return f"{WOE_DETAIL_OPEN_PREFIX}{data_key}::{target}"


def render_woe_variable_selector(variables: list[str], *, data_key: str, target: str) -> list[str]:
    selected_key = woe_variable_selection_key(data_key, target)

    st.markdown("**WOE variables**")
    selected_variables = current_woe_variable_selection(variables, data_key=data_key, target=target)
    scope_signature = hashlib.sha256(f"{data_key}::{target}".encode("utf-8")).hexdigest()[:10]
    with st.popover("Dropdown Filter Panel"):
        query = st.text_input(
            "Search variable",
            placeholder="Search",
            key=f"woe_variable_search::{scope_signature}",
            label_visibility="collapsed",
            help="Case-insensitive contains search. Use comma/newline for multiple terms; * or % works as wildcard.",
        )
        filtered_variables = filter_variable_options(variables, query)
        filtered_signature = hashlib.sha256(
            json.dumps(filtered_variables, default=str).encode("utf-8")
        ).hexdigest()[:10]
        st.caption(
            f"{len(selected_variables):,} selected | {len(filtered_variables):,} matching | "
            f"{len(variables):,} active Data Setup variable(s)"
        )

        selected_set = set(selected_variables)
        all_filtered_selected = bool(filtered_variables) and all(
            variable in selected_set for variable in filtered_variables
        )
        visible_variables = filtered_variables[:WOE_VARIABLE_FILTER_MAX_VISIBLE]
        select_all_key = woe_filter_all_key(scope_signature, filtered_signature)
        set_widget_state(select_all_key, all_filtered_selected)
        st.checkbox(
            "(Select All)",
            key=select_all_key,
            disabled=not filtered_variables,
            on_change=apply_woe_filtered_selection,
            args=(
                selected_key,
                variables,
                filtered_variables,
                visible_variables,
                select_all_key,
                scope_signature,
                filtered_signature,
            ),
        )

        list_container = st.container(height=260, border=True)
        if len(filtered_variables) > len(visible_variables):
            list_container.caption(
                f"Showing first {len(visible_variables):,} matching variable(s). Use search to narrow the list."
            )
        for variable in visible_variables:
            was_selected = variable in selected_set
            item_key = woe_filter_item_key(scope_signature, filtered_signature, variable)
            set_widget_state(item_key, was_selected)
            list_container.checkbox(
                str(variable),
                key=item_key,
                on_change=apply_woe_item_selection,
                args=(selected_key, variables, variable, item_key, select_all_key, filtered_variables),
            )
        if not filtered_variables:
            st.caption("No matching variables.")
        st.caption("Selections control this WOE workspace and are used when you press Run initial WOE binning.")

    selected_variables = normalize_variable_selection(
        variables,
        st.session_state.get(selected_key, selected_variables),
    )
    st.caption(f"{len(selected_variables):,} selected | {len(variables):,} active Data Setup variable(s)")
    return selected_variables


def build_config_from_sidebar(
    features: list[str],
    *,
    data_key: str,
    target: str,
) -> tuple[list[str], WoeBuildConfig]:
    selected_variables = render_woe_variable_selector(features, data_key=data_key, target=target)
    max_bins = st.number_input(
        "Max bins",
        min_value=2,
        max_value=WOE_MAX_BINS,
        value=6,
        step=1,
        key="woe_max_bins",
        help=(
            "Maximum normal bins. Missing and special bins are added separately. "
            "To obtain more than 20 populated bins, set Min bin size below 5%."
        ),
    )
    min_bin_size = st.slider(
        "Min bin size",
        min_value=0.01,
        max_value=0.30,
        value=0.05,
        step=0.01,
        key="woe_min_bin_size",
    )
    monotonic_trend = st.selectbox(
        "Monotonic trend",
        ["auto", "none", "ascending", "descending"],
        index=0,
        key="woe_monotonic_trend",
    )
    binomial_confidence_level = st.selectbox(
        "Binomial confidence level",
        options=[0.90, 0.95, 0.99],
        index=1,
        format_func=lambda value: f"{value:.0%} family-wise (two-sided)",
        key="woe_binomial_confidence_level",
        help="Exact two-sided binomial test with Bonferroni adjustment across non-empty bins.",
    )
    engine = "optbinning"
    status = optbinning_status()
    version_text = status.get("optbinning_version") or "not installed"
    sklearn_text = status.get("sklearn_version") or "unknown"
    if status.get("available"):
        st.caption(f"Binning engine: optbinning {version_text}, scikit-learn {sklearn_text}.")
    else:
        st.error(f"optbinning cannot run: {status.get('error')}")
    missing_separate = st.checkbox("Missing as separate bin", value=True, key="woe_missing_separate")
    blank_as_missing = st.checkbox("Blank string as missing", value=True, key="woe_blank_as_missing")
    config = WoeBuildConfig(
        max_bins=int(max_bins),
        min_bin_size=float(min_bin_size),
        monotonic_trend=str(monotonic_trend),
        binomial_confidence_level=float(binomial_confidence_level),
        missing_separate=bool(missing_separate),
        blank_as_missing=bool(blank_as_missing),
        engine=str(engine),
    )
    return list(selected_variables), config


def metric_delta(current: Any, original: Any) -> float | None:
    try:
        if current is None or original is None:
            return None
        return float(current) - float(original)
    except (TypeError, ValueError):
        return None


def variable_summary_row(
    variable: str,
    state: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
) -> dict[str, Any]:
    current_spec = state["current_spec"]
    original_spec = state["original_spec"]
    current_train = evaluate_spec(df, target, current_spec, positive_class, "Train")["metrics"]
    if spec_signature(current_spec) == spec_signature(original_spec):
        original_train = current_train
    else:
        original_train = evaluate_spec(df, target, original_spec, positive_class, "Train")["metrics"]
    row = {
        "variable": variable,
        "type": current_spec.get("feature_kind"),
        "mapping_state": state_mapping_state(state),
        "export_decision": state_export_decision(state),
        "engine": current_train.get("engine_used"),
        "original_iv": original_train.get("export_iv"),
        "current_iv": current_train.get("export_iv"),
        "iv_delta": metric_delta(current_train.get("export_iv"), original_train.get("export_iv")),
        "original_gini": original_train.get("export_gini"),
        "current_gini": current_train.get("export_gini"),
        "gini_delta": metric_delta(current_train.get("export_gini"), original_train.get("export_gini")),
        "bins": current_train.get("bin_count"),
        "manual_woe_bins": current_train.get("manual_woe_bins"),
        "monotonic": current_train.get("is_monotonic"),
        "monotonic_direction": current_train.get("monotonic_direction"),
        "monotonic_violations": current_train.get("monotonic_violation_count"),
        "hhi_total": current_train.get("hhi_total"),
        "normalized_hhi": current_train.get("normalized_hhi"),
        "hhi_concentration": current_train.get("hhi_concentration"),
        "max_bin_concentration": current_train.get("max_bin_concentration"),
        "max_bucket_weight": current_train.get("max_bucket_weight"),
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
    return row


def variable_summary_placeholder(variable: str, state: dict[str, Any]) -> dict[str, Any]:
    current_spec = state.get("current_spec") if isinstance(state.get("current_spec"), dict) else {}
    bins = current_spec.get("bins", []) if isinstance(current_spec, dict) else []
    return {
        "variable": variable,
        "type": current_spec.get("feature_kind"),
        "mapping_state": state_mapping_state(state),
        "export_decision": state_export_decision(state),
        "engine": current_spec.get("config", {}).get("engine_used") if isinstance(current_spec.get("config"), dict) else None,
        "original_iv": None,
        "current_iv": None,
        "iv_delta": None,
        "original_gini": None,
        "current_gini": None,
        "gini_delta": None,
        "bins": len(bins),
        "manual_woe_bins": sum(1 for bin_spec in bins if assigned_woe_value(bin_spec) is not None),
        "monotonic": None,
        "monotonic_direction": None,
        "monotonic_violations": None,
        "hhi_total": None,
        "normalized_hhi": None,
        "hhi_concentration": None,
        "max_bin_concentration": None,
        "max_bucket_weight": None,
        "binomial_significant_bins": None,
        "binomial_signal_rate": None,
        "binomial_signal_share": None,
        "metrics_status": "not loaded",
    }


@st.fragment
def render_woe_sidebar_controls(
    project: dict[str, Any],
    df: pd.DataFrame,
    target: str,
    features: list[str],
    positive_class: Any,
    data_key: str,
) -> None:
    selected_variables, config = build_config_from_sidebar(
        features,
        data_key=data_key,
        target=target,
    )
    status_key = woe_initial_run_status_key(data_key, target)
    run_clicked = st.button(
        "Run initial WOE binning",
        width="stretch",
        type="primary",
        disabled=not selected_variables,
    )
    status_slot = st.empty()
    with status_slot.container():
        render_woe_initial_run_status(st.session_state.get(status_key))
    if run_clicked:
        started_at = datetime.now(timezone.utc).isoformat()
        st.session_state[status_key] = {
            "state": "running",
            "message": f"Running initial WOE binning for {len(selected_variables):,} selected variable(s).",
            "started_at": started_at,
            "total": len(selected_variables),
            "processed": 0,
        }
        with status_slot.container():
            render_woe_initial_run_status(st.session_state.get(status_key))
        try:
            result = run_initial_binning(
                project,
                df,
                target,
                selected_variables,
                positive_class,
                config,
            )
        except Exception as exc:
            st.session_state[status_key] = {
                "state": "failed",
                "message": f"Failed: {exc}",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "total": len(selected_variables),
            }
            with status_slot.container():
                render_woe_initial_run_status(st.session_state.get(status_key))
            raise
        st.session_state[status_key] = {
            "state": "done",
            "message": (
                f"Done: {result['processed']:,} binned, {result['total']:,} selected. "
                "Catalog and variable metrics load on demand."
            ),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "processed": result["processed"],
            "total": result["total"],
        }
        with status_slot.container():
            render_woe_initial_run_status(st.session_state.get(status_key))
        st.rerun()


def cached_variable_state_rows(
    project: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
    *,
    compute_missing: bool = False,
) -> pd.DataFrame:
    row_cache = session_cache(WOE_VARIABLE_ROW_CACHE_KEY)
    data_sig = data_signature(df, test_df, target, positive_class)
    rows: list[dict[str, Any]] = []
    for variable, state in scoped_variable_states(project):
        current_spec = state.get("current_spec")
        original_spec = state.get("original_spec")
        if not isinstance(current_spec, dict) or not isinstance(original_spec, dict):
            continue
        key = (
            str(project.get("project_key")),
            str(variable),
            data_sig,
            str(state.get("status", "")),
            state_export_decision(state),
            str(state.get("updated_at", "")),
            spec_signature(original_spec),
            spec_signature(current_spec),
        )
        if key not in row_cache:
            if compute_missing:
                computed_row = variable_summary_row(variable, state, df, test_df, target, positive_class)
                computed_row["metrics_status"] = "loaded"
                bounded_cache_set(
                    row_cache,
                    key,
                    computed_row,
                    max_items=4096,
                )
            else:
                rows.append(variable_summary_placeholder(variable, state))
                continue
        cached_row = dict(row_cache[key])
        cached_row.setdefault("metrics_status", "loaded")
        rows.append(cached_row)
    return pd.DataFrame(rows)


def report_cache_key(
    project: dict[str, Any],
    state: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
) -> tuple[Any, ...]:
    return (
        str(project.get("project_key")),
        str(state.get("name")),
        data_signature(df, test_df, target, positive_class),
        str(state.get("status", "")),
        str(state.get("export_decision", "")),
        str(state.get("updated_at", "")),
        spec_signature(state.get("original_spec")),
        spec_signature(state.get("current_spec")),
    )


def has_cached_evaluate_original_current(
    project: dict[str, Any],
    state: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
) -> bool:
    cache = session_cache(WOE_REPORT_CACHE_KEY)
    return report_cache_key(project, state, df, test_df, target, positive_class) in cache


def cached_evaluate_original_current(
    project: dict[str, Any],
    state: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
    *,
    compute_missing: bool = True,
) -> dict[str, Any] | None:
    cache = session_cache(WOE_REPORT_CACHE_KEY)
    key = report_cache_key(project, state, df, test_df, target, positive_class)
    if key not in cache and not compute_missing:
        return None
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
    *,
    compute_missing: bool = False,
) -> pd.DataFrame:
    summary = cached_variable_state_rows(
        project,
        df,
        test_df,
        target,
        positive_class,
        compute_missing=compute_missing,
    )
    st.subheader("WOE variable catalog")
    if summary.empty:
        st.info("Run initial WOE binning to create variable mappings.")
        return summary
    unloaded_count = int((summary.get("metrics_status") == "not loaded").sum()) if "metrics_status" in summary else 0
    if unloaded_count:
        st.caption(
            f"{unloaded_count:,} variable metric row(s) are not loaded. "
            "Use Load / refresh selected catalog metrics when you want to scan the active data."
        )
    display_columns = [
        "variable",
        "type",
        "mapping_state",
        "export_decision",
        "original_iv",
        "current_iv",
        "iv_delta",
        "original_gini",
        "current_gini",
        "gini_delta",
        "bins",
        "monotonic_direction",
        "hhi_total",
        "hhi_concentration",
        "max_bucket_weight",
        "binomial_significant_bins",
        "test_iv",
        "test_gini",
        "metrics_status",
    ]
    display_summary = summary[[column for column in display_columns if column in summary.columns]]
    st.dataframe(
        display_summary,
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
            "hhi_total": st.column_config.NumberColumn(format="%.6f"),
            "max_bucket_weight": st.column_config.NumberColumn("Max Bucket Weight", format="%.6f"),
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
                "label": "Included selected variables",
                "included_only": True,
                "excluded_only": False,
            },
            "excluded": {
                "label": "Excluded selected variables",
                "included_only": False,
                "excluded_only": True,
            },
            "all": {
                "label": "All selected variables",
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
            binomial_confidence_level=float(
                config.get("binomial_confidence_level", WOE_BINOMIAL_CONFIDENCE_LEVEL)
            ),
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


def render_current_bin_structure_controls(state: dict[str, Any]) -> dict[str, Any]:
    current_spec = state["current_spec"]
    if current_spec.get("feature_kind") == "numeric":
        st.markdown("**Numeric cutpoints**")
        cutpoint_key = f"woe_cutpoints::{state['name']}"
        current_cutpoints = cutpoints_from_spec(current_spec)
        sync_text_widget_to_spec(cutpoint_key, current_spec, cutpoints_text(current_spec))
        cutpoint_text = st.text_area(
            "Cutpoints",
            key=cutpoint_key,
            help="Comma or newline separated numeric cutpoints.",
        )
        parsed_cutpoints = parse_cutpoints(cutpoint_text)
        return {
            "kind": "numeric",
            "changed": parsed_cutpoints != current_cutpoints,
            "cutpoints": parsed_cutpoints,
        }
    else:
        st.markdown("**Categorical groups**")
        group_key = f"woe_category_groups::{state['name']}"
        current_groups = categorical_groups_from_spec(current_spec)
        sync_text_widget_to_spec(group_key, current_spec, category_groups_text(current_spec))
        groups_text = st.text_area(
            "One bin per line, values comma separated",
            key=group_key,
            height=180,
        )
        parsed_groups = parse_category_groups(groups_text)
        return {
            "kind": "categorical",
            "changed": parsed_groups != current_groups,
            "groups": parsed_groups,
        }


def apply_current_bin_changes(
    state: dict[str, Any],
    edited_table: pd.DataFrame,
    selected_merge_bins: list[str],
    structure_change: dict[str, Any],
) -> list[str]:
    current_spec = state["current_spec"]
    actions: list[dict[str, Any]] = []

    if bool(structure_change.get("changed")):
        if len(selected_merge_bins) >= 2:
            raise ValueError("Apply cutpoint/group changes first, then select bins to merge after the table refreshes.")
        if structure_change.get("kind") == "numeric":
            updated_spec = apply_numeric_cutpoints(current_spec, list(structure_change.get("cutpoints", [])))
            actions.append({"action": "apply_cutpoints", "cutpoints": list(structure_change.get("cutpoints", []))})
        else:
            groups = list(structure_change.get("groups", []))
            updated_spec = apply_categorical_groups(current_spec, groups)
            actions.append({"action": "apply_categorical_groups", "group_count": len(groups)})
        state["current_spec"] = updated_spec
        mark_mapping_edited(state, "apply_current_bin_changes", {"actions": actions})
        return [str(action["action"]) for action in actions]

    updated_spec = apply_bin_table_edits(current_spec, edited_table)
    if spec_signature(updated_spec) != spec_signature(current_spec):
        actions.append({"action": "apply_table_edits"})

    if len(selected_merge_bins) >= 2:
        updated_spec = merge_selected_bins(updated_spec, selected_merge_bins)
        actions.append({"action": "merge_selected_bins", "bin_ids": selected_merge_bins})

    if not actions:
        return []

    state["current_spec"] = updated_spec
    mark_mapping_edited(state, "apply_current_bin_changes", {"actions": actions})
    return [str(action["action"]) for action in actions]


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
    if remembered not in variables:
        st.session_state[WOE_ACTIVE_VARIABLE_KEY] = variables[0]
        remembered = variables[0]
    default_index = variables.index(remembered)
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

    reports_are_cached = has_cached_evaluate_original_current(project, state, df, test_df, target, positive_class)
    load_report = st.button(
        "Load selected variable metrics",
        width="stretch",
        key=f"woe_load_variable_metrics::{variable}",
        help="Scans the active data only for this variable. Use this before editing current bins.",
    )
    reports = cached_evaluate_original_current(
        project,
        state,
        df,
        test_df,
        target,
        positive_class,
        compute_missing=load_report or reports_are_cached,
    )
    if reports is None:
        st.info("Selected variable bin table and metrics are not loaded. Press Load selected variable metrics.")
        return

    current_train = reports["current_train"]
    original_train = reports["original_train"]

    metric_cols = st.columns(5)
    metric_cols[0].metric("Current IV", f"{current_train['metrics']['export_iv']:.6f}")
    current_gini = current_train["metrics"].get("export_gini")
    metric_cols[1].metric("Current Gini", "n/a" if current_gini is None else f"{current_gini:.6f}")
    metric_cols[2].metric("Bins", str(current_train["metrics"].get("bin_count")))
    metric_cols[3].metric("Manual WOE bins", str(current_train["metrics"].get("manual_woe_bins")))
    metric_cols[4].metric("Monotonic", str(current_train["metrics"].get("monotonic_direction")))
    test_cols = st.columns(4)
    hhi_total = current_train["metrics"].get("hhi_total")
    hhi_label = current_train["metrics"].get("hhi_concentration")
    max_bucket_weight = current_train["metrics"].get("max_bucket_weight")
    signal_share = current_train["metrics"].get("binomial_signal_share")
    signal_bins = current_train["metrics"].get("binomial_significant_bins")
    hhi_value = "n/a" if hhi_total is None else f"{hhi_total:.6f}"
    if hhi_label and hhi_value != "n/a":
        hhi_value = f"{hhi_value} ({hhi_label})"
    test_cols[0].metric("HHI (bucket weights)", hhi_value)
    test_cols[1].metric(
        "Max Bucket Weight",
        "n/a" if max_bucket_weight is None else f"{max_bucket_weight:.2%}",
    )
    test_cols[2].metric("Different buckets", "n/a" if signal_bins is None else str(signal_bins))
    test_cols[3].metric(
        "Different bucket weight",
        "n/a" if signal_share is None else f"{signal_share:.2%}",
    )

    confidence_level = float(
        current_train["metrics"].get("binomial_confidence_level", WOE_BINOMIAL_CONFIDENCE_LEVEL)
    )
    reference_rate = current_train["metrics"].get("binomial_reference_event_rate")
    family_size = current_train["metrics"].get("binomial_family_size")
    reference_text = "n/a" if reference_rate is None else f"{float(reference_rate):.4%}"
    st.caption(
        f"Binomial: exact two-sided (two-tail), H0 bucket event rate = total event rate ({reference_text}); "
        f"{confidence_level:.0%} family-wise confidence with Bonferroni across {family_size or 0} non-empty "
        "bucket(s). Bucket Weight = bucket count / scored rows."
    )

    bins_tab, special_tab, compare_tab = st.tabs(
        ["Current bins", "Special / missing", "Original comparison"]
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
            "bucket_weight",
            "event_rate",
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
        editor_columns = ["merge", "kind", "label"]
        if state["current_spec"].get("feature_kind") == "numeric":
            editor_columns.extend(["lower", "upper"])
        else:
            editor_columns.append("values")
        editor_columns.extend(
            [
                "count",
                "event_count",
                "non_event_count",
                "bucket_weight",
                "event_rate",
                "event_concentration",
                "non_event_concentration",
                "calculated_woe",
                "assigned_woe",
                "export_woe",
                "calculated_iv",
                "export_iv",
                "note",
            ]
        )
        edited = st.data_editor(
            editable_bin_table(current_train["table"], include_merge=True),
            hide_index=True,
            width="stretch",
            key=editor_key,
            disabled=disabled_columns,
            column_order=editor_columns,
            column_config=woe_column_config(confidence_level),
        )
        selected_merge_bins = selected_normal_merge_bin_ids(edited)
        merge_help = (
            "Numeric variables require adjacent selected bins. "
            "Categorical variables can merge any selected normal bins."
        )
        st.caption(f"{len(selected_merge_bins):,} normal bin(s) selected for merge. {merge_help}")
        structure_change = render_current_bin_structure_controls(state)
        if st.button("Apply current bin changes", width="stretch", key=f"woe_apply_current_bins::{variable}"):
            try:
                applied_actions = apply_current_bin_changes(state, edited, selected_merge_bins, structure_change)
            except ValueError as exc:
                st.error(str(exc))
            else:
                if applied_actions:
                    st.rerun()
                elif len(selected_merge_bins) == 1:
                    st.warning("Select at least two normal bins to merge.")
                else:
                    st.info("No current bin changes to apply.")

        with st.expander("Binomial / HHI diagnostics", expanded=False):
            st.dataframe(
                binomial_test_frame(current_train["table"]),
                hide_index=True,
                width="stretch",
                column_config=woe_column_config(confidence_level),
            )

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
        original_confidence = float(
            original_train["metrics"].get("binomial_confidence_level", WOE_BINOMIAL_CONFIDENCE_LEVEL)
        )
        st.dataframe(
            editable_bin_table(original_train["table"]),
            hide_index=True,
            width="stretch",
            column_order=[column for column in editor_columns if column != "merge"],
            column_config=woe_column_config(original_confidence),
        )


def render_woe_workspace(
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    features: list[str],
    positive_class: Any,
    data_key: str,
) -> None:
    st.caption("Variable generation workspace: each variable is edited independently, then exported as one mapping.")
    if positive_class is None:
        st.error("WOE Binning requires a binary target and a positive class.")
        return
    if not features:
        st.info("No active variables are selected in Data Setup.")
        return

    project = get_project(data_key, target, positive_class)
    active_woe_variables = current_woe_variable_selection(features, data_key=data_key, target=target)
    visible_project = scoped_woe_project(project, active_woe_variables)
    visible_mapping_count = len(visible_project.get("variables", {}))
    with st.sidebar:
        render_woe_sidebar_controls(
            project,
            df,
            target,
            features,
            positive_class,
            data_key,
        )

    st.caption("Configure WOE variables and binning settings first. Data scans run only from explicit action buttons.")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Active variables", f"{len(features):,}")
    metric_cols[1].metric("WOE selected", f"{len(active_woe_variables):,}")
    metric_cols[2].metric("Selected mappings", f"{visible_mapping_count:,}")
    metric_cols[3].metric("All mappings", f"{len(project.get('variables', {})):,}")
    if project.get("variables") and not visible_project.get("variables"):
        st.info("No stored mappings match the current WOE variable selection.")

    detail_key = woe_detail_open_key(data_key, target)
    detail_cols = st.columns([1, 1])
    if detail_cols[0].button(
        "Open WOE detail workspace",
        width="stretch",
        disabled=visible_mapping_count == 0,
        help="Shows catalog, exports, and variable editor. Metric and bin scans still load on demand.",
    ):
        st.session_state[detail_key] = True
        st.rerun()
    if detail_cols[1].button(
        "Hide WOE detail workspace",
        width="stretch",
        disabled=not bool(st.session_state.get(detail_key)),
    ):
        st.session_state[detail_key] = False
        st.rerun()

    if visible_mapping_count == 0:
        st.info("Use the sidebar to select variables and run initial WOE binning.")
        mark_woe_refresh_done(data_key, target)
        return

    if not st.session_state.get(detail_key):
        st.info("WOE mappings are ready. Open the detail workspace only when you need catalog, export, or bin editing.")
        mark_woe_refresh_done(data_key, target)
        return

    refresh_catalog = st.button(
        "Load / refresh selected catalog metrics",
        width="stretch",
        disabled=not bool(visible_project.get("variables")),
        help="Scans the active data for the selected WOE variables. Page switching does not run this automatically.",
    )
    summary = render_catalog(
        visible_project,
        df,
        test_df,
        target,
        positive_class,
        compute_missing=refresh_catalog,
    )
    if not summary.empty:
        render_project_exports(visible_project, df, test_df, target, positive_class)
        render_variable_editor(visible_project, df, test_df, target, positive_class, summary)
    mark_woe_refresh_done(data_key, target)
