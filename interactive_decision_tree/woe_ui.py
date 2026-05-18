from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from .woe_binning import (
    WoeBuildConfig,
    apply_bin_table_edits,
    apply_categorical_groups,
    apply_numeric_cutpoints,
    build_initial_spec,
    categorical_groups_from_spec,
    copy_spec,
    cutpoints_from_spec,
    evaluate_original_current,
    evaluate_spec,
    merge_selected_bins,
    normal_bins,
    parse_category_groups,
    parse_cutpoints,
    parse_special_values,
)
from .woe_export import (
    build_project_export,
    project_excel_bytes,
    project_json_bytes,
    project_python_transformer_text,
    project_sql_text,
    variable_state_rows,
)


WOE_PROJECTS_KEY = "_interactive_tree_woe_projects"
WOE_ACTIVE_VARIABLE_KEY = "_interactive_tree_woe_active_variable"


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


def status_options() -> list[str]:
    return ["auto", "edited", "approved", "rejected", "needs_review"]


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


def build_config_from_sidebar(features: list[str]) -> tuple[list[str], WoeBuildConfig, bool]:
    default_variables = features[: min(20, len(features))]
    remembered_variables = [
        str(variable)
        for variable in st.session_state.get("woe_selected_variables", default_variables)
        if str(variable) in features
    ]
    if not remembered_variables:
        remembered_variables = default_variables
    selected_variables = st.sidebar.multiselect(
        "WOE variables",
        options=features,
        default=remembered_variables,
        key="woe_selected_variables",
    )
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
    replace_existing = st.sidebar.checkbox("Replace existing variable specs", value=False, key="woe_replace_existing")
    config = WoeBuildConfig(
        max_bins=int(max_bins),
        min_bin_size=float(min_bin_size),
        monotonic_trend=str(monotonic_trend),
        missing_separate=bool(missing_separate),
        blank_as_missing=bool(blank_as_missing),
        engine=str(engine),
    )
    return list(selected_variables), config, bool(replace_existing)


def render_catalog(
    project: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
) -> pd.DataFrame:
    summary = variable_state_rows(project, df, test_df, target, positive_class)
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
    export_payload = build_project_export(project, df, test_df, target, positive_class)
    approved_payload = build_project_export(project, df, test_df, target, positive_class, approved_only=True)
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
        data=project_excel_bytes(project, df, test_df, target, positive_class),
        file_name="interactive_woe_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    export_cols[2].download_button(
        "Download Python transformer",
        data=project_python_transformer_text(export_payload),
        file_name="woe_transformer.py",
        mime="text/x-python",
        width="stretch",
    )
    export_cols[3].download_button(
        "Download SQL CASE",
        data=project_sql_text(approved_payload if approved_payload.get("variables") else export_payload),
        file_name="woe_transform.sql",
        mime="text/plain",
        width="stretch",
    )


def render_special_missing_editor(
    state: dict[str, Any],
    df: pd.DataFrame,
    target: str,
    positive_class: Any,
) -> None:
    current_spec = state["current_spec"]
    config = dict(current_spec.get("config", {}))
    st.markdown("**Special / missing policy**")
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
    if st.button("Apply special / missing policy", width="stretch", key=f"woe_apply_special::{state['name']}"):
        new_config = WoeBuildConfig(
            max_bins=int(config.get("max_bins", 6)),
            min_bin_size=float(config.get("min_bin_size", 0.05)),
            monotonic_trend=str(config.get("monotonic_trend", "auto")),
            missing_separate=bool(missing_separate),
            blank_as_missing=bool(blank_as_missing),
            special_values=tuple(parse_special_values(special_values_text)),
            protected_special=bool(protected_special),
            engine=str(config.get("engine", "auto")),
        )
        state["current_spec"] = build_initial_spec(df, target, state["name"], positive_class, new_config)
        state["status"] = "edited"
        add_edit(
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
                str(bin_spec.get("label"))
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
                state["status"] = "edited"
                add_edit(state, "merge_selected_bins", {"bin_ids": selected_bins})
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
            state["status"] = "edited"
            add_edit(state, "apply_cutpoints", {"cutpoints": parse_cutpoints(cutpoint_text)})
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
            state["status"] = "edited"
            add_edit(state, "apply_categorical_groups", {"group_count": len(parse_category_groups(groups_text))})
            st.rerun()


def render_variable_editor(
    project: dict[str, Any],
    df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    positive_class: Any,
) -> None:
    variables = sorted(project.get("variables", {}))
    if not variables:
        return

    remembered = st.session_state.get(WOE_ACTIVE_VARIABLE_KEY)
    default_index = variables.index(remembered) if remembered in variables else 0
    variable = st.selectbox("Variable editor", variables, index=default_index, key=WOE_ACTIVE_VARIABLE_KEY)
    state = project["variables"][variable]
    status = st.selectbox(
        "Variable status",
        status_options(),
        index=status_options().index(state.get("status", "auto"))
        if state.get("status", "auto") in status_options()
        else 0,
        key=f"woe_status::{variable}",
    )
    if status != state.get("status"):
        state["status"] = status
        add_edit(state, "status_change", {"status": status})

    reports = evaluate_original_current(
        df,
        test_df,
        target,
        state["original_spec"],
        state["current_spec"],
        positive_class,
    )
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
                state["status"] = "edited"
                add_edit(state, "apply_table_edits")
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
        render_variable_editor(project, df, test_df, target, positive_class)
