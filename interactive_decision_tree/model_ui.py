from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .model_tools import (
    changed_value_rows,
    feature_alignment,
    kernel_shap_contributions,
    model_capabilities,
    model_class_labels,
    model_feature_names,
    predict_model_scores,
    prepare_model_frame,
    require_shap,
    shap_global_importance,
    shap_local_contributions,
    load_model_from_bytes,
)


MODEL_STATE_KEY = "_interactive_tree_model_pipeline_state"
SHAP_RESULT_KEY = "_interactive_tree_shap_result"


def model_state() -> dict[str, Any] | None:
    state = st.session_state.get(MODEL_STATE_KEY)
    return state if isinstance(state, dict) else None


def active_model_state(data_key: str) -> dict[str, Any] | None:
    state = model_state()
    if not state or state.get("data_key") != data_key:
        return None
    return state


def selected_model_features(state: dict[str, Any], _fallback_features: list[str]) -> list[str]:
    feature_names = state.get("feature_names")
    if isinstance(feature_names, list) and feature_names:
        return [str(feature) for feature in feature_names]
    return []


def render_model_summary(state: dict[str, Any], df: pd.DataFrame, features: list[str]) -> None:
    feature_names = selected_model_features(state, features)
    capabilities = state.get("capabilities", {})
    metric_cols = st.columns(4)
    metric_cols[0].metric("Model", str(state.get("name") or "loaded"))
    metric_cols[1].metric("Input columns", f"{len(feature_names):,}")
    metric_cols[2].metric("Classes", f"{len(state.get('classes') or []):,}")
    metric_cols[3].metric("Output", state.get("score_output", "predict_proba"))
    st.dataframe(
        pd.DataFrame(
            [
                {"setting": "predict_proba", "value": bool(capabilities.get("predict_proba"))},
                {"setting": "decision_function", "value": bool(capabilities.get("decision_function"))},
                {"setting": "predict", "value": bool(capabilities.get("predict"))},
                {"setting": "positive_class", "value": str(state.get("positive_class"))},
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Feature alignment", expanded=False):
        st.dataframe(feature_alignment(df, feature_names), hide_index=True, width="stretch")


def render_model_setup_workspace(
    *,
    df: pd.DataFrame,
    target: str,
    features: list[str],
    positive_class: Any,
    data_key: str,
) -> None:
    st.subheader("Pipeline model")
    st.warning("Pickle/joblib files execute Python object loading. Use only trusted local model artifacts.")
    uploaded = st.file_uploader("Model pickle/joblib", type=["pkl", "pickle", "joblib"])
    if st.button("Load model pipeline", width="stretch", type="primary", disabled=uploaded is None):
        try:
            model = load_model_from_bytes(uploaded.getvalue())
            feature_names = model_feature_names(model)
            if not feature_names:
                raise ValueError("Model pipeline must expose feature_names_in_ for raw feature alignment.")
            input_frame = prepare_model_frame(df.head(5), feature_names)
            classes = model_class_labels(model)
            prediction = predict_model_scores(model, input_frame, positive_class=positive_class)
        except Exception as exc:
            st.session_state.pop(MODEL_STATE_KEY, None)
            st.error(f"Model load failed: {exc}")
        else:
            st.session_state[MODEL_STATE_KEY] = {
                "data_key": data_key,
                "name": uploaded.name,
                "model": model,
                "feature_names": feature_names,
                "classes": classes,
                "positive_class": positive_class,
                "positive_index": prediction.positive_index,
                "score_output": prediction.output_name,
                "capabilities": model_capabilities(model),
            }
            st.session_state.pop(SHAP_RESULT_KEY, None)
            st.success("Model pipeline loaded.")

    state = active_model_state(data_key)
    if state is None:
        st.info("No model pipeline is loaded for this active dataset.")
        return
    render_model_summary(state, df, features)
    feature_names = selected_model_features(state, features)
    missing = [feature for feature in feature_names if feature not in df.columns]
    if missing:
        st.error(f"Active data is missing model input column(s): {', '.join(missing)}")
    else:
        sample_scores = predict_model_scores(
            state["model"],
            prepare_model_frame(df.head(10), feature_names),
            positive_class=state.get("positive_class"),
            positive_index=state.get("positive_index"),
        )
        st.dataframe(
            pd.DataFrame({"row_index": df.head(10).index.tolist(), "score": sample_scores.scores}),
            hide_index=True,
            width="stretch",
        )


def render_shap_workspace(
    *,
    df: pd.DataFrame,
    features: list[str],
    positive_class: Any,
    data_key: str,
) -> None:
    st.subheader("SHAP Analysis")
    state = active_model_state(data_key)
    if state is None:
        st.info("Load a raw-feature pipeline in Model Setup first.")
        return
    try:
        shap = require_shap()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    st.caption(f"Using shap {getattr(shap, '__version__', '')}. No permutation fallback is used.")
    feature_names = selected_model_features(state, features)
    missing = [feature for feature in feature_names if feature not in df.columns]
    if missing:
        st.error(f"Active data is missing model input column(s): {', '.join(missing)}")
        return

    settings_col1, settings_col2, settings_col3 = st.columns(3)
    background_rows = settings_col1.number_input(
        "Background rows",
        min_value=1,
        max_value=min(500, len(df)),
        value=min(50, len(df)),
        step=10,
        format="%d",
        key="shap_background_rows",
    )
    explain_rows = settings_col2.number_input(
        "Explain rows",
        min_value=1,
        max_value=min(100, len(df)),
        value=min(10, len(df)),
        step=1,
        format="%d",
        key="shap_explain_rows",
    )
    nsamples = settings_col3.number_input(
        "Kernel samples",
        min_value=50,
        max_value=2000,
        value=100,
        step=50,
        format="%d",
        key="shap_kernel_samples",
    )
    seed = st.number_input("SHAP sample seed", value=20260514, step=1, key="shap_sample_seed")
    if st.button("Run SHAP analysis", width="stretch", type="primary"):
        try:
            background = df.sample(n=int(background_rows), random_state=int(seed)) if len(df) > int(background_rows) else df
            explain = df.sample(n=int(explain_rows), random_state=int(seed) + 1) if len(df) > int(explain_rows) else df
            with st.status("Running Kernel SHAP", expanded=True) as status:
                status.write(f"Background rows: {len(background):,}")
                status.write(f"Explain rows: {len(explain):,}")
                result = kernel_shap_contributions(
                    state["model"],
                    background,
                    explain,
                    feature_names,
                    positive_class=positive_class,
                    positive_index=state.get("positive_index"),
                    nsamples=int(nsamples),
                )
                status.update(label="SHAP analysis complete", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"SHAP failed: {exc}")
        else:
            st.session_state[SHAP_RESULT_KEY] = result

    result = st.session_state.get(SHAP_RESULT_KEY)
    if not isinstance(result, dict):
        return
    importance = shap_global_importance(result)
    st.bar_chart(importance.set_index("feature")["mean_abs_shap"])
    st.dataframe(importance, hide_index=True, width="stretch")
    row_labels = [str(index) for index in result.get("row_index", [])]
    selected_row = st.selectbox("Local SHAP row", options=list(range(len(row_labels))), format_func=lambda i: row_labels[i])
    local = shap_local_contributions(result, int(selected_row))
    st.dataframe(local, hide_index=True, width="stretch")


def render_what_if_workspace(
    *,
    df: pd.DataFrame,
    features: list[str],
    data_key: str,
) -> None:
    st.subheader("What-if Simulator")
    state = active_model_state(data_key)
    if state is None:
        st.info("Load a raw-feature pipeline in Model Setup first.")
        return
    feature_names = selected_model_features(state, features)
    missing = [feature for feature in feature_names if feature not in df.columns]
    if missing:
        st.error(f"Active data is missing model input column(s): {', '.join(missing)}")
        return

    row_position = st.number_input(
        "Row position",
        min_value=0,
        max_value=max(0, len(df) - 1),
        value=0,
        step=1,
        format="%d",
        key="what_if_row_position",
    )
    base_row = prepare_model_frame(df.iloc[[int(row_position)]], feature_names)
    edited = st.data_editor(
        base_row,
        hide_index=True,
        width="stretch",
        key=f"what_if_editor::{data_key}::{int(row_position)}",
    )
    if st.button("Score scenario", width="stretch", type="primary"):
        try:
            base_prediction = predict_model_scores(
                state["model"],
                base_row,
                positive_class=state.get("positive_class"),
                positive_index=state.get("positive_index"),
            )
            edited_prediction = predict_model_scores(
                state["model"],
                prepare_model_frame(edited, feature_names),
                positive_class=state.get("positive_class"),
                positive_index=state.get("positive_index"),
            )
        except Exception as exc:
            st.error(f"Scenario scoring failed: {exc}")
        else:
            old_score = float(base_prediction.scores[0])
            new_score = float(edited_prediction.scores[0])
            metric_cols = st.columns(3)
            metric_cols[0].metric("Base score", f"{old_score:.6f}")
            metric_cols[1].metric("Scenario score", f"{new_score:.6f}")
            metric_cols[2].metric("Delta", f"{new_score - old_score:+.6f}")
            changes = changed_value_rows(base_row.iloc[0], prepare_model_frame(edited, feature_names).iloc[0])
            if changes.empty:
                st.caption("No raw input value changed.")
            else:
                st.dataframe(changes, hide_index=True, width="stretch")
