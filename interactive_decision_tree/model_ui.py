from __future__ import annotations

from typing import Any

import numpy as np
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
    shap_explanation,
    shap_global_importance,
    shap_local_contributions,
    shap_plot_data_frame,
    load_model_from_bytes,
)


MODEL_STATE_KEY = "_interactive_tree_model_pipeline_state"
SHAP_RESULT_KEY = "_interactive_tree_shap_result"
WHAT_IF_SENSITIVITY_CACHE_KEY = "_interactive_tree_what_if_sensitivity_cache"
DEFAULT_SHAP_SCORE_BINS = 10
WHAT_IF_SENSITIVITY_MAX_VALUES = 8
DEFAULT_SHAP_STRATIFY_CANDIDATES = (
    "segment",
    "product",
    "channel",
    "region",
    "risk_band_hint",
    "collection_status",
    "employment_type",
)


def model_state() -> dict[str, Any] | None:
    state = st.session_state.get(MODEL_STATE_KEY)
    return state if isinstance(state, dict) else None


def active_model_state(data_key: str) -> dict[str, Any] | None:
    state = model_state()
    if not state or state.get("data_key") != data_key:
        return None
    return state


def selected_model_features(state: dict[str, Any], _fallback_features: list[str]) -> list[str]:
    model = state.get("model")
    if model is not None:
        current_feature_names = model_feature_names(model)
        if current_feature_names:
            saved_feature_names = state.get("feature_names")
            if list(saved_feature_names or []) != current_feature_names:
                state["feature_names"] = current_feature_names
                st.session_state.pop(SHAP_RESULT_KEY, None)
            return current_feature_names
    feature_names = state.get("feature_names")
    if isinstance(feature_names, list) and feature_names:
        return [str(feature) for feature in feature_names]
    return []


def what_if_id_columns(df: pd.DataFrame, target: str) -> list[str]:
    return [str(column) for column in df.columns if str(column) != str(target)]


def default_id_column(columns: list[str]) -> str | None:
    if not columns:
        return None
    exact_candidates = {"customer_id", "cust_id", "client_id", "account_id", "application_id", "id"}
    for column in columns:
        lowered = column.lower()
        if lowered in exact_candidates or lowered.endswith("_id"):
            return column
    for column in columns:
        lowered = column.lower()
        if "id" in lowered or "musteri" in lowered or "customer" in lowered:
            return column
    return columns[0]


def find_row_position_by_id_value(df: pd.DataFrame, id_column: str, raw_value: Any) -> int | None:
    if id_column not in df.columns:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    series = df[id_column]
    mask = series.astype("string").str.strip().eq(text).fillna(False)
    if not bool(mask.any()):
        numeric_value = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
        if pd.notna(numeric_value):
            mask = pd.to_numeric(series, errors="coerce").eq(float(numeric_value)).fillna(False)
    positions = np.flatnonzero(mask.to_numpy(dtype=bool))
    if len(positions) == 0:
        return None
    return int(positions[0])


def same_raw_value(left: Any, right: Any) -> bool:
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    if bool(left_missing) and bool(right_missing):
        return True
    if bool(left_missing) != bool(right_missing):
        return False
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def what_if_candidate_values(series: pd.Series, current_value: Any, max_values: int = WHAT_IF_SENSITIVITY_MAX_VALUES) -> list[Any]:
    max_values = max(1, int(max_values))
    candidates: list[Any] = []

    def add(value: Any) -> None:
        if same_raw_value(value, current_value):
            return
        if any(same_raw_value(value, existing) for existing in candidates):
            return
        candidates.append(value)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return []
        quantiles = np.linspace(0.05, 0.95, min(max_values, 7))
        for value in numeric.quantile(quantiles).tolist():
            add(float(value))
        add(float(numeric.min()))
        add(float(numeric.max()))
    else:
        values = series.dropna().value_counts(dropna=True).head(max_values).index.tolist()
        for value in values:
            add(value)
        if not pd.isna(current_value) and series.isna().any() and len(candidates) < max_values:
            add(np.nan)

    return candidates[:max_values]


def what_if_local_sensitivity(
    *,
    model: Any,
    df: pd.DataFrame,
    row_position: int,
    feature_names: list[str],
    positive_class: Any | None = None,
    positive_index: int | None = None,
    max_values: int = WHAT_IF_SENSITIVITY_MAX_VALUES,
) -> pd.DataFrame:
    base_row = prepare_model_frame(df.iloc[[int(row_position)]], feature_names)
    base_values = base_row.iloc[0].to_dict()
    base_prediction = predict_model_scores(
        model,
        base_row,
        positive_class=positive_class,
        positive_index=positive_index,
    )
    base_score = float(base_prediction.scores[0])

    variant_rows: list[dict[str, Any]] = []
    variant_meta: list[dict[str, Any]] = []
    for feature in feature_names:
        for candidate_value in what_if_candidate_values(df[feature], base_values[feature], max_values=max_values):
            row = dict(base_values)
            row[feature] = candidate_value
            variant_rows.append(row)
            variant_meta.append({"feature": feature, "candidate_value": candidate_value})

    if not variant_rows:
        return pd.DataFrame(
            [
                {
                    "feature": feature,
                    "sensitivity": 0.0,
                    "delta": 0.0,
                    "base_score": base_score,
                    "scenario_score": base_score,
                    "candidate_value": base_values[feature],
                    "candidate_count": 0,
                }
                for feature in feature_names
            ]
        )

    variant_frame = pd.DataFrame(variant_rows, columns=feature_names)
    variant_prediction = predict_model_scores(
        model,
        prepare_model_frame(variant_frame, feature_names),
        positive_class=positive_class,
        positive_index=positive_index,
    )
    rows_by_feature: dict[str, dict[str, Any]] = {
        feature: {
            "feature": feature,
            "sensitivity": 0.0,
            "delta": 0.0,
            "base_score": base_score,
            "scenario_score": base_score,
            "candidate_value": base_values[feature],
            "candidate_count": 0,
        }
        for feature in feature_names
    }
    for meta, score in zip(variant_meta, variant_prediction.scores):
        feature = str(meta["feature"])
        delta = float(score) - base_score
        row = rows_by_feature[feature]
        row["candidate_count"] = int(row["candidate_count"]) + 1
        if abs(delta) > float(row["sensitivity"]):
            row.update(
                {
                    "sensitivity": abs(delta),
                    "delta": delta,
                    "scenario_score": float(score),
                    "candidate_value": meta["candidate_value"],
                }
            )

    return pd.DataFrame(rows_by_feature.values()).sort_values(
        ["sensitivity", "feature"],
        ascending=[False, True],
    )


def shap_default_stratify_columns(df: pd.DataFrame, target: str, feature_names: list[str]) -> list[str]:
    out: list[str] = []
    for column in DEFAULT_SHAP_STRATIFY_CANDIDATES:
        if column in df.columns and column != target:
            out.append(column)
        if len(out) >= 2:
            break
    if len(out) < 2:
        for column in df.columns:
            name = str(column)
            if name == target or name in out:
                continue
            if pd.api.types.is_object_dtype(df[column]) or isinstance(df[column].dtype, pd.CategoricalDtype):
                out.append(name)
            if len(out) >= 2:
                break
    return out


def shap_interaction_color_options(feature_names: list[str], main_feature: str) -> list[str]:
    return ["auto"] + [feature for feature in feature_names if feature != main_feature]


def shap_score_band_labels(scores: pd.Series, bins: int = DEFAULT_SHAP_SCORE_BINS) -> pd.Series:
    numeric = pd.to_numeric(scores, errors="coerce")
    out = pd.Series("score_missing", index=scores.index, dtype="object")
    valid = numeric[numeric.notna()]
    if valid.empty:
        return out
    band_count = max(1, min(int(bins), len(valid)))
    if band_count == 1:
        out.loc[valid.index] = "score_01"
        return out
    ranked = valid.rank(method="first")
    codes = pd.qcut(ranked, q=band_count, labels=False, duplicates="drop")
    out.loc[valid.index] = [f"score_{int(code) + 1:02d}" for code in codes]
    return out


def shap_strata_labels(
    df: pd.DataFrame,
    scores: pd.Series,
    target: str,
    extra_columns: list[str],
    score_bins: int = DEFAULT_SHAP_SCORE_BINS,
    include_target: bool = True,
) -> pd.Series:
    parts = [shap_score_band_labels(scores, bins=score_bins).astype(str)]
    if include_target and target in df.columns:
        parts.append(shap_stratify_column_labels(df[target], prefix=str(target), force_categorical=True))
    for column in extra_columns:
        if column in df.columns:
            parts.append(shap_stratify_column_labels(df[column], prefix=str(column)))
    labels = parts[0].copy()
    for part in parts[1:]:
        labels = labels + "|" + part
    return labels


def shap_stratify_column_labels(
    series: pd.Series,
    prefix: str,
    bins: int = DEFAULT_SHAP_SCORE_BINS,
    force_categorical: bool = False,
) -> pd.Series:
    missing_label = f"{prefix}=__MISSING__"
    if force_categorical:
        values = series.astype("object").where(series.notna(), "__MISSING__").astype(str)
        return prefix + "=" + values.astype(str)

    if pd.api.types.is_datetime64_any_dtype(series):
        periods = series.dt.to_period("M").astype("object")
        return pd.Series(
            [missing_label if pd.isna(value) else f"{prefix}={value}" for value in periods],
            index=series.index,
            dtype="object",
        )

    if pd.api.types.is_numeric_dtype(series):
        labels = shap_score_band_labels(pd.to_numeric(series, errors="coerce"), bins=bins)
        return labels.map(lambda value: missing_label if value == "score_missing" else f"{prefix}_{value}")

    lowered_prefix = prefix.lower()
    looks_like_date = any(token in lowered_prefix for token in ("date", "dt", "time", "month", "tarih"))
    if looks_like_date:
        try:
            parsed_dates = pd.to_datetime(series, errors="coerce", format="mixed")
        except TypeError:
            parsed_dates = pd.to_datetime(series, errors="coerce")
        if parsed_dates.notna().sum() >= max(3, int(series.notna().sum() * 0.8)):
            periods = parsed_dates.dt.to_period("M").astype("object")
            return pd.Series(
                [missing_label if pd.isna(value) else f"{prefix}={value}" for value in periods],
                index=series.index,
                dtype="object",
            )

    values = series.astype("object").where(series.notna(), "__MISSING__").astype(str)
    counts = values.value_counts(dropna=False)
    if len(counts) > 50:
        kept = set(counts.head(50).index)
        values = values.where(values.isin(kept), "__OTHER__")
    return prefix + "=" + values.astype(str)


def stratified_sample_by_labels(
    df: pd.DataFrame,
    labels: pd.Series,
    n: int,
    random_state: int,
    exclude_index: pd.Index | None = None,
) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    available = df
    available_labels = labels.reindex(df.index)
    if exclude_index is not None:
        mask = ~available.index.isin(exclude_index)
        if int(mask.sum()) >= int(n):
            available = available.loc[mask]
            available_labels = available_labels.loc[available.index]
    if len(available) <= int(n):
        return available.copy()

    group_sizes = available_labels.groupby(available_labels, dropna=False).size().sort_values(ascending=False)
    if group_sizes.empty:
        return available.sample(n=int(n), random_state=int(random_state)).copy()

    if int(n) >= len(group_sizes):
        quotas = pd.Series(1, index=group_sizes.index, dtype=int)
        remaining = int(n) - int(quotas.sum())
    else:
        quotas = pd.Series(1, index=group_sizes.head(int(n)).index, dtype=int)
        remaining = 0

    if remaining > 0:
        raw = (group_sizes / float(group_sizes.sum())) * remaining
        floors = np.floor(raw).astype(int)
        quotas = quotas.add(floors, fill_value=0).astype(int)
        quotas = pd.Series(
            {label: min(int(quotas.get(label, 0)), int(group_sizes[label])) for label in group_sizes.index},
            dtype=int,
        )
        leftover = int(n) - int(quotas.sum())
        fractional = (raw - floors).sort_values(ascending=False)
        for label in fractional.index:
            if leftover <= 0:
                break
            if int(quotas.get(label, 0)) < int(group_sizes[label]):
                quotas[label] = int(quotas.get(label, 0)) + 1
                leftover -= 1

    sampled_parts: list[pd.DataFrame] = []
    for offset, (label, quota) in enumerate(quotas.items()):
        take = int(quota)
        if take <= 0:
            continue
        group = available.loc[available_labels == label]
        if group.empty:
            continue
        sampled_parts.append(group.sample(n=min(take, len(group)), random_state=int(random_state) + offset))

    sampled = pd.concat(sampled_parts) if sampled_parts else available.iloc[0:0].copy()
    if len(sampled) < int(n):
        remaining_frame = available.drop(index=sampled.index, errors="ignore")
        if not remaining_frame.empty:
            fill = remaining_frame.sample(
                n=min(int(n) - len(sampled), len(remaining_frame)),
                random_state=int(random_state) + 10_000,
            )
            sampled = pd.concat([sampled, fill])
    return sampled.sample(frac=1.0, random_state=int(random_state) + 20_000).copy()


def scored_shap_sampling_pool(
    model: Any,
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    positive_class: Any | None,
    positive_index: int | None,
    pool_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    bounded_rows = max(1, min(int(pool_rows), len(df)))
    pool = df.sample(n=bounded_rows, random_state=int(random_state)).copy() if len(df) > bounded_rows else df.copy()
    scores = predict_model_scores(
        model,
        prepare_model_frame(pool, feature_names),
        positive_class=positive_class,
        positive_index=positive_index,
    ).scores
    return pool, pd.Series(scores, index=pool.index, name="model_score")


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
                {"setting": "artifact_type", "value": str(state.get("artifact_type") or "pipeline")},
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
    uploaded = st.file_uploader("Model or interactive tree pickle/joblib", type=["pkl", "pickle", "joblib"])
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
                "artifact_type": getattr(model, "artifact_type_", "pipeline"),
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
    target: str,
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

    settings_col1, settings_col2, settings_col3, settings_col4 = st.columns(4)
    background_rows = settings_col1.number_input(
        "Background rows",
        min_value=1,
        max_value=min(2_000, len(df)),
        value=min(50, len(df)),
        step=10,
        format="%d",
        key="shap_background_rows",
    )
    explain_rows = settings_col2.number_input(
        "Explain rows",
        min_value=1,
        max_value=min(2_000, len(df)),
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
    default_pool_rows = min(len(df), max(5_000, int(background_rows) + int(explain_rows) * 20))
    pool_rows = settings_col4.number_input(
        "Sampling pool rows",
        min_value=min(len(df), max(1, int(background_rows), int(explain_rows))),
        max_value=min(200_000, len(df)),
        value=default_pool_rows,
        step=1_000,
        format="%d",
        key="shap_sampling_pool_rows",
        help="Rows scored before stratified background/explain selection. Larger pools represent the active data better but add one scoring pass.",
    )
    seed = st.number_input("SHAP sample seed", value=20260514, step=1, key="shap_sample_seed")
    target_stratify = st.checkbox(
        "Stratify by target",
        value=target in df.columns,
        disabled=target not in df.columns,
        key="shap_stratify_target",
        help="Keeps event/non-event or target-class balance in SHAP background and explain samples.",
    )
    candidate_strata_columns = [
        str(column)
        for column in df.columns
        if str(column) != target
    ]
    default_strata_columns = [
        column for column in shap_default_stratify_columns(df, target, feature_names) if column in candidate_strata_columns
    ]
    stratify_columns = st.multiselect(
        "Additional stratify columns",
        options=candidate_strata_columns,
        default=default_strata_columns,
        key="shap_stratify_columns",
        help="Score band and target are always used when available. These columns add business segment balance.",
    )
    if st.button("Run SHAP analysis", width="stretch", type="primary"):
        try:
            with st.status("Running Kernel SHAP", expanded=True) as status:
                status.write(f"Sampling pool rows: {int(pool_rows):,}")
                pool, pool_scores = scored_shap_sampling_pool(
                    state["model"],
                    df,
                    feature_names,
                    positive_class=positive_class,
                    positive_index=state.get("positive_index"),
                    pool_rows=int(pool_rows),
                    random_state=int(seed),
                )
                strata = shap_strata_labels(
                    pool,
                    pool_scores,
                    target=target,
                    extra_columns=list(stratify_columns),
                    score_bins=DEFAULT_SHAP_SCORE_BINS,
                    include_target=bool(target_stratify),
                )
                background = stratified_sample_by_labels(
                    pool,
                    strata,
                    int(background_rows),
                    random_state=int(seed) + 1,
                )
                explain = stratified_sample_by_labels(
                    pool,
                    strata,
                    int(explain_rows),
                    random_state=int(seed) + 2,
                    exclude_index=background.index,
                )
                background_scores = pool_scores.reindex(background.index)
                explain_scores = pool_scores.reindex(explain.index)
                status.write(f"Background rows: {len(background):,}")
                status.write(f"Explain rows: {len(explain):,}")
                strata_column_text = "score_band"
                if target_stratify and target in pool.columns:
                    strata_column_text += f", {target}"
                if stratify_columns:
                    strata_column_text += f", {', '.join(stratify_columns)}"
                status.write(f"Strata columns: {strata_column_text}")
                result = kernel_shap_contributions(
                    state["model"],
                    background,
                    explain,
                    feature_names,
                    positive_class=positive_class,
                    positive_index=state.get("positive_index"),
                    nsamples=int(nsamples),
                )
                result["sampling"] = {
                    "mode": "score_band_stratified",
                    "pool_rows": int(len(pool)),
                    "score_bins": int(DEFAULT_SHAP_SCORE_BINS),
                    "target": target if target_stratify and target in pool.columns else None,
                    "stratify_columns": list(stratify_columns),
                    "background_rows": int(len(background)),
                    "explain_rows": int(len(explain)),
                    "background_score_mean": float(background_scores.mean()) if background_scores.notna().any() else None,
                    "explain_score_mean": float(explain_scores.mean()) if explain_scores.notna().any() else None,
                    "strata_count": int(strata.nunique(dropna=False)),
                }
                status.update(label="SHAP analysis complete", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"SHAP failed: {exc}")
        else:
            st.session_state[SHAP_RESULT_KEY] = result

    result = st.session_state.get(SHAP_RESULT_KEY)
    if not isinstance(result, dict):
        return
    if list(result.get("feature_names", [])) != feature_names:
        st.session_state.pop(SHAP_RESULT_KEY, None)
        st.info("Existing SHAP result was cleared because model input columns changed.")
        return
    sampling = result.get("sampling")
    if isinstance(sampling, dict):
        with st.expander("SHAP sampling", expanded=False):
            st.dataframe(
                pd.DataFrame([sampling]),
                hide_index=True,
                width="stretch",
            )
    importance = shap_global_importance(result)
    st.bar_chart(importance.set_index("feature")["mean_abs_shap"])
    st.dataframe(importance, hide_index=True, width="stretch")
    row_labels = [str(index) for index in result.get("row_index", [])]
    selected_row = st.selectbox("Local SHAP row", options=list(range(len(row_labels))), format_func=lambda i: row_labels[i])
    local = shap_local_contributions(result, int(selected_row))
    st.dataframe(local, hide_index=True, width="stretch")
    render_shap_plots(result)


def render_shap_plots(result: dict[str, Any]) -> None:
    shap = require_shap()
    values = result["values"]
    plot_data = shap_plot_data_frame(result)
    feature_names = list(result["feature_names"])
    top_display = st.number_input(
        "Plot top features",
        min_value=1,
        max_value=max(1, len(feature_names)),
        value=min(20, len(feature_names)),
        step=1,
        format="%d",
        key="shap_plot_top_features",
    )
    beeswarm_tab, violin_tab, interaction_tab = st.tabs(["Beeswarm", "Violin", "Interaction"])
    with beeswarm_tab:
        try:
            import matplotlib.pyplot as plt

            shap.plots.beeswarm(shap_explanation(result), max_display=int(top_display), show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close(plt.gcf())
        except Exception as exc:
            st.error(f"Beeswarm plot failed: {exc}")
    with violin_tab:
        try:
            import matplotlib.pyplot as plt

            shap.summary_plot(
                values,
                plot_data,
                feature_names=feature_names,
                plot_type="violin",
                max_display=int(top_display),
                show=False,
            )
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close(plt.gcf())
        except Exception as exc:
            st.error(f"Violin plot failed: {exc}")
    with interaction_tab:
        try:
            import matplotlib.pyplot as plt

            importance = shap_global_importance(result)
            default_feature = str(importance.iloc[0]["feature"]) if not importance.empty else feature_names[0]
            main_feature = st.selectbox(
                "SHAP feature",
                options=feature_names,
                index=feature_names.index(default_feature) if default_feature in feature_names else 0,
                key="shap_interaction_main_feature",
            )
            st.caption(
                "X-axis is the raw value of the selected feature. Y-axis is that same feature's SHAP "
                "contribution. Pair interaction is represented by color."
            )
            interaction_options = shap_interaction_color_options(feature_names, main_feature)
            interaction_key = "shap_interaction_color_feature"
            if st.session_state.get(interaction_key) not in interaction_options:
                st.session_state[interaction_key] = "auto"
            interaction_feature = st.selectbox(
                "Interaction color",
                options=interaction_options,
                key=interaction_key,
            )
            if len(interaction_options) == 1:
                st.info("Only one model feature is available, so a pairwise interaction color cannot be shown.")
            shap.dependence_plot(
                main_feature,
                values,
                plot_data,
                feature_names=feature_names,
                interaction_index=interaction_feature if len(interaction_options) > 1 else None,
                show=False,
            )
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close(plt.gcf())
        except Exception as exc:
            st.error(f"Interaction plot failed: {exc}")


def render_what_if_workspace(
    *,
    df: pd.DataFrame,
    features: list[str],
    target: str,
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

    lookup_options = ["ID value", "Row position"]
    lookup_mode = st.radio(
        "Customer lookup",
        options=lookup_options,
        index=0,
        horizontal=True,
        key="what_if_lookup_mode",
    )
    if "what_if_row_position" not in st.session_state:
        st.session_state["what_if_row_position"] = 0
    if lookup_mode == "ID value":
        id_columns = what_if_id_columns(df, target)
        if not id_columns:
            st.warning("No non-target ID column is available. Use row position lookup.")
            row_position = int(st.session_state.get("what_if_row_position", 0))
            row_position = min(max(row_position, 0), max(0, len(df) - 1))
        else:
            default_column = default_id_column(id_columns)
            id_column = st.selectbox(
                "ID column",
                options=id_columns,
                index=id_columns.index(default_column) if default_column in id_columns else 0,
                key="what_if_id_column",
            )
            id_value = st.text_input("ID value", key="what_if_id_value")
            if st.button("Load customer", width="stretch"):
                resolved_position = find_row_position_by_id_value(df, id_column, id_value)
                if resolved_position is None:
                    st.error(f"No row found for {id_column} = {id_value}.")
                else:
                    st.session_state["what_if_row_position"] = resolved_position
                    st.success(f"Loaded row position {resolved_position:,}.")
            row_position = int(st.session_state.get("what_if_row_position", 0))
            row_position = min(max(row_position, 0), max(0, len(df) - 1))
    else:
        row_position = st.number_input(
            "Row position",
            min_value=0,
            max_value=max(0, len(df) - 1),
            value=min(int(st.session_state.get("what_if_row_position", 0)), max(0, len(df) - 1)),
            step=1,
            format="%d",
            key="what_if_row_position",
        )
    actual_target = df.iloc[int(row_position)][target] if target in df.columns else None
    row_cols = st.columns(3)
    row_cols[0].metric("Source row", str(df.index[int(row_position)]))
    row_cols[1].metric("Actual target", str(actual_target))
    row_cols[2].metric("Model input columns", f"{len(feature_names):,}")
    base_row = prepare_model_frame(df.iloc[[int(row_position)]], feature_names)
    order_mode = st.selectbox(
        "Editable variable order",
        options=["Local sensitivity", "Model input order"],
        index=0,
        help=(
            "Local sensitivity scores one-at-a-time alternatives for this customer and places the "
            "variables with the largest absolute score impact first."
        ),
    )
    editor_features = list(feature_names)
    sensitivity_frame: pd.DataFrame | None = None
    if order_mode == "Local sensitivity":
        cache = st.session_state.setdefault(WHAT_IF_SENSITIVITY_CACHE_KEY, {})
        if not isinstance(cache, dict):
            st.session_state[WHAT_IF_SENSITIVITY_CACHE_KEY] = {}
            cache = st.session_state[WHAT_IF_SENSITIVITY_CACHE_KEY]
        cache_key = (
            data_key,
            id(state.get("model")),
            int(row_position),
            tuple(feature_names),
            str(df.index[int(row_position)]),
            int(len(df)),
        )
        try:
            if cache_key not in cache:
                with st.spinner("Ranking editable variables by local sensitivity..."):
                    cache[cache_key] = what_if_local_sensitivity(
                        model=state["model"],
                        df=df,
                        row_position=int(row_position),
                        feature_names=feature_names,
                        positive_class=state.get("positive_class"),
                        positive_index=state.get("positive_index"),
                    )
            sensitivity_frame = cache[cache_key].copy()
            editor_features = [
                feature for feature in sensitivity_frame["feature"].astype(str).tolist() if feature in feature_names
            ]
            editor_features.extend([feature for feature in feature_names if feature not in editor_features])
        except Exception as exc:
            st.warning(f"Local sensitivity ranking failed; using model input order. Detail: {exc}")
            sensitivity_frame = None
            editor_features = list(feature_names)

    if sensitivity_frame is not None:
        with st.expander("Local sensitivity ranking", expanded=False):
            st.caption(
                "Sensitivity is the largest absolute score delta found by replacing one variable at a time "
                "with representative values from the active data."
            )
            st.dataframe(
                sensitivity_frame.head(50),
                hide_index=True,
                width="stretch",
                column_config={
                    "sensitivity": st.column_config.NumberColumn(format="%.6f"),
                    "delta": st.column_config.NumberColumn(format="%+.6f"),
                    "base_score": st.column_config.NumberColumn(format="%.6f"),
                    "scenario_score": st.column_config.NumberColumn(format="%.6f"),
                },
            )

    editor_base_row = base_row.loc[:, editor_features]
    editor_order_signature = str(abs(hash(tuple(editor_features))))
    edited = st.data_editor(
        editor_base_row,
        hide_index=True,
        width="stretch",
        key=f"what_if_editor::{data_key}::{int(row_position)}::{editor_order_signature}",
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
