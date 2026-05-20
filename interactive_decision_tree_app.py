from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from interactive_decision_tree.session_store import (
    DATA_ID_QUERY_PARAM,
    default_session_dir,
    load_dataframe_session,
    normalize_data_id,
    save_dataframe_session,
    session_data_key,
)
from interactive_decision_tree.sql_source import DEFAULT_SQL_LIMIT, read_sql_dataframe
from interactive_decision_tree.woe_ui import WOE_PROJECTS_KEY, render_woe_workspace


TREE_SCHEMA_VERSION = 4
CHECKPOINT_SCHEMA_VERSION = 1
WORK_ID_QUERY_PARAM = "work_id"
CHECKPOINT_DIR = Path(__file__).with_name(".tree_checkpoints")
POSITIVE_CLASS_SESSION_KEY = "_interactive_tree_positive_class"
APPLIED_DATA_CONTEXT_KEY = "_interactive_tree_applied_data_context"
MIN_INFORMATION_GAIN_EPSILON = 1e-12
GRAPH_TOOLTIP_LIMIT = 900
AUTO_COMPUTE_CANDIDATE_ROWS = 100_000
AUTO_APPLY_DATA_SETUP_MAX_ROWS = 100_000
DEFAULT_DATA_SAMPLE_ROWS = 100_000
DEFAULT_STRATIFY_NUMERIC_BINS = 10
FEATURE_PROFILE_SAMPLE_ROWS = 10_000
FEATURE_FILTER_MAX_VISIBLE = 250
CANDIDATE_RANDOM_STATE = 20260514
CHECKPOINT_EMBED_MAX_ROWS = 50_000
CPU_COUNT = max(1, os.cpu_count() or 1)
DEFAULT_PARALLEL_WORKERS = max(1, min(8, CPU_COUNT))
DEFAULT_DEMO_ROWS = 5_000
DEFAULT_MAX_VALIDATION_GINI_GAP = 0.10
VALIDATION_CANDIDATE_LIMIT = 100
NODE_SUMMARY_CACHE_KEY = "_interactive_tree_node_summary_cache"
TREE_UI_METRIC_CACHE_KEY = "_interactive_tree_ui_metric_cache"
TARGET_META_CACHE_KEY = "_interactive_tree_target_meta_cache"


@dataclass(frozen=True)
class SplitCandidate:
    feature: str
    split_type: str
    value: Any
    parent_entropy: float
    weighted_entropy: float
    information_gain: float
    branch_count: int
    branch_labels: tuple[str, ...]
    branch_ns: tuple[int, ...]
    branch_entropies: tuple[float, ...]
    label: str
    missing_policy: str = "right"


@dataclass(frozen=True)
class LoadedDataSource:
    df: pd.DataFrame
    name: str | None
    data_key: str
    source: str
    data_id: str | None
    metadata: dict[str, Any]
    restored_upload: bool = False


def make_demo_data(n: int = DEFAULT_DEMO_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    age = rng.integers(21, 72, size=n).astype(float)
    income = rng.normal(52_000, 18_000, size=n).clip(12_000, 140_000)
    tenure = rng.integers(0, 120, size=n).astype(float)
    segment = rng.choice(
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
        size=n,
        p=[0.11, 0.1, 0.1, 0.09, 0.09, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06],
    ).astype(object)
    channel = rng.choice(
        ["branch", "mobile", "web", "call_center", "agent", "atm"],
        size=n,
        p=[0.18, 0.32, 0.24, 0.1, 0.09, 0.07],
    ).astype(object)
    region = rng.choice(
        [
            "marmara",
            "ege",
            "akdeniz",
            "ic_anadolu",
            "karadeniz",
            "dogu_anadolu",
            "guneydogu",
            "trakya",
            "kibris",
            "yurtdisi",
        ],
        size=n,
        p=[0.22, 0.12, 0.11, 0.14, 0.1, 0.08, 0.08, 0.06, 0.04, 0.05],
    ).astype(object)
    product = rng.choice(
        [
            "card",
            "loan",
            "mortgage",
            "overdraft",
            "deposit",
            "investment",
            "insurance",
            "leasing",
            "pos",
            "fx",
            "cash_management",
            "trade_finance",
            "factoring",
            "retail_bundle",
            "sme_bundle",
        ],
        size=n,
        p=[0.11, 0.1, 0.07, 0.08, 0.12, 0.07, 0.07, 0.04, 0.09, 0.05, 0.05, 0.04, 0.03, 0.04, 0.04],
    ).astype(object)

    score = (
        (income < 42_000).astype(int)
        + (tenure < 18).astype(int)
        + np.isin(segment, ["C", "D", "E", "F", "K"]).astype(int)
        + (channel == "mobile").astype(int)
        + np.isin(region, ["dogu_anadolu", "guneydogu", "yurtdisi"]).astype(int)
        + np.isin(product, ["overdraft", "pos", "factoring", "trade_finance"]).astype(int)
        - (age > 55).astype(int)
    )
    target = np.where(score >= 3, "high_risk", "low_risk")

    missing_income = rng.choice(n, size=max(1, int(n * 0.035)), replace=False)
    remaining = np.setdiff1d(np.arange(n), missing_income)
    special_income = rng.choice(remaining, size=max(1, int(n * 0.012)), replace=False)
    income[missing_income] = np.nan
    income[special_income] = -999.0

    missing_age = rng.choice(n, size=max(1, int(n * 0.025)), replace=False)
    age[missing_age] = np.nan

    missing_tenure = rng.choice(n, size=max(1, int(n * 0.02)), replace=False)
    tenure[missing_tenure] = np.nan

    segment_missing = rng.choice(n, size=max(1, int(n * 0.025)), replace=False)
    segment[segment_missing] = None

    channel_blank = rng.choice(n, size=max(1, int(n * 0.015)), replace=False)
    channel[channel_blank] = ""

    region_unknown = rng.choice(n, size=max(1, int(n * 0.012)), replace=False)
    region[region_unknown] = "UNKNOWN"

    product_no_info = rng.choice(n, size=max(1, int(n * 0.01)), replace=False)
    product[product_no_info] = "NO_INFO"

    return pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "tenure_months": tenure,
            "segment": segment,
            "channel": channel,
            "region": region,
            "product": product,
            "risk_flag": target,
        }
    )


def entropy(y: pd.Series) -> float:
    counts = y.value_counts(dropna=False)
    if counts.empty:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def session_cache(key: str) -> dict[Any, Any]:
    cache = st.session_state.setdefault(key, {})
    if not isinstance(cache, dict):
        st.session_state[key] = {}
        cache = st.session_state[key]
    return cache


def bounded_cache_set(cache: dict[Any, Any], key: Any, value: Any, max_items: int = 64) -> None:
    if len(cache) >= max_items and key not in cache:
        cache.pop(next(iter(cache)), None)
    cache[key] = value


def clear_ui_metric_cache() -> None:
    for key in (NODE_SUMMARY_CACHE_KEY, TREE_UI_METRIC_CACHE_KEY):
        st.session_state.pop(key, None)


def target_meta_cache_key(y: pd.Series) -> tuple[Any, ...]:
    return (id(y), str(y.name), int(len(y)), str(y.dtype))


def target_unique_values(y: pd.Series) -> list[Any]:
    cache = session_cache(TARGET_META_CACHE_KEY)
    key = ("unique", target_meta_cache_key(y))
    if key not in cache:
        bounded_cache_set(cache, key, list(y.dropna().unique()), max_items=128)
    return list(cache[key])


def infer_target_kind(y: pd.Series) -> str:
    cache = session_cache(TARGET_META_CACHE_KEY)
    key = ("kind", target_meta_cache_key(y))
    if key in cache:
        return str(cache[key])
    non_missing = y.dropna()
    unique_count = non_missing.nunique()
    if pd.api.types.is_numeric_dtype(non_missing) and unique_count > 10:
        kind = "regression"
    elif unique_count == 2:
        kind = "binary"
    else:
        kind = "classification"
    bounded_cache_set(cache, key, kind, max_items=128)
    return kind


def class_values_equal(left: Any, right: Any) -> bool:
    try:
        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    try:
        if left == right:
            return True
    except (TypeError, ValueError):
        pass
    return str(left) == str(right)


def class_option_index(classes: list[Any], selected: Any) -> int:
    for index, cls in enumerate(classes):
        if class_values_equal(cls, selected):
            return index
    return 0


def session_positive_class() -> Any:
    try:
        return st.session_state.get(POSITIVE_CLASS_SESSION_KEY)
    except Exception:
        return None


def choose_positive_class(y: pd.Series, preferred: Any = None, use_session_default: bool = True) -> Any:
    classes = target_unique_values(y)
    if not classes:
        return None

    explicit = preferred if preferred is not None else (
        session_positive_class() if use_session_default else None
    )
    if explicit is not None:
        for cls in classes:
            if class_values_equal(cls, explicit):
                return cls

    preferred_terms = ("bad", "default", "event", "fail", "high", "yes", "true", "1")
    for cls in classes:
        text = str(cls).lower()
        if any(term in text for term in preferred_terms):
            return cls

    try:
        return sorted(classes)[-1]
    except TypeError:
        return classes[-1]


def target_impurity(y: pd.Series, target_kind: str | None = None) -> float:
    if (target_kind or infer_target_kind(y)) == "regression":
        numeric = pd.to_numeric(y, errors="coerce").dropna()
        if numeric.empty:
            return 0.0
        return float(np.var(numeric))
    return entropy(y)


def impurity_label(y: pd.Series, target_kind: str | None = None) -> str:
    return "Variance" if (target_kind or infer_target_kind(y)) == "regression" else "Entropy"


def node_summary(df: pd.DataFrame, target: str, row_idx: list[int]) -> dict[str, Any]:
    cache = session_cache(NODE_SUMMARY_CACHE_KEY)
    cache_key = (id(df), str(target), id(row_idx), int(len(row_idx)))
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return dict(cached)
    target_kind = infer_target_kind(df[target])
    positive_class = choose_positive_class(df[target]) if target_kind == "binary" else None
    summary = target_series_summary(df.loc[row_idx, target], target_kind, positive_class)
    bounded_cache_set(cache, cache_key, summary, max_items=256)
    return dict(summary)


def target_series_summary(
    y: pd.Series,
    target_kind: str,
    positive_class: Any = None,
) -> dict[str, Any]:
    counts = y.value_counts(dropna=False)
    numeric = pd.to_numeric(y, errors="coerce") if target_kind == "regression" else None
    prediction = float(numeric.mean()) if target_kind == "regression" and numeric is not None else (
        counts.index[0] if not counts.empty else None
    )
    out = {
        "n": len(y),
        "entropy": entropy(y),
        "impurity": target_impurity(y, target_kind),
        "impurity_label": impurity_label(y, target_kind),
        "majority": prediction,
        "prediction": prediction,
        "class_counts": counts.to_dict(),
        "target_kind": target_kind,
        "positive_class": positive_class,
    }
    if target_kind == "binary" and positive_class is not None:
        out["default_rate"] = float((y == positive_class).mean())
        out["event_count"] = int((y == positive_class).sum())
    if target_kind == "regression" and numeric is not None:
        out["target_mean"] = float(numeric.mean())
        out["target_std"] = float(numeric.std(ddof=0))
    return out


def split_score_name(y: pd.Series) -> str:
    return "variance_gain" if infer_target_kind(y) == "regression" else "information_gain"


def arrow_safe_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for column in out.columns:
        if out[column].dtype == "object":
            out[column] = out[column].map(lambda x: "" if pd.isna(x) else str(x))
    return out


def numeric_thresholds(s: pd.Series, max_thresholds: int) -> list[float]:
    clean = pd.to_numeric(s, errors="coerce").dropna().sort_values().unique()
    if len(clean) <= 1:
        return []

    mids = (clean[:-1] + clean[1:]) / 2
    if len(mids) <= max_thresholds:
        return [float(x) for x in mids]

    quantile_positions = np.linspace(0, 1, max_thresholds + 2)[1:-1]
    qs = np.quantile(mids, quantile_positions)
    return sorted({float(x) for x in qs})


def numeric_bin_thresholds(s: pd.Series, bin_count: int) -> list[float]:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.nunique() < bin_count:
        return []

    quantile_positions = np.linspace(0, 1, bin_count + 1)[1:-1]
    thresholds = np.quantile(clean, quantile_positions)
    return sorted({float(x) for x in thresholds})


def score_branch_split(
    frame: pd.DataFrame,
    target: str,
    masks_and_labels: list[tuple[pd.Series, str]],
    min_leaf: int,
    target_kind: str | None = None,
) -> tuple[float, float, tuple[str, ...], tuple[int, ...], tuple[float, ...]] | None:
    y_parent = frame[target]
    parent_impurity = target_impurity(y_parent, target_kind)

    branch_labels: list[str] = []
    branch_ns: list[int] = []
    branch_impurities: list[float] = []
    weighted_impurity = 0.0

    for mask, label in masks_and_labels:
        child = frame[mask]
        child_n = len(child)
        if child_n < min_leaf:
            return None
        child_impurity = target_impurity(child[target], target_kind)
        branch_labels.append(label)
        branch_ns.append(child_n)
        branch_impurities.append(child_impurity)
        weighted_impurity += (child_n / len(frame)) * child_impurity

    if len(branch_ns) < 2 or sum(branch_ns) != len(frame):
        return None

    return (
        parent_impurity,
        weighted_impurity,
        tuple(branch_labels),
        tuple(branch_ns),
        tuple(branch_impurities),
    )


def numeric_missing_policies(numeric: pd.Series) -> list[str]:
    return ["right", "left", "separate"] if numeric.isna().any() else ["right"]


def numeric_masks_and_labels(
    numeric: pd.Series,
    thresholds: list[float],
    missing_policy: str = "right",
) -> list[tuple[pd.Series, str]]:
    thresholds = sorted({float(threshold) for threshold in thresholds})
    if not thresholds:
        return []

    masks_and_labels: list[tuple[pd.Series, str]] = []
    previous_threshold: float | None = None
    for threshold in thresholds:
        if previous_threshold is None:
            mask = numeric <= threshold
            branch_label = f"<= {threshold:.6g}"
        else:
            mask = (numeric > previous_threshold) & (numeric <= threshold)
            branch_label = f"> {previous_threshold:.6g} and <= {threshold:.6g}"
        masks_and_labels.append((mask, branch_label))
        previous_threshold = threshold

    masks_and_labels.append((numeric > thresholds[-1], f"> {thresholds[-1]:.6g}"))
    missing = numeric.isna()
    if missing.any():
        if missing_policy == "left":
            mask, label = masks_and_labels[0]
            masks_and_labels[0] = (mask | missing, f"{label} or missing")
        elif missing_policy == "separate":
            masks_and_labels.append((missing, "missing"))
        else:
            mask, label = masks_and_labels[-1]
            masks_and_labels[-1] = (mask | missing, f"{label} or missing")
    return masks_and_labels


def numeric_split_label(
    feature: str,
    thresholds: list[float],
    missing_policy: str = "right",
    has_missing: bool = False,
) -> str:
    thresholds = sorted({float(threshold) for threshold in thresholds})
    if len(thresholds) == 1:
        base = f"{feature} <= {thresholds[0]:.6g}"
    else:
        base = f"{feature} manual bins: {', '.join(f'{x:.6g}' for x in thresholds)}"
    if not has_missing:
        return base
    if missing_policy == "left":
        return f"{base} (missing -> first/lower branch)"
    if missing_policy == "separate":
        return f"{base} (missing separate)"
    return f"{base} (missing -> last/upper branch)"


def score_split(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    split_type: str,
    value: Any,
    min_leaf: int,
    target_kind: str | None = None,
    missing_policy: str = "right",
) -> SplitCandidate | None:
    frame = df.loc[row_idx, [feature, target]]

    if split_type == "numeric_le":
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        masks_and_labels = numeric_masks_and_labels(numeric, [float(value)], missing_policy)
        label = numeric_split_label(feature, [float(value)], missing_policy, has_missing=bool(numeric.isna().any()))
    elif split_type == "category_eq":
        values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__")
        masks_and_labels = [
            (values == value, f"== {value}"),
            (values != value, f"!= {value}"),
        ]
        label = f"{feature} == {value}"
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    scored = score_branch_split(frame, target, masks_and_labels, min_leaf, target_kind or infer_target_kind(df[target]))
    if scored is None:
        return None
    parent_entropy, weighted_entropy, branch_labels, branch_ns, branch_entropies = scored

    return SplitCandidate(
        feature=feature,
        split_type=split_type,
        value=value,
        parent_entropy=parent_entropy,
        weighted_entropy=weighted_entropy,
        information_gain=parent_entropy - weighted_entropy,
        branch_count=len(branch_labels),
        branch_labels=branch_labels,
        branch_ns=branch_ns,
        branch_entropies=branch_entropies,
        label=label,
        missing_policy=missing_policy if split_type == "numeric_le" else "category_level",
    )


def score_numeric_multiway_split(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    bin_count: int,
    min_leaf: int,
    target_kind: str | None = None,
    missing_policy: str = "right",
) -> SplitCandidate | None:
    frame = df.loc[row_idx, [feature, target]]
    numeric = pd.to_numeric(frame[feature], errors="coerce")
    thresholds = numeric_bin_thresholds(numeric, bin_count)
    if len(thresholds) != bin_count - 1:
        return None

    masks_and_labels = numeric_masks_and_labels(numeric, thresholds, missing_policy)

    scored = score_branch_split(frame, target, masks_and_labels, min_leaf, target_kind or infer_target_kind(df[target]))
    if scored is None:
        return None
    parent_entropy, weighted_entropy, branch_labels, branch_ns, branch_entropies = scored

    return SplitCandidate(
        feature=feature,
        split_type="numeric_bins",
        value=tuple(thresholds),
        parent_entropy=parent_entropy,
        weighted_entropy=weighted_entropy,
        information_gain=parent_entropy - weighted_entropy,
        branch_count=len(branch_labels),
        branch_labels=branch_labels,
        branch_ns=branch_ns,
        branch_entropies=branch_entropies,
        label=numeric_split_label(feature, thresholds, missing_policy, has_missing=bool(numeric.isna().any())),
        missing_policy=missing_policy,
    )


def score_category_multiway_split(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    max_categories: int,
    min_leaf: int,
) -> SplitCandidate | None:
    frame = df.loc[row_idx, [feature, target]]
    values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__")
    counts = values.value_counts()
    if len(counts) < 2:
        return None

    kept_values = counts.head(max_categories).index.tolist()
    masks_and_labels: list[tuple[pd.Series, str]] = [
        (values == value, f"= {value}") for value in kept_values
    ]
    if len(kept_values) < len(counts):
        masks_and_labels.append((~values.isin(kept_values), "other"))

    scored = score_branch_split(frame, target, masks_and_labels, min_leaf, infer_target_kind(df[target]))
    if scored is None:
        return None
    parent_entropy, weighted_entropy, branch_labels, branch_ns, branch_entropies = scored

    return SplitCandidate(
        feature=feature,
        split_type="category_multi",
        value=tuple(kept_values),
        parent_entropy=parent_entropy,
        weighted_entropy=weighted_entropy,
        information_gain=parent_entropy - weighted_entropy,
        branch_count=len(branch_labels),
        branch_labels=branch_labels,
        branch_ns=branch_ns,
        branch_entropies=branch_entropies,
        label=f"{feature} multiway categories",
    )


def category_profile_order(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    target_kind: str | None = None,
    positive_class: Any = None,
) -> list[Any]:
    frame = df.loc[row_idx, [feature, target]]
    values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__")
    target_kind = target_kind or infer_target_kind(df[target])

    if target_kind == "binary":
        positive_class = positive_class if positive_class is not None else choose_positive_class(df[target])
        profile = (
            pd.DataFrame({"value": values, "event": frame[target] == positive_class})
            .groupby("value", dropna=False)
            .agg(rate=("event", "mean"), n=("event", "size"))
            .sort_values(["rate", "n"], ascending=[True, False])
        )
        return profile.index.tolist()

    if target_kind == "regression":
        numeric_target = pd.to_numeric(frame[target], errors="coerce")
        profile = (
            pd.DataFrame({"value": values, "target": numeric_target})
            .groupby("value", dropna=False)
            .agg(mean=("target", "mean"), n=("target", "size"))
            .sort_values(["mean", "n"], ascending=[True, False])
        )
        return profile.index.tolist()

    profile = (
        pd.DataFrame({"value": values, "target": frame[target]})
        .groupby("value", dropna=False)
        .agg(impurity=("target", target_impurity), n=("target", "size"))
        .sort_values(["impurity", "n"], ascending=[True, False])
    )
    return profile.index.tolist()


def contiguous_groups(values: list[Any], group_count: int) -> list[list[Any]]:
    if group_count < 2 or len(values) < group_count:
        return []
    boundaries = np.linspace(0, len(values), group_count + 1).round().astype(int)
    groups: list[list[Any]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group = values[start:end]
        if not group:
            return []
        groups.append(group)
    return groups


def score_category_profile_groups(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    max_groups: int,
    min_leaf: int,
    target_kind: str | None = None,
    positive_class: Any = None,
) -> list[SplitCandidate]:
    frame = df.loc[row_idx, [feature, target]]
    values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__")
    resolved_target_kind = target_kind or infer_target_kind(df[target])
    ordered_values = category_profile_order(
        df,
        target,
        row_idx,
        feature,
        target_kind=resolved_target_kind,
        positive_class=positive_class,
    )
    if len(ordered_values) < 2:
        return []

    candidates: list[SplitCandidate] = []
    for group_count in range(2, min(max_groups, len(ordered_values)) + 1):
        groups = contiguous_groups(ordered_values, group_count)
        if not groups:
            continue
        masks_and_labels: list[tuple[pd.Series, str]] = []
        for group in groups:
            preview = ", ".join(str(x) for x in group[:4])
            suffix = "" if len(group) <= 4 else f", +{len(group) - 4}"
            masks_and_labels.append((values.isin(group), f"{{{preview}{suffix}}}"))

        scored = score_branch_split(frame, target, masks_and_labels, min_leaf, resolved_target_kind)
        if scored is None:
            continue
        parent_impurity, weighted_impurity, branch_labels, branch_ns, branch_entropies = scored
        candidates.append(
            SplitCandidate(
                feature=feature,
                split_type="category_profile_groups",
                value=tuple(tuple(group) for group in groups),
                parent_entropy=parent_impurity,
                weighted_entropy=weighted_impurity,
                information_gain=parent_impurity - weighted_impurity,
                branch_count=len(branch_labels),
                branch_labels=branch_labels,
                branch_ns=branch_ns,
                branch_entropies=branch_entropies,
                label=f"{feature} target-profile groups ({group_count})",
            )
        )
    return candidates


def score_numeric_manual_bins(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    thresholds: list[float],
    min_leaf: int,
    missing_policy: str = "right",
) -> SplitCandidate | None:
    thresholds = sorted({float(x) for x in thresholds})
    if not thresholds:
        return None

    frame = df.loc[row_idx, [feature, target]]
    numeric = pd.to_numeric(frame[feature], errors="coerce")
    masks_and_labels = numeric_masks_and_labels(numeric, thresholds, missing_policy)

    scored = score_branch_split(frame, target, masks_and_labels, min_leaf, infer_target_kind(df[target]))
    if scored is None:
        return None
    parent_impurity, weighted_impurity, branch_labels, branch_ns, branch_entropies = scored
    split_type = "numeric_le" if len(thresholds) == 1 else "numeric_manual_bins"
    label = numeric_split_label(feature, thresholds, missing_policy, has_missing=bool(numeric.isna().any()))

    return SplitCandidate(
        feature=feature,
        split_type=split_type,
        value=thresholds[0] if len(thresholds) == 1 else tuple(thresholds),
        parent_entropy=parent_impurity,
        weighted_entropy=weighted_impurity,
        information_gain=parent_impurity - weighted_impurity,
        branch_count=len(branch_labels),
        branch_labels=branch_labels,
        branch_ns=branch_ns,
        branch_entropies=branch_entropies,
        label=label,
        missing_policy=missing_policy,
    )


def score_category_group_split(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    selected_values: list[Any],
    min_leaf: int,
) -> SplitCandidate | None:
    if not selected_values:
        return None

    frame = df.loc[row_idx, [feature, target]]
    values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__")
    selected = set(selected_values)
    masks_and_labels = [
        (values.isin(selected), f"in {{{', '.join(map(str, selected_values))}}}"),
        (~values.isin(selected), "other"),
    ]
    scored = score_branch_split(frame, target, masks_and_labels, min_leaf, infer_target_kind(df[target]))
    if scored is None:
        return None
    parent_impurity, weighted_impurity, branch_labels, branch_ns, branch_entropies = scored

    return SplitCandidate(
        feature=feature,
        split_type="category_group",
        value=tuple(selected_values),
        parent_entropy=parent_impurity,
        weighted_entropy=weighted_impurity,
        information_gain=parent_impurity - weighted_impurity,
        branch_count=len(branch_labels),
        branch_labels=branch_labels,
        branch_ns=branch_ns,
        branch_entropies=branch_entropies,
        label=f"{feature} in {{{', '.join(map(str, selected_values))}}}",
    )


def score_category_manual_groups(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    groups: list[list[Any]],
    min_leaf: int,
) -> SplitCandidate | None:
    cleaned_groups = [[value for value in group if value is not None] for group in groups]
    cleaned_groups = [group for group in cleaned_groups if group]
    if len(cleaned_groups) < 2:
        return None

    frame = df.loc[row_idx, [feature, target]]
    values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__")
    assigned: set[Any] = set()
    masks_and_labels: list[tuple[pd.Series, str]] = []

    for group in cleaned_groups:
        deduped_group = []
        for value in group:
            if value not in assigned:
                deduped_group.append(value)
                assigned.add(value)
        if not deduped_group:
            continue
        masks_and_labels.append(
            (
                values.isin(deduped_group),
                "{" + ", ".join(map(str, deduped_group)) + "}",
            )
        )

    remaining_mask = ~values.isin(assigned)
    if remaining_mask.any():
        masks_and_labels.append((remaining_mask, "other"))

    scored = score_branch_split(frame, target, masks_and_labels, min_leaf, infer_target_kind(df[target]))
    if scored is None:
        return None
    parent_impurity, weighted_impurity, branch_labels, branch_ns, branch_entropies = scored

    value_groups: list[tuple[Any, ...]] = [tuple(group) for group in cleaned_groups]
    if remaining_mask.any():
        value_groups.append(tuple(values[remaining_mask].drop_duplicates().tolist()))

    return SplitCandidate(
        feature=feature,
        split_type="category_manual_groups",
        value=tuple(value_groups),
        parent_entropy=parent_impurity,
        weighted_entropy=weighted_impurity,
        information_gain=parent_impurity - weighted_impurity,
        branch_count=len(branch_labels),
        branch_labels=branch_labels,
        branch_ns=branch_ns,
        branch_entropies=branch_entropies,
        label=f"{feature} manual groups",
    )


def parse_category_group_text(text: str, levels: list[Any]) -> list[list[Any]]:
    by_text = {str(level).strip(): level for level in levels}
    groups: list[list[Any]] = []
    for raw_group in text.split("|"):
        group: list[Any] = []
        for raw_value in raw_group.split(","):
            value_text = raw_value.strip()
            if value_text:
                group.append(by_text.get(value_text, value_text))
        if group:
            groups.append(group)
    return groups


def category_group_state_key(data_key: str, target: str, node_id: int, feature: str) -> str:
    return f"category_groups::{data_key}::{target}::{node_id}::{feature}"


def normalize_category_groups(groups: list[list[str]], level_texts: list[str]) -> list[list[str]]:
    valid = set(level_texts)
    seen: set[str] = set()
    cleaned: list[list[str]] = []

    for group in groups:
        cleaned_group: list[str] = []
        for value in group:
            value_text = str(value)
            if value_text in valid and value_text not in seen:
                cleaned_group.append(value_text)
                seen.add(value_text)
        if cleaned_group:
            cleaned.append(cleaned_group)

    missing = [value for value in level_texts if value not in seen]
    if missing:
        if cleaned:
            cleaned[0].extend(missing)
        else:
            cleaned = [missing]

    return cleaned


def profile_group_texts(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    max_groups: int,
) -> list[list[str]]:
    ordered_values = category_profile_order(df, target, row_idx, feature)
    if len(ordered_values) < 2:
        return [[str(value) for value in ordered_values]]
    group_count = min(max_groups, len(ordered_values))
    return [[str(value) for value in group] for group in contiguous_groups(ordered_values, group_count)]


def category_level_rows(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
) -> list[dict[str, Any]]:
    frame = df.loc[row_idx, [feature, target]]
    values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__")
    target_kind = infer_target_kind(df[target])
    positive_class = choose_positive_class(df[target]) if target_kind == "binary" else None
    rows: list[dict[str, Any]] = []

    for value in category_profile_order(df, target, row_idx, feature):
        mask = values == value
        y_value = frame.loc[mask, target]
        row = {
            "value": str(value),
            "n": int(mask.sum()),
            "impurity": target_impurity(y_value, target_kind),
        }
        if target_kind == "binary":
            row["default_rate"] = float((y_value == positive_class).mean())
            row["event_count"] = int((y_value == positive_class).sum())
        elif target_kind == "regression":
            numeric = pd.to_numeric(y_value, errors="coerce")
            row["target_mean"] = float(numeric.mean())
            row["target_std"] = float(numeric.std(ddof=0))
        else:
            row["majority"] = y_value.value_counts(dropna=False).index[0]
        rows.append(row)
    return rows


def category_group_rows(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    groups: list[list[str]],
    text_to_value: dict[str, Any],
) -> list[dict[str, Any]]:
    frame = df.loc[row_idx, [feature, target]]
    values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__")
    target_kind = infer_target_kind(df[target])
    positive_class = choose_positive_class(df[target]) if target_kind == "binary" else None
    rows: list[dict[str, Any]] = []

    for i, group in enumerate(groups, start=1):
        actual_values = [text_to_value[value] for value in group if value in text_to_value]
        mask = values.isin(actual_values)
        y_group = frame.loc[mask, target]
        row = {
            "group": f"G{i}",
            "values": ", ".join(group),
            "value_count": len(group),
            "n": int(mask.sum()),
            "impurity": target_impurity(y_group, target_kind),
        }
        if target_kind == "binary":
            row["default_rate"] = float((y_group == positive_class).mean()) if len(y_group) else np.nan
            row["event_count"] = int((y_group == positive_class).sum()) if len(y_group) else 0
        elif target_kind == "regression":
            numeric = pd.to_numeric(y_group, errors="coerce")
            row["target_mean"] = float(numeric.mean()) if len(numeric) else np.nan
            row["target_std"] = float(numeric.std(ddof=0)) if len(numeric) else np.nan
        else:
            row["majority"] = y_group.value_counts(dropna=False).index[0] if len(y_group) else ""
        rows.append(row)
    return rows


def candidate_splits_for_feature(
    df: pd.DataFrame,
    target: str,
    feature: str,
    row_idx: list[int],
    min_leaf: int,
    max_thresholds: int,
    max_categories: int,
    max_numeric_bins: int,
    max_category_groups: int,
    target_kind: str,
    positive_class: Any = None,
) -> list[SplitCandidate]:
    candidates: list[SplitCandidate] = []
    s = df.loc[row_idx, feature]

    if pd.api.types.is_numeric_dtype(s):
        policies = numeric_missing_policies(pd.to_numeric(s, errors="coerce"))
        for threshold in numeric_thresholds(s, max_thresholds):
            for missing_policy in policies:
                candidate = score_split(
                    df=df,
                    target=target,
                    row_idx=row_idx,
                    feature=feature,
                    split_type="numeric_le",
                    value=threshold,
                    min_leaf=min_leaf,
                    target_kind=target_kind,
                    missing_policy=missing_policy,
                )
                if candidate is not None:
                    candidates.append(candidate)
        for bin_count in range(3, max_numeric_bins + 1):
            for missing_policy in policies:
                candidate = score_numeric_multiway_split(
                    df=df,
                    target=target,
                    row_idx=row_idx,
                    feature=feature,
                    bin_count=bin_count,
                    min_leaf=min_leaf,
                    target_kind=target_kind,
                    missing_policy=missing_policy,
                )
                if candidate is not None:
                    candidates.append(candidate)
    else:
        values = (
            s.astype("object")
            .where(s.notna(), "__MISSING__")
            .value_counts()
            .head(max_categories)
            .index.tolist()
        )
        for value in values:
            candidate = score_split(
                df=df,
                target=target,
                row_idx=row_idx,
                feature=feature,
                split_type="category_eq",
                value=value,
                min_leaf=min_leaf,
                target_kind=target_kind,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates.extend(
            score_category_profile_groups(
                df=df,
                target=target,
                row_idx=row_idx,
                feature=feature,
                max_groups=max_category_groups,
                min_leaf=min_leaf,
                target_kind=target_kind,
                positive_class=positive_class,
            )
        )

    return candidates


def candidate_splits(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    row_idx: list[int],
    min_leaf: int,
    max_thresholds: int,
    max_categories: int,
    max_numeric_bins: int,
    max_category_groups: int,
    parallel_workers: int = 1,
) -> list[SplitCandidate]:
    candidates: list[SplitCandidate] = []
    if not features:
        return candidates

    target_kind = infer_target_kind(df[target])
    positive_class = choose_positive_class(df[target]) if target_kind == "binary" else None
    try:
        worker_count = int(parallel_workers)
    except (TypeError, ValueError):
        worker_count = 1
    worker_count = max(1, min(worker_count, len(features)))

    def score_feature(feature: str) -> list[SplitCandidate]:
        return candidate_splits_for_feature(
            df=df,
            target=target,
            feature=feature,
            row_idx=row_idx,
            min_leaf=min_leaf,
            max_thresholds=max_thresholds,
            max_categories=max_categories,
            max_numeric_bins=max_numeric_bins,
            max_category_groups=max_category_groups,
            target_kind=target_kind,
            positive_class=positive_class,
        )

    if worker_count == 1:
        for feature in features:
            candidates.extend(score_feature(feature))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for feature_candidates in executor.map(score_feature, features):
                candidates.extend(feature_candidates)

    feature_rank = {feature: i for i, feature in enumerate(features)}
    return sorted(candidates, key=lambda x: (-x.information_gain, feature_rank.get(x.feature, 9999), x.label))


def normalize_stratify_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if not columns:
        return []
    seen: set[str] = set()
    valid_columns: list[str] = []
    for column in columns:
        column_name = str(column)
        if column_name in df.columns and column_name not in seen:
            valid_columns.append(column_name)
            seen.add(column_name)
    return valid_columns


def stratify_label_series(
    df: pd.DataFrame,
    row_idx: list[int],
    columns: list[str],
    numeric_bins: int = DEFAULT_STRATIFY_NUMERIC_BINS,
) -> pd.Series | None:
    stratify_columns = normalize_stratify_columns(df, columns)
    if not stratify_columns:
        return None

    labels: list[pd.Series] = []
    frame = df.loc[row_idx, stratify_columns]
    bin_count = max(2, int(numeric_bins))
    for column in stratify_columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            unique_count = int(numeric.dropna().nunique())
            if unique_count > 1:
                try:
                    binned = pd.qcut(
                        numeric,
                        q=min(bin_count, unique_count),
                        duplicates="drop",
                    )
                    label = binned.astype("object").where(binned.notna(), "__MISSING__").astype(str)
                except ValueError:
                    label = numeric.astype("object").where(numeric.notna(), "__MISSING__").astype(str)
            else:
                label = numeric.astype("object").where(numeric.notna(), "__MISSING__").astype(str)
        else:
            label = series.astype("object").where(series.notna(), "__MISSING__").astype(str)
        labels.append(column + "=" + label)

    combined = labels[0]
    for label in labels[1:]:
        combined = combined + "|" + label
    return combined


def stratified_sample_indices(
    df: pd.DataFrame,
    row_idx: list[int],
    max_rows: int,
    random_state: int = CANDIDATE_RANDOM_STATE,
    stratify_columns: list[str] | None = None,
    stratify_numeric_bins: int = DEFAULT_STRATIFY_NUMERIC_BINS,
) -> list[int]:
    if max_rows <= 0 or len(row_idx) <= max_rows:
        return row_idx
    labels = stratify_label_series(
        df,
        row_idx,
        normalize_stratify_columns(df, stratify_columns),
        numeric_bins=stratify_numeric_bins,
    )
    if labels is not None:
        counts = labels.value_counts(dropna=False)
        raw_take = counts / counts.sum() * max_rows
        take = np.floor(raw_take).astype(int)
        for value in take.index:
            if take.sum() >= max_rows:
                break
            if counts[value] > 0 and take[value] == 0:
                take[value] = 1
        while take.sum() < max_rows:
            available = counts[counts > take]
            if available.empty:
                break
            remainders = (raw_take - take).reindex(available.index).sort_values(ascending=False)
            take[remainders.index[0]] += 1
        while take.sum() > max_rows:
            removable = take[take > 1]
            if removable.empty:
                removable = take[take > 0]
            remainders = (raw_take - take).reindex(removable.index).sort_values()
            take[remainders.index[0]] -= 1

        sampled_indices: list[int] = []
        for value, n in take.items():
            if int(n) <= 0:
                continue
            value_idx = labels[labels == value].index
            sampled = value_idx.to_series(index=value_idx).sample(
                n=min(int(n), len(value_idx)),
                random_state=random_state,
            )
            sampled_indices.extend(sampled.tolist())
        return sorted(sampled_indices)

    index = pd.Index(row_idx)
    sampled = index.to_series(index=index).sample(n=max_rows, random_state=random_state).sort_index()
    return sampled.tolist()


def analysis_row_idx(
    row_idx: list[int],
    max_rows: int,
    random_state: int = CANDIDATE_RANDOM_STATE,
    df: pd.DataFrame | None = None,
    target: str | None = None,
    stratify_columns: list[str] | None = None,
    stratify_numeric_bins: int = DEFAULT_STRATIFY_NUMERIC_BINS,
) -> list[int]:
    if df is not None:
        columns = normalize_stratify_columns(df, stratify_columns)
        if not columns and target is not None:
            columns = normalize_stratify_columns(df, [target])
        if columns:
            return stratified_sample_indices(
                df=df,
                row_idx=row_idx,
                max_rows=max_rows,
                random_state=random_state,
                stratify_columns=columns,
                stratify_numeric_bins=stratify_numeric_bins,
            )

    if max_rows <= 0 or len(row_idx) <= max_rows:
        return row_idx
    index = pd.Index(row_idx)
    sampled = index.to_series(index=index).sample(n=max_rows, random_state=random_state).sort_index()
    return sampled.tolist()


def train_test_split_indices(
    df: pd.DataFrame,
    target: str,
    test_fraction: float,
    random_state: int = CANDIDATE_RANDOM_STATE,
    stratify: bool = True,
    stratify_columns: list[str] | None = None,
    stratify_numeric_bins: int = DEFAULT_STRATIFY_NUMERIC_BINS,
) -> tuple[list[int], list[int]]:
    if len(df) < 2:
        return df.index.tolist(), []
    bounded_fraction = min(max(float(test_fraction), 0.01), 0.9)
    test_rows = int(round(len(df) * bounded_fraction))
    test_rows = max(1, min(len(df) - 1, test_rows))
    columns = normalize_stratify_columns(df, stratify_columns)
    if stratify and not columns and target in df.columns and infer_target_kind(df[target]) != "regression":
        columns = [target]
    if stratify and columns:
        test_idx = analysis_row_idx(
            df.index.tolist(),
            max_rows=test_rows,
            random_state=random_state,
            df=df,
            stratify_columns=columns,
            stratify_numeric_bins=stratify_numeric_bins,
        )
    else:
        index = pd.Index(df.index)
        test_idx = index.to_series(index=index).sample(n=test_rows, random_state=random_state).sort_index().tolist()
    test_set = set(test_idx)
    train_idx = [idx for idx in df.index.tolist() if idx not in test_set]
    return train_idx, test_idx


def validate_test_dataframe(test_df: pd.DataFrame, target: str, features: list[str]) -> list[str]:
    required_columns = [target] + features
    return [column for column in required_columns if column not in test_df.columns]


def candidate_cache_key(
    data_key: str,
    target: str,
    node_id: int,
    features: list[str],
    row_count: int,
    parameters: dict[str, Any],
    max_rows: int,
) -> str:
    payload = {
        "schema": TREE_SCHEMA_VERSION,
        "data_key": data_key,
        "target": target,
        "node_id": node_id,
        "features": list(features),
        "row_count": int(row_count),
        "parameters": json_safe(parameters),
        "max_rows": int(max_rows),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def candidate_state_key(data_key: str, target: str, node_id: int) -> str:
    return f"candidate_cache::{data_key}::{target}::{node_id}"


def clear_candidate_cache(data_key: str | None = None, target: str | None = None, node_id: int | None = None) -> None:
    prefix = "candidate_cache::"
    for key in list(st.session_state.keys()):
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        if data_key is not None and f"::{data_key}::" not in key:
            continue
        if target is not None and f"::{target}::" not in key:
            continue
        if node_id is not None and not key.endswith(f"::{node_id}"):
            continue
        del st.session_state[key]


def get_cached_candidates(data_key: str, target: str, node_id: int, cache_key: str) -> list[SplitCandidate] | None:
    payload = st.session_state.get(candidate_state_key(data_key, target, node_id))
    if not isinstance(payload, dict) or payload.get("cache_key") != cache_key:
        return None
    candidates = payload.get("candidates")
    return candidates if isinstance(candidates, list) else None


def store_cached_candidates(
    data_key: str,
    target: str,
    node_id: int,
    cache_key: str,
    candidates: list[SplitCandidate],
    analyzed_rows: int,
    full_rows: int,
) -> None:
    st.session_state[candidate_state_key(data_key, target, node_id)] = {
        "cache_key": cache_key,
        "candidates": candidates,
        "analyzed_rows": int(analyzed_rows),
        "full_rows": int(full_rows),
    }


def cached_candidate_meta(data_key: str, target: str, node_id: int) -> dict[str, Any]:
    payload = st.session_state.get(candidate_state_key(data_key, target, node_id))
    return payload if isinstance(payload, dict) else {}


def split_ranking_scope_caption(
    candidate_feature_count: int,
    total_feature_count: int,
    selected_node_rows: int,
    active_train_rows: int,
) -> str:
    if selected_node_rows == active_train_rows:
        row_scope = f"all {active_train_rows:,} active train row(s) in the selected node"
    else:
        row_scope = (
            f"{selected_node_rows:,} row(s) in the selected leaf "
            f"out of {active_train_rows:,} active train row(s)"
        )
    return (
        f"Split search will rank the first {candidate_feature_count:,} of {total_feature_count:,} "
        f"active split variable(s) on {row_scope}. Select or reset the root node to rank on the "
        "full active train data. Use Data Setup > Sample to reduce the active train data before building the tree."
    )


def cached_ranking_ready_message(candidate_count: int, analyzed_rows: int, selected_node_rows: int, active_train_rows: int) -> str:
    if selected_node_rows == active_train_rows:
        row_scope = f"{analyzed_rows:,} active train row(s)"
    else:
        row_scope = (
            f"{analyzed_rows:,} selected-leaf row(s) "
            f"out of {active_train_rows:,} active train row(s)"
        )
    return f"Cached ranking ready: {candidate_count:,} candidate(s), {row_scope}."


def sync_feature_order(selected_features: list[str], data_key: str, target: str) -> list[str]:
    order_key = (data_key, target)
    if st.session_state.get("feature_order_key") != order_key:
        st.session_state.feature_order_key = order_key
        st.session_state.feature_order = selected_features.copy()
        return st.session_state.feature_order

    old_order = st.session_state.get("feature_order", [])
    st.session_state.feature_order = [f for f in old_order if f in selected_features] + [
        f for f in selected_features if f not in old_order
    ]
    return st.session_state.feature_order


def move_feature(feature: str, direction: int) -> None:
    order = st.session_state.feature_order
    index = order.index(feature)
    new_index = index + direction
    if new_index < 0 or new_index >= len(order):
        return
    order[index], order[new_index] = order[new_index], order[index]
    st.session_state.feature_order = order


def get_node_children(node: dict[str, Any]) -> list[dict[str, Any]]:
    if "children" in node and node["children"]:
        return node["children"]

    children: list[dict[str, Any]] = []
    if node.get("left") is not None:
        children.append({"id": node["left"], "label": "YES"})
    if node.get("right") is not None:
        children.append({"id": node["right"], "label": "NO"})
    return children


def init_tree(df: pd.DataFrame) -> None:
    clear_ui_metric_cache()
    st.session_state.tree = {
        0: {
            "id": 0,
            "depth": 0,
            "path": "root",
            "row_idx": df.index.tolist(),
            "split": None,
            "children": [],
            "left": None,
            "right": None,
        }
    }
    st.session_state.next_node_id = 1
    st.session_state.current_node_id = 0
    st.session_state.split_history = []
    st.session_state.split_action_history = []
    st.session_state.auto_tree_message = ""


def split_branch_indices(
    df: pd.DataFrame,
    row_idx: list[int],
    candidate: SplitCandidate,
) -> list[tuple[str, list[int]]]:
    frame = df.loc[row_idx, [candidate.feature]]

    if candidate.split_type == "numeric_le":
        numeric = pd.to_numeric(frame[candidate.feature], errors="coerce")
        masks_and_labels = numeric_masks_and_labels(numeric, [float(candidate.value)], candidate.missing_policy)
        return [
            (label, frame[mask].index.tolist())
            for label, (mask, _) in zip(candidate.branch_labels, masks_and_labels)
        ]
    elif candidate.split_type == "category_eq":
        values = frame[candidate.feature].astype("object").where(
            frame[candidate.feature].notna(), "__MISSING__"
        )
        return [
            (candidate.branch_labels[0], frame[values == candidate.value].index.tolist()),
            (candidate.branch_labels[1], frame[values != candidate.value].index.tolist()),
        ]
    elif candidate.split_type in ("numeric_bins", "numeric_manual_bins"):
        numeric = pd.to_numeric(frame[candidate.feature], errors="coerce")
        thresholds = list(candidate.value)
        masks_and_labels = numeric_masks_and_labels(numeric, thresholds, candidate.missing_policy)
        return [
            (label, frame[mask].index.tolist())
            for label, (mask, _) in zip(candidate.branch_labels, masks_and_labels)
        ]
    elif candidate.split_type == "category_multi":
        values = frame[candidate.feature].astype("object").where(
            frame[candidate.feature].notna(), "__MISSING__"
        )
        kept_values = list(candidate.value)
        branches = []
        for i, value in enumerate(kept_values):
            branches.append((candidate.branch_labels[i], frame[values == value].index.tolist()))
        if len(candidate.branch_labels) > len(kept_values):
            branches.append((candidate.branch_labels[-1], frame[~values.isin(kept_values)].index.tolist()))
        return branches
    elif candidate.split_type == "category_group":
        values = frame[candidate.feature].astype("object").where(
            frame[candidate.feature].notna(), "__MISSING__"
        )
        selected = set(candidate.value)
        return [
            (candidate.branch_labels[0], frame[values.isin(selected)].index.tolist()),
            (candidate.branch_labels[1], frame[~values.isin(selected)].index.tolist()),
        ]
    elif candidate.split_type == "category_profile_groups":
        values = frame[candidate.feature].astype("object").where(
            frame[candidate.feature].notna(), "__MISSING__"
        )
        return [
            (label, frame[values.isin(set(group))].index.tolist())
            for label, group in zip(candidate.branch_labels, candidate.value)
        ]
    elif candidate.split_type == "category_manual_groups":
        values = frame[candidate.feature].astype("object").where(
            frame[candidate.feature].notna(), "__MISSING__"
        )
        return [
            (label, frame[values.isin(set(group))].index.tolist())
            for label, group in zip(candidate.branch_labels, candidate.value)
        ]
    else:
        raise ValueError(f"Unknown split_type: {candidate.split_type}")


def split_node(
    df: pd.DataFrame,
    node_id: int,
    candidate: SplitCandidate,
    select_first_child: bool = True,
    record_action: bool = True,
) -> None:
    clear_ui_metric_cache()
    node = st.session_state.tree[node_id]
    if node["split"] is not None:
        prune_node(node_id)
        node = st.session_state.tree[node_id]

    row_idx = node["row_idx"]
    branch_indices = split_branch_indices(df, row_idx, candidate)

    first_child_id = st.session_state.next_node_id
    child_ids = list(range(first_child_id, first_child_id + len(branch_indices)))
    st.session_state.next_node_id = first_child_id + len(branch_indices)

    node["split"] = {
        "feature": candidate.feature,
        "split_type": candidate.split_type,
        "value": candidate.value,
        "label": candidate.label,
        "branch_count": candidate.branch_count,
        "branch_labels": list(candidate.branch_labels),
        "information_gain": candidate.information_gain,
        "weighted_entropy": candidate.weighted_entropy,
        "missing_policy": candidate.missing_policy,
    }
    node["children"] = [
        {"id": child_id, "label": branch_label}
        for child_id, (branch_label, _) in zip(child_ids, branch_indices)
    ]
    node["left"] = child_ids[0] if len(child_ids) == 2 else None
    node["right"] = child_ids[1] if len(child_ids) == 2 else None

    for child_id, (branch_label, child_idx) in zip(child_ids, branch_indices):
        st.session_state.tree[child_id] = {
            "id": child_id,
            "depth": node["depth"] + 1,
            "path": f"{node['path']} -> {candidate.feature} {branch_label}",
            "row_idx": child_idx,
            "split": None,
            "children": [],
            "left": None,
            "right": None,
        }
    st.session_state.setdefault("split_history", []).append(node_id)
    if record_action:
        st.session_state.setdefault("split_action_history", []).append([node_id])
    st.session_state.current_node_id = child_ids[0] if select_first_child else node_id


def apply_split(df: pd.DataFrame, candidate: SplitCandidate) -> None:
    split_node(df, st.session_state.current_node_id, candidate, select_first_child=True)


def prune_node(node_id: int) -> None:
    clear_ui_metric_cache()
    tree = st.session_state.tree
    node = tree[node_id]
    to_delete: list[int] = []

    def collect(child_id: int | None) -> None:
        if child_id is None:
            return
        child = tree[child_id]
        for grandchild in get_node_children(child):
            collect(grandchild["id"])
        to_delete.append(child_id)

    for child in get_node_children(node):
        collect(child["id"])

    for child_id in to_delete:
        tree.pop(child_id, None)

    node["split"] = None
    node["children"] = []
    node["left"] = None
    node["right"] = None
    removed = set(to_delete + [node_id])
    st.session_state.split_history = [
        split_node_id
        for split_node_id in st.session_state.get("split_history", [])
        if split_node_id not in removed
    ]
    st.session_state.split_action_history = [
        kept_action
        for action in st.session_state.get("split_action_history", [])
        if (kept_action := [split_node_id for split_node_id in action if split_node_id not in removed])
    ]
    st.session_state.current_node_id = node_id


def undo_last_split() -> bool:
    while st.session_state.get("split_action_history"):
        action_history = st.session_state.get("split_action_history", [])
        node_ids = action_history.pop()
        st.session_state.split_action_history = action_history
        undone_node_id: int | None = None
        for node_id in reversed(node_ids):
            node = st.session_state.tree.get(node_id)
            if node is not None and node["split"] is not None:
                prune_node(node_id)
                undone_node_id = node_id
        if undone_node_id is not None:
            st.session_state.current_node_id = undone_node_id
            return True

    history = st.session_state.get("split_history", [])
    while history:
        node_id = history.pop()
        node = st.session_state.tree.get(node_id)
        if node is not None and node["split"] is not None:
            prune_node(node_id)
            st.session_state.current_node_id = node_id
            return True
    st.session_state.split_history = []
    return False


def best_candidate_for_node(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    node: dict[str, Any],
    min_leaf: int,
    max_thresholds: int,
    max_categories: int,
    max_numeric_bins: int,
    max_category_groups: int,
) -> SplitCandidate | None:
    if len(set(df.loc[node["row_idx"], target].astype(str))) <= 1:
        return None
    candidates = candidate_splits(
        df=df,
        target=target,
        features=features,
        row_idx=node["row_idx"],
        min_leaf=min_leaf,
        max_thresholds=max_thresholds,
        max_categories=max_categories,
        max_numeric_bins=max_numeric_bins,
        max_category_groups=max_category_groups,
    )
    return candidates[0] if candidates else None


def build_optimal_tree(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    test_df: pd.DataFrame | None,
    min_leaf: int,
    max_thresholds: int,
    max_categories: int,
    max_numeric_bins: int,
    max_category_groups: int,
    max_depth: int,
    max_leaves: int,
    min_information_gain: float,
    candidate_rows: int,
    parallel_workers: int,
    max_validation_gini_gap: float,
    reset_tree: bool = True,
) -> int:
    if reset_tree or not st.session_state.get("tree"):
        init_tree(df)
    else:
        st.session_state.auto_tree_message = ""
    split_count = 0
    action_node_ids: list[int] = []
    validation_enabled = test_df is not None and infer_target_kind(df[target]) == "binary"

    while True:
        if len(current_leaves()) >= max_leaves:
            break

        best_node_id: int | None = None
        best_candidate: SplitCandidate | None = None
        best_score: tuple[Any, ...] | None = None

        leaf_count_now = len(current_leaves())
        baseline_train_predictions = tree_predictions_for_dataframe(df, df, target) if validation_enabled else None
        baseline_test_predictions = (
            tree_predictions_for_dataframe(df, test_df, target)
            if validation_enabled and test_df is not None
            else None
        )
        for leaf in current_leaves():
            if leaf["depth"] >= max_depth:
                continue

            search_row_idx = analysis_row_idx(
                leaf["row_idx"],
                candidate_rows,
                df=df,
                target=target,
            )
            candidates = candidate_splits(
                df=df,
                target=target,
                features=features,
                row_idx=search_row_idx,
                min_leaf=min_leaf,
                max_thresholds=max_thresholds,
                max_categories=max_categories,
                max_numeric_bins=max_numeric_bins,
                max_category_groups=max_category_groups,
                parallel_workers=parallel_workers,
            )
            validation_lookup: dict[int, dict[str, Any]] = {}
            if validation_enabled and test_df is not None:
                validation_lookup = {}
                for validation_candidate in candidates_selected_for_validation(candidates, VALIDATION_CANDIDATE_LIMIT):
                    stats = candidate_validation_stats(
                        train_df=df,
                        test_df=test_df,
                        target=target,
                        node_id=leaf["id"],
                        candidate=validation_candidate,
                        max_gini_gap=max_validation_gini_gap,
                        baseline_train_predictions=baseline_train_predictions,
                        baseline_test_predictions=baseline_test_predictions,
                    )
                    if stats is not None:
                        validation_lookup[id(validation_candidate)] = stats

            ordered_candidates = sorted(
                candidates,
                key=lambda candidate: candidate_validation_sort_key(candidate, validation_lookup),
                reverse=True,
            ) if validation_lookup else candidates
            for candidate in ordered_candidates:
                if candidate.information_gain < min_information_gain:
                    if validation_lookup:
                        continue
                    break
                if leaf_count_now + candidate.branch_count - 1 > max_leaves:
                    continue
                if not candidate_passes_validation(candidate, validation_lookup):
                    continue

                weighted_delta = candidate_total_gain_delta(df, candidate, leaf["row_idx"])
                if validation_lookup:
                    stats = validation_lookup.get(id(candidate), {})
                    candidate_score = (
                        1 if stats.get("validation_safe") else 0,
                        numeric_sort_value(stats.get("test_gini_after"), -np.inf),
                        numeric_sort_value(stats.get("test_gini_delta"), -np.inf),
                        -numeric_sort_value(stats.get("gini_gap_after"), np.inf),
                        weighted_delta,
                    )
                else:
                    candidate_score = (weighted_delta,)
                if best_score is None or candidate_score > best_score:
                    best_score = candidate_score
                    best_candidate = candidate
                    best_node_id = leaf["id"]
                break

        if best_node_id is None or best_candidate is None:
            break

        split_node(df, best_node_id, best_candidate, select_first_child=False, record_action=False)
        action_node_ids.append(best_node_id)
        split_count += 1

    if action_node_ids:
        st.session_state.setdefault("split_action_history", []).append(action_node_ids)
    st.session_state.current_node_id = 0
    st.session_state.tree_zoom = recommended_tree_zoom()
    return split_count


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def normalize_work_id(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in ("-", "_"))
    return cleaned[:80] or None


def ensure_work_id() -> str:
    work_id = normalize_work_id(st.query_params.get(WORK_ID_QUERY_PARAM))
    if work_id is None:
        work_id = uuid4().hex
        st.query_params[WORK_ID_QUERY_PARAM] = work_id
    return work_id


def checkpoint_path(work_id: str) -> Path:
    safe_work_id = normalize_work_id(work_id) or uuid4().hex
    return CHECKPOINT_DIR / f"{safe_work_id}.json"


def load_work_checkpoint(work_id: str) -> dict[str, Any] | None:
    path = checkpoint_path(work_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    columns = "\x1f".join(map(str, df.columns)).encode("utf-8", errors="replace")
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    except TypeError:
        row_hashes = df.astype(str).to_csv(index=True).encode("utf-8", errors="replace")
    return hashlib.sha256(columns + row_hashes).hexdigest()[:16]


def demo_data_key(df: pd.DataFrame) -> str:
    return f"demo:{len(df)}:{len(df.columns)}:{dataframe_fingerprint(df)}"


def uploaded_data_key(uploaded_name: str, df: pd.DataFrame) -> str:
    return f"uploaded:{uploaded_name}:{len(df)}:{len(df.columns)}:{dataframe_fingerprint(df)}"


def read_uploaded_table(uploaded: Any) -> pd.DataFrame:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(uploaded)
    raise ValueError("Unsupported upload type. Use CSV or Excel.")


def load_session_dataframe_from_query() -> tuple[pd.DataFrame, dict[str, Any], str, str] | None:
    data_id = normalize_data_id(st.query_params.get(DATA_ID_QUERY_PARAM))
    if data_id is None:
        return None
    try:
        df, metadata = load_dataframe_session(data_id)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        st.sidebar.warning(
            f"Session data could not be loaded: {exc}. "
            f"Session directory: {default_session_dir()}"
        )
        return None
    return df, metadata, data_id, session_data_key(data_id, df, metadata)


def source_metadata_value(metadata: dict[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if value in ("", [], {}):
        return None
    return value


def data_context_signature(context: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if not context:
        return None
    return (
        str(context.get("setup_key") or context.get("data_key") or ""),
        str(context.get("target") or ""),
        tuple(str(feature) for feature in context.get("features", []) or []),
        str(context.get("positive_class") or ""),
        int(context.get("split_variable_limit") or 0),
    )


def feature_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):
        return "categorical"
    return "other"


def normalize_feature_selection(features: list[str], selected: Any) -> list[str]:
    feature_set = set(map(str, features))
    if selected is None:
        return []
    if isinstance(selected, str):
        selected_iterable = [selected]
    else:
        try:
            selected_iterable = list(selected)
        except TypeError:
            selected_iterable = []
    return [str(feature) for feature in selected_iterable if str(feature) in feature_set]


def filter_feature_options(features: list[str], query: str | None) -> list[str]:
    text = str(query or "").strip().lower()
    if not text:
        return list(features)
    tokens = [
        token.strip().lower().replace("%", "*")
        for chunk in text.replace("\n", ",").split(",")
        for token in [chunk]
        if token.strip()
    ]
    if not tokens:
        return list(features)

    def matches(feature: str) -> bool:
        lowered = str(feature).lower()
        return any(
            fnmatch.fnmatch(lowered, token) if "*" in token or "?" in token else token in lowered
            for token in tokens
        )

    return [feature for feature in features if matches(feature)]


def ordered_feature_selection(features: list[str], selected: set[str]) -> list[str]:
    return [feature for feature in features if feature in selected]


def update_feature_selection_for_filtered(
    features: list[str],
    current_selection: list[str],
    filtered_features: list[str],
    include: bool,
) -> list[str]:
    selected = set(normalize_feature_selection(features, current_selection))
    filtered = set(map(str, filtered_features))
    if include:
        selected.update(filtered)
    else:
        selected.difference_update(filtered)
    return ordered_feature_selection(features, selected)


def rerun_current_fragment() -> None:
    try:
        st.rerun(scope="fragment")
    except Exception:
        st.rerun()


def apply_feature_manager_edits(
    features: list[str],
    current_selection: list[str],
    edited_table: pd.DataFrame,
) -> list[str]:
    selected = set(normalize_feature_selection(features, current_selection))
    if edited_table.empty or not {"include", "variable"}.issubset(edited_table.columns):
        return [feature for feature in features if feature in selected]

    visible_features = [str(feature) for feature in edited_table["variable"].tolist()]
    selected.difference_update(visible_features)
    included = edited_table.loc[edited_table["include"].fillna(False).astype(bool), "variable"].astype(str)
    selected.update(included.tolist())
    return [feature for feature in features if feature in selected]


def render_feature_filter_popover(
    container: Any,
    features: list[str],
    selected_features: list[str],
    *,
    state_key: str,
    source_key: str,
    target: str,
) -> list[str]:
    with container.popover("Dropdown Filter Panel"):
        selected_features = normalize_feature_selection(
            features,
            st.session_state.get(state_key, selected_features),
        )
        query = st.text_input(
            "Search variable",
            placeholder="Search",
            key=f"feature_search::{source_key}::{target}",
            label_visibility="collapsed",
            help="Case-insensitive contains search. Use comma/newline for multiple terms; * or % works as wildcard.",
        )
        filtered_features = filter_feature_options(features, query)
        filtered_signature = hashlib.sha256(
            json.dumps(filtered_features, default=str).encode("utf-8")
        ).hexdigest()[:10]
        selection_signature = hashlib.sha256(
            json.dumps(selected_features, default=str).encode("utf-8")
        ).hexdigest()[:10]
        st.caption(
            f"{len(selected_features):,} selected | {len(filtered_features):,} matching | {len(features):,} total"
        )

        selected_set = set(selected_features)
        all_filtered_selected = bool(filtered_features) and all(
            feature in selected_set for feature in filtered_features
        )
        select_all_value = st.checkbox(
            "(Select All)",
            value=all_filtered_selected,
            key=f"feature_filter_all::{source_key}::{target}::{filtered_signature}::{selection_signature}",
            disabled=not filtered_features,
        )
        if filtered_features and select_all_value != all_filtered_selected:
            st.session_state[state_key] = update_feature_selection_for_filtered(
                features,
                selected_features,
                filtered_features,
                select_all_value,
            )
            rerun_current_fragment()

        list_container = st.container(height=260, border=True)
        next_selected = set(selected_features)
        changed = False
        visible_features = filtered_features[:FEATURE_FILTER_MAX_VISIBLE]
        if len(filtered_features) > len(visible_features):
            list_container.caption(
                f"Showing first {len(visible_features):,} matching variable(s). Use search to narrow the list."
            )
        for feature in visible_features:
            was_selected = feature in selected_set
            is_selected = list_container.checkbox(
                str(feature),
                value=was_selected,
                key=(
                    f"feature_filter_item::{source_key}::{target}::"
                    f"{filtered_signature}::{selection_signature}::{feature}"
                ),
            )
            if is_selected != was_selected:
                changed = True
                if is_selected:
                    next_selected.add(feature)
                else:
                    next_selected.discard(feature)
        if changed:
            st.session_state[state_key] = ordered_feature_selection(features, next_selected)
            rerun_current_fragment()
        if not filtered_features:
            st.caption("No matching variables.")
        st.caption("Changes apply to Data Table Search & Filter Component before Apply data setup.")

    return filtered_features


def feature_manager_frame(
    df: pd.DataFrame,
    features: list[str],
    selected_features: list[str],
    *,
    sample_rows: int = FEATURE_PROFILE_SAMPLE_ROWS,
) -> pd.DataFrame:
    selected = set(map(str, selected_features))
    profile_n = min(len(df), max(0, sample_rows))
    sample = df[features].head(profile_n) if features and profile_n else pd.DataFrame(index=[])
    rows: list[dict[str, Any]] = []
    for feature in features:
        series = df[feature]
        sample_series = sample[feature] if feature in sample.columns else series.head(0)
        missing_rate = float(sample_series.isna().mean()) if len(sample_series) else 0.0
        unique_count = int(sample_series.nunique(dropna=True)) if len(sample_series) else 0
        rows.append(
            {
                "include": feature in selected,
                "variable": str(feature),
                "kind": feature_kind(series),
                "dtype": str(series.dtype),
                "missing_rate_sample": missing_rate,
                "unique_count_sample": unique_count,
                "profile_rows": int(profile_n),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "include",
            "variable",
            "kind",
            "dtype",
            "missing_rate_sample",
            "unique_count_sample",
            "profile_rows",
        ],
    )


def feature_profile_cache_key(data_key: str, target: str, features: list[str], sample_rows: int) -> str:
    payload = {
        "data_key": str(data_key),
        "target": str(target),
        "features": list(map(str, features)),
        "sample_rows": int(sample_rows),
    }
    return hashlib.sha256(json.dumps(payload, default=str).encode("utf-8")).hexdigest()[:16]


def feature_profile_cache() -> dict[str, pd.DataFrame]:
    key = "_interactive_tree_feature_profile_cache"
    cache = st.session_state.setdefault(key, {})
    if not isinstance(cache, dict):
        st.session_state[key] = {}
        cache = st.session_state[key]
    return cache


def cached_feature_profile_frame(
    df: pd.DataFrame,
    features: list[str],
    *,
    data_key: str,
    target: str,
    sample_rows: int = FEATURE_PROFILE_SAMPLE_ROWS,
) -> pd.DataFrame:
    cache = feature_profile_cache()
    cache_key = feature_profile_cache_key(data_key, target, features, sample_rows)
    if cache_key not in cache:
        profile = feature_manager_frame(df, features, features, sample_rows=sample_rows).drop(columns=["include"])
        if len(cache) >= 8:
            cache.pop(next(iter(cache)))
        cache[cache_key] = profile
    return cache[cache_key].copy()


def feature_manager_frame_from_profile(
    profile: pd.DataFrame,
    visible_features: list[str],
    selected_features: list[str],
) -> pd.DataFrame:
    selected = set(map(str, selected_features))
    rows_by_feature = {str(row["variable"]): dict(row) for row in profile.to_dict("records")}
    rows = []
    for feature in visible_features:
        row = dict(rows_by_feature.get(str(feature), {"variable": str(feature)}))
        row["include"] = str(feature) in selected
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "include",
            "variable",
            "kind",
            "dtype",
            "missing_rate_sample",
            "unique_count_sample",
            "profile_rows",
        ],
    )


def effective_feature_selection(
    df: pd.DataFrame,
    features: list[str],
    table_selected_features: list[str],
    include_mode: str,
) -> list[str]:
    if include_mode == "Include all":
        return list(features)
    if include_mode == "Numeric only":
        return [feature for feature in features if feature_kind(df[feature]) == "numeric"]
    if include_mode == "Categorical only":
        return [feature for feature in features if feature_kind(df[feature]) == "categorical"]
    return list(table_selected_features)


@st.fragment
def render_feature_manager_fragment(
    df: pd.DataFrame,
    features: list[str],
    default_selected_features: list[str],
    *,
    data_key: str,
    target: str,
    saved_parameters: dict[str, Any],
) -> tuple[list[str], int]:
    feature_selection_key = f"feature_selection::{data_key}::{target}"
    st.session_state[feature_selection_key] = normalize_feature_selection(
        features,
        st.session_state.get(feature_selection_key, default_selected_features),
    )
    current_feature_selection = st.session_state[feature_selection_key]

    st.markdown("**Feature manager**")
    st.caption(
        f"Profile columns are estimated on the first {min(len(df), FEATURE_PROFILE_SAMPLE_ROWS):,} row(s). "
        "Only included variables are available in Tree Builder and WOE Binning."
    )
    feature_include_mode_key = f"feature_include_mode::{data_key}::{target}"
    legacy_include_mode = st.session_state.get("feature_include_mode", "Use table selection")
    feature_include_mode = st.radio(
        "Feature include mode",
        ["Use table selection", "Include all", "Numeric only", "Categorical only"],
        horizontal=True,
        key=feature_include_mode_key,
        index=["Use table selection", "Include all", "Numeric only", "Categorical only"].index(
            legacy_include_mode
            if legacy_include_mode in {"Use table selection", "Include all", "Numeric only", "Categorical only"}
            else "Use table selection"
        )
        if feature_include_mode_key not in st.session_state
        else None,
    )
    filtered_features = render_feature_filter_popover(
        st,
        features,
        current_feature_selection,
        state_key=feature_selection_key,
        source_key=data_key,
        target=target,
    )
    current_feature_selection = normalize_feature_selection(
        features,
        st.session_state.get(feature_selection_key, current_feature_selection),
    )
    filtered_signature = hashlib.sha256(
        json.dumps(filtered_features, default=str).encode("utf-8")
    ).hexdigest()[:10]
    selection_signature = hashlib.sha256(
        json.dumps(current_feature_selection, default=str).encode("utf-8")
    ).hexdigest()[:10]
    feature_profile = cached_feature_profile_frame(
        df,
        features,
        data_key=data_key,
        target=target,
    )
    st.markdown("**Data Table Search & Filter Component**")
    st.caption("The table shows matching variables only; selections outside the filter are preserved.")
    feature_manager = st.data_editor(
        feature_manager_frame_from_profile(feature_profile, filtered_features, current_feature_selection),
        hide_index=True,
        width="stretch",
        key=f"feature_manager::{data_key}::{target}::{filtered_signature}::{selection_signature}",
        disabled=[
            "variable",
            "kind",
            "dtype",
            "missing_rate_sample",
            "unique_count_sample",
            "profile_rows",
        ],
        column_config={
            "include": st.column_config.CheckboxColumn("include"),
            "variable": st.column_config.TextColumn("variable"),
            "kind": st.column_config.TextColumn("kind"),
            "dtype": st.column_config.TextColumn("dtype"),
            "missing_rate_sample": st.column_config.NumberColumn("missing_rate_sample", format="%.4f"),
            "unique_count_sample": st.column_config.NumberColumn("unique_count_sample", format="%d"),
            "profile_rows": st.column_config.NumberColumn("profile_rows", format="%d"),
        },
    )
    table_selected_features = apply_feature_manager_edits(
        features,
        current_feature_selection,
        feature_manager,
    )
    st.session_state[feature_selection_key] = table_selected_features
    draft_features = effective_feature_selection(df, features, table_selected_features, str(feature_include_mode))
    st.caption(f"Staged split variables: {len(draft_features):,} / {len(features):,}")

    split_limit_key = f"split_variable_limit::{data_key}::{target}"
    saved_limit = safe_int(
        saved_parameters.get("split_variable_limit", min(50, max(1, len(draft_features)))),
        default=min(50, max(1, len(draft_features))),
        minimum=1,
    )
    split_limit_default = min(max(1, len(draft_features)), saved_limit) if draft_features else 1
    if split_limit_key not in st.session_state:
        st.session_state[split_limit_key] = split_limit_default
    elif draft_features:
        st.session_state[split_limit_key] = min(
            max(1, safe_int(st.session_state.get(split_limit_key), default=split_limit_default, minimum=1)),
            max(1, len(draft_features)),
        )
    draft_split_variable_limit_input = st.number_input(
        "Split ranking variable limit",
        value=int(st.session_state.get(split_limit_key, split_limit_default)),
        min_value=1,
        max_value=max(1, len(draft_features)),
        step=1,
        format="%d",
        disabled=not draft_features,
        key=split_limit_key,
        help=(
            "Only the first N active split variables are evaluated for split ranking and optimal tree. "
            "Set this here, then apply data setup before computing rankings."
        ),
    )
    draft_split_variable_limit = min(
        len(draft_features),
        safe_int(
            draft_split_variable_limit_input,
            default=min(50, max(1, len(draft_features))),
            minimum=1,
        ),
    ) if draft_features else 0
    return draft_features, draft_split_variable_limit


def secret_sql_connections() -> dict[str, str]:
    connections: dict[str, str] = {}

    def add_mapping(prefix: str, mapping: Any) -> None:
        if not hasattr(mapping, "items"):
            return
        for name, value in mapping.items():
            label = f"{prefix}{name}"
            if isinstance(value, str):
                connections[label] = value
            elif hasattr(value, "get"):
                url = value.get("url") or value.get("connection_url") or value.get("sqlalchemy_url")
                if url:
                    connections[label] = str(url)

    try:
        add_mapping("", st.secrets.get("sql_connections", {}))
        add_mapping("", st.secrets.get("connections", {}))
    except (FileNotFoundError, KeyError, AttributeError):
        return {}
    return connections


def save_source_session(
    df: pd.DataFrame,
    *,
    source: str,
    name: str,
    target: str | None = None,
    features: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    data_id, _ = save_dataframe_session(
        df,
        source=source,
        name=name,
        target=target,
        features=features,
        metadata=metadata,
    )
    st.query_params[DATA_ID_QUERY_PARAM] = data_id
    st.session_state["_last_query_data_id"] = None
    return data_id


def persist_source_session(
    df: pd.DataFrame,
    *,
    source: str,
    name: str,
    target: str | None = None,
    features: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    update_query_params: bool = False,
) -> str:
    if update_query_params:
        return save_source_session(
            df,
            source=source,
            name=name,
            target=target,
            features=features,
            metadata=metadata,
        )
    data_id, _ = save_dataframe_session(
        df,
        source=source,
        name=name,
        target=target,
        features=features,
        metadata=metadata,
    )
    return data_id


def loaded_source_from_session(data_id: str, fallback_name: str | None = None) -> LoadedDataSource:
    df, metadata = load_dataframe_session(data_id)
    name = str(metadata.get("name") or fallback_name or "Session DataFrame")
    source = str(metadata.get("source") or "session")
    return LoadedDataSource(
        df=df,
        name=name,
        data_key=session_data_key(data_id, df, metadata),
        source=source,
        data_id=data_id,
        metadata=metadata,
    )


def render_session_source_loader(
    container: Any,
    *,
    role: str,
    query_session: tuple[pd.DataFrame, dict[str, Any], str, str] | None = None,
) -> LoadedDataSource | None:
    session_key = f"{role}_session_data_id"
    if role == "train" and query_session is not None:
        query_data_id = query_session[2]
        widget_marker_key = f"{role}_session_widget_query_data_id"
        if st.session_state.get(widget_marker_key) != query_data_id:
            st.session_state[session_key] = query_data_id
            st.session_state[widget_marker_key] = query_data_id

    data_id_text = container.text_input(
        "Session data_id",
        key=session_key,
        help="Notebook launch_tree(...) URL'indeki data_id degerini kullanir; raw dosya path kabul edilmez.",
    )
    data_id = normalize_data_id(data_id_text)
    if data_id is None:
        container.info("Session DataFrame icin notebook URL'indeki data_id degerini girin.")
        return None
    if query_session is not None and data_id == query_session[2]:
        df, metadata, source_data_id, data_key = query_session
        return LoadedDataSource(
            df=df,
            name=str(metadata.get("name") or "Session DataFrame"),
            data_key=data_key,
            source=str(metadata.get("source") or "session"),
            data_id=source_data_id,
            metadata=metadata,
        )
    try:
        return loaded_source_from_session(data_id)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        container.error(f"Session data could not be loaded: {exc}")
        return None


def render_upload_source_loader(
    container: Any,
    *,
    role: str,
    checkpoint: dict[str, Any] | None = None,
    allow_checkpoint_restore: bool = False,
    update_query_params: bool = False,
) -> LoadedDataSource | None:
    uploaded = container.file_uploader(
        "CSV veya Excel yukle",
        type=["csv", "xlsx", "xls"],
        key=f"{role}_csv_excel_upload",
    )
    if uploaded is None:
        if allow_checkpoint_restore:
            restored_dataframe = restore_checkpoint_dataframe(checkpoint)
            if restored_dataframe is not None:
                df, uploaded_name, data_key = restored_dataframe
                return LoadedDataSource(
                    df=df,
                    name=uploaded_name,
                    data_key=data_key,
                    source="uploaded",
                    data_id=None,
                    metadata={"source": "uploaded", "name": uploaded_name},
                    restored_upload=True,
                )
        container.info("CSV veya Excel dosyasi yukleyin.")
        return None

    try:
        df = read_uploaded_table(uploaded)
    except Exception as exc:
        container.error(f"File load failed: {exc}")
        return None

    upload_key = uploaded_data_key(uploaded.name, df)
    metadata = {
        "source": "uploaded",
        "name": uploaded.name,
        "upload_name": uploaded.name,
        "fingerprint": dataframe_fingerprint(df),
    }
    cache_key = f"_{role}_uploaded_session"
    cached_upload = st.session_state.get(cache_key, {})
    if cached_upload.get("upload_key") == upload_key:
        data_id = cached_upload.get("data_id")
    else:
        data_id = persist_source_session(
            df,
            source="uploaded",
            name=uploaded.name,
            metadata={"upload_name": uploaded.name},
            update_query_params=update_query_params,
        )
        st.session_state[cache_key] = {"upload_key": upload_key, "data_id": data_id}

    return LoadedDataSource(
        df=df,
        name=uploaded.name,
        data_key=session_data_key(str(data_id), df, metadata),
        source="uploaded",
        data_id=str(data_id),
        metadata=metadata,
    )


def render_sql_source_loader(
    container: Any = st.sidebar,
    *,
    role: str = "train",
    update_query_params: bool = True,
    use_form: bool = True,
) -> LoadedDataSource | None:
    loaded_key = f"{role}_sql_loaded_source"
    loaded_payload = st.session_state.get(loaded_key)
    if isinstance(loaded_payload, dict):
        loaded_data_id = normalize_data_id(loaded_payload.get("data_id"))
        if loaded_data_id is not None:
            try:
                return loaded_source_from_session(
                    loaded_data_id,
                    fallback_name=str(loaded_payload.get("name") or "SQL data"),
                )
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                st.session_state.pop(loaded_key, None)

    connections = secret_sql_connections()
    connection_mode_options = ["Manual SQLAlchemy URL"]
    if connections:
        connection_mode_options.insert(0, "Saved secret connection")

    def render_sql_controls(control_container: Any) -> tuple[str, str, str, str, bool, int, int]:
        connection_mode = control_container.selectbox(
            "SQL connection",
            connection_mode_options,
            key=f"{role}_sql_connection_mode",
        )
        if connection_mode == "Saved secret connection":
            selected_connection = control_container.selectbox(
                "Saved connection",
                sorted(connections),
                key=f"{role}_sql_saved_connection",
            )
            connection_url = connections[selected_connection]
        else:
            connection_url = control_container.text_input(
                "SQLAlchemy connection URL",
                type="password",
                key=f"{role}_sql_connection_url",
            )

        sql_mode = control_container.radio("SQL mode", ["Table", "Query"], horizontal=True, key=f"{role}_sql_mode")
        table_name = ""
        query_text = ""
        if sql_mode == "Table":
            table_name = control_container.text_input("Table name", key=f"{role}_sql_table")
        else:
            query_text = control_container.text_area("SQL query", height=120, key=f"{role}_sql_query")

        full_table = control_container.checkbox("Load full result", value=False, key=f"{role}_sql_full_table")
        limit_value = control_container.number_input(
            "Row limit",
            value=DEFAULT_SQL_LIMIT,
            min_value=1,
            step=1000,
            disabled=full_table,
            format="%d",
            key=f"{role}_sql_limit",
        )
        sample_n_value = control_container.number_input(
            "Optional sample rows",
            value=0,
            min_value=0,
            step=100,
            format="%d",
            key=f"{role}_sql_sample_n",
        )
        return connection_url, sql_mode, table_name, query_text, full_table, int(limit_value), int(sample_n_value)

    if use_form:
        with container.form(f"{role}_sql_source_form"):
            connection_url, sql_mode, table_name, query_text, full_table, limit_value, sample_n_value = render_sql_controls(st)
            submitted = st.form_submit_button(f"Load {role} SQL data", width="stretch")
    else:
        connection_url, sql_mode, table_name, query_text, full_table, limit_value, sample_n_value = render_sql_controls(container)
        submitted = bool(connection_url and (table_name.strip() or query_text.strip()))

    if not submitted:
        container.info("Choose a SQL source in the sidebar and load data.")
        return None

    if not connection_url:
        container.error("SQL connection URL is required.")
        return None

    try:
        df = read_sql_dataframe(
            connection_url,
            table=table_name.strip() or None,
            query=query_text.strip() or None,
            limit=None if full_table else limit_value,
            sample_n=sample_n_value or None,
            full_table=bool(full_table),
        )
        metadata = {
            "sql": {
                "mode": sql_mode.lower(),
                "table": table_name.strip() or None,
                "limit": None if full_table else limit_value,
                "sample_n": sample_n_value or None,
                "full_table": bool(full_table),
            }
        }
        source_name = table_name.strip() or "SQL query"
        data_id = persist_source_session(
            df,
            source="sql",
            name=source_name,
            metadata=metadata,
            update_query_params=update_query_params,
        )
        st.session_state[loaded_key] = {"data_id": data_id, "name": source_name}
        if update_query_params:
            st.rerun()
        return LoadedDataSource(
            df=df,
            name=source_name,
            data_key=session_data_key(str(data_id), df, {"source": "sql", "name": source_name, **metadata}),
            source="sql",
            data_id=str(data_id),
            metadata={"source": "sql", "name": source_name, **metadata},
        )
    except Exception as exc:
        container.error(f"SQL load failed: {exc}")
        return None


def render_dataframe_source_loader(
    container: Any,
    *,
    role: str,
    source_choice: str,
    query_session: tuple[pd.DataFrame, dict[str, Any], str, str] | None = None,
    checkpoint: dict[str, Any] | None = None,
    allow_checkpoint_restore: bool = False,
    update_query_params: bool = False,
    use_sql_form: bool = True,
) -> LoadedDataSource | None:
    if source_choice == "Session DataFrame":
        return render_session_source_loader(container, role=role, query_session=query_session)
    if source_choice == "CSV / Excel Upload":
        return render_upload_source_loader(
            container,
            role=role,
            checkpoint=checkpoint,
            allow_checkpoint_restore=allow_checkpoint_restore,
            update_query_params=update_query_params,
        )
    if source_choice == "SQL":
        return render_sql_source_loader(
            container,
            role=role,
            update_query_params=update_query_params,
            use_form=use_sql_form,
        )

    df = make_demo_data()
    metadata = {"source": "demo", "name": "Demo"}
    return LoadedDataSource(
        df=df,
        name="Demo",
        data_key=demo_data_key(df),
        source="demo",
        data_id=None,
        metadata=metadata,
    )


def restore_checkpoint_dataframe(checkpoint: dict[str, Any] | None) -> tuple[pd.DataFrame, str, str] | None:
    if not checkpoint:
        return None
    data = checkpoint.get("data")
    if not isinstance(data, dict) or data.get("source") != "uploaded":
        return None
    uploaded_name = str(data.get("name") or "restored_upload.csv")
    data_id = normalize_data_id(data.get("data_id"))
    if data_id is not None:
        try:
            df, metadata = load_dataframe_session(data_id)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            pass
        else:
            data_key = str(data.get("data_key") or session_data_key(data_id, df, metadata))
            return df, uploaded_name, data_key
    frame_json = data.get("frame_json")
    if not isinstance(frame_json, str) or not frame_json:
        return None
    try:
        df = pd.read_json(io.StringIO(frame_json), orient="split")
    except ValueError:
        return None
    data_key = str(data.get("data_key") or uploaded_data_key(uploaded_name, df))
    return df, uploaded_name, data_key


def loaded_source_from_query_session(
    query_session: tuple[pd.DataFrame, dict[str, Any], str, str] | None,
) -> LoadedDataSource | None:
    if query_session is None:
        return None
    df, metadata, source_data_id, data_key = query_session
    return LoadedDataSource(
        df=df,
        name=str(metadata.get("name") or "Session DataFrame"),
        data_key=data_key,
        source=str(metadata.get("source") or "session"),
        data_id=source_data_id,
        metadata=metadata,
    )


def restore_applied_context_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    query_session: tuple[pd.DataFrame, dict[str, Any], str, str] | None,
) -> dict[str, Any] | None:
    if not checkpoint:
        return None
    checkpoint_data = checkpoint.get("data")
    if not isinstance(checkpoint_data, dict):
        return None
    checkpoint_data_key = checkpoint_data.get("data_key")
    if not checkpoint_data_key:
        return None

    source = loaded_source_from_query_session(query_session)
    if source is None and checkpoint_data.get("source") == "uploaded":
        restored_upload = restore_checkpoint_dataframe(checkpoint)
        if restored_upload is not None:
            source_df, source_name, source_data_key = restored_upload
            source = LoadedDataSource(
                df=source_df,
                name=source_name,
                data_key=source_data_key,
                source="uploaded",
                data_id=None,
                metadata=dict(checkpoint_data.get("metadata") or {}),
                restored_upload=True,
            )
    if source is None:
        return None
    checkpoint_data_id = normalize_data_id(checkpoint_data.get("data_id"))
    if checkpoint_data_id is not None and source.data_id is not None and checkpoint_data_id != source.data_id:
        return None

    target = str(checkpoint.get("target") or source_metadata_value(source.metadata, "target") or "")
    if not target or target not in source.df.columns:
        return None

    selected_features = checkpoint.get("selected_features")
    if isinstance(selected_features, list):
        features = [
            str(feature)
            for feature in selected_features
            if str(feature) in source.df.columns and str(feature) != target
        ]
    else:
        metadata_features = source_metadata_value(source.metadata, "features")
        if isinstance(metadata_features, list):
            features = [
                str(feature)
                for feature in metadata_features
                if str(feature) in source.df.columns and str(feature) != target
            ]
        else:
            features = [str(column) for column in source.df.columns if str(column) != target]

    metadata = dict(source.metadata)
    checkpoint_metadata = checkpoint_data.get("metadata")
    if isinstance(checkpoint_metadata, dict):
        metadata.update(checkpoint_metadata)

    working_df = source.df
    sample_metadata = metadata.get("sample") if isinstance(metadata, dict) else None
    if isinstance(sample_metadata, dict) and sample_metadata.get("enabled"):
        sample_rows = safe_int(sample_metadata.get("sample_rows"), default=len(source.df), minimum=1)
        sample_rows = min(len(source.df), sample_rows)
        if sample_rows < len(source.df):
            sample_idx = analysis_row_idx(
                source.df.index.tolist(),
                max_rows=sample_rows,
                random_state=safe_int(sample_metadata.get("random_state"), default=CANDIDATE_RANDOM_STATE),
                df=source.df,
                stratify_columns=normalize_stratify_columns(
                    source.df,
                    sample_metadata.get("stratify_columns") or [],
                ),
                stratify_numeric_bins=safe_int(
                    sample_metadata.get("numeric_bins"),
                    default=DEFAULT_STRATIFY_NUMERIC_BINS,
                    minimum=2,
                ),
            )
            working_df = source.df.loc[sample_idx].copy()

    active_df = working_df
    test_df: pd.DataFrame | None = None
    validation_metadata = metadata.get("validation") if isinstance(metadata, dict) else None
    if isinstance(validation_metadata, dict) and validation_metadata.get("mode") == "split_train_data":
        train_idx, test_idx = train_test_split_indices(
            working_df,
            target,
            test_fraction=float(validation_metadata.get("test_fraction", 0.2)),
            random_state=safe_int(validation_metadata.get("random_state"), default=CANDIDATE_RANDOM_STATE),
            stratify=bool(validation_metadata.get("stratify_columns")),
            stratify_columns=normalize_stratify_columns(
                working_df,
                validation_metadata.get("stratify_columns") or [],
            ),
            stratify_numeric_bins=safe_int(
                validation_metadata.get("numeric_bins"),
                default=DEFAULT_STRATIFY_NUMERIC_BINS,
                minimum=2,
            ),
        )
        active_df = working_df.loc[train_idx].copy()
        test_df = working_df.loc[test_idx].copy()

    parameters = checkpoint.get("parameters") if isinstance(checkpoint.get("parameters"), dict) else {}
    return {
        "df": active_df,
        "test_df": test_df,
        "uploaded_name": source.name,
        "source_metadata": metadata,
        "source_data_id": source.data_id,
        "data_key": str(checkpoint_data_key),
        "data_source": source.source,
        "target": target,
        "target_kind": infer_target_kind(active_df[target]),
        "features": features,
        "positive_class": checkpoint.get("positive_class"),
        "split_variable_limit": safe_int(
            parameters.get("split_variable_limit"),
            default=max(1, len(features)),
            minimum=1,
        ),
    }


def staged_setup_key(
    source_data_key: str,
    sample_metadata: dict[str, Any] | None,
    validation_metadata: dict[str, Any] | None,
) -> str:
    payload = {
        "source_data_key": str(source_data_key),
        "sample": json_safe(sample_metadata or {}),
        "validation": json_safe(validation_metadata or {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def staged_data_key(
    source_data_key: str,
    sample_metadata: dict[str, Any] | None,
    validation_metadata: dict[str, Any] | None,
) -> str:
    parts = [str(source_data_key)]
    if isinstance(sample_metadata, dict) and sample_metadata.get("enabled"):
        parts.append(
            "sample_pending:"
            f"{sample_metadata.get('sample_rows')}:"
            f"{sample_metadata.get('random_state')}:"
            f"{','.join(map(str, sample_metadata.get('stratify_columns') or [])) or 'random'}:"
            f"{sample_metadata.get('numeric_bins')}"
        )
    if isinstance(validation_metadata, dict):
        mode = validation_metadata.get("mode")
        if mode == "split_train_data":
            parts.append(
                "train_split_pending:"
                f"{float(validation_metadata.get('test_fraction', 0.2)):.4f}:"
                f"{validation_metadata.get('random_state')}:"
                f"{','.join(map(str, validation_metadata.get('stratify_columns') or [])) or 'random'}:"
                f"{validation_metadata.get('numeric_bins')}"
            )
        elif mode == "separate_data_source":
            parts.append(
                "separate_test_pending:"
                f"{validation_metadata.get('data_id') or validation_metadata.get('name') or 'loaded'}"
            )
    return ":".join(parts)


def materialize_staged_data_context(context: dict[str, Any]) -> dict[str, Any]:
    if not context.get("_staged"):
        return context

    source_df = context["_source_df"]
    metadata = dict(context.get("source_metadata") or {})
    source_data_key = str(context.get("source_data_key") or context.get("data_key") or "data")
    target = str(context["target"])
    working_df = source_df
    working_data_key = source_data_key

    sample_metadata = metadata.get("sample") if isinstance(metadata.get("sample"), dict) else None
    if isinstance(sample_metadata, dict) and sample_metadata.get("enabled"):
        sample_rows = min(
            len(source_df),
            safe_int(sample_metadata.get("sample_rows"), default=len(source_df), minimum=1),
        )
        sample_seed = safe_int(sample_metadata.get("random_state"), default=CANDIDATE_RANDOM_STATE)
        sample_stratify_columns = normalize_stratify_columns(
            source_df,
            sample_metadata.get("stratify_columns") or [],
        )
        sample_numeric_bins = safe_int(
            sample_metadata.get("numeric_bins"),
            default=DEFAULT_STRATIFY_NUMERIC_BINS,
            minimum=2,
        )
        if sample_rows < len(source_df):
            sample_idx = analysis_row_idx(
                source_df.index.tolist(),
                max_rows=sample_rows,
                random_state=sample_seed,
                df=source_df,
                stratify_columns=sample_stratify_columns,
                stratify_numeric_bins=sample_numeric_bins,
            )
            working_df = source_df.loc[sample_idx].copy()
            working_data_key = (
                f"{source_data_key}:sample:{sample_rows}:{sample_seed}:"
                f"{','.join(sample_stratify_columns) or 'random'}:{sample_numeric_bins}:"
                f"{dataframe_fingerprint(working_df)}"
            )
        metadata["sample"] = {
            **sample_metadata,
            "enabled": True,
            "source_rows": int(len(source_df)),
            "sample_rows": int(len(working_df)),
            "random_state": int(sample_seed),
            "stratify_columns": list(sample_stratify_columns),
            "numeric_bins": int(sample_numeric_bins),
        }
    else:
        metadata.pop("sample", None)

    active_df = working_df
    test_df: pd.DataFrame | None = context.get("_separate_test_df")
    validation_metadata = metadata.get("validation") if isinstance(metadata.get("validation"), dict) else None
    data_key = working_data_key
    if isinstance(validation_metadata, dict) and validation_metadata.get("mode") == "split_train_data":
        split_stratify_columns = normalize_stratify_columns(
            working_df,
            validation_metadata.get("stratify_columns") or [],
        )
        split_numeric_bins = safe_int(
            validation_metadata.get("numeric_bins"),
            default=DEFAULT_STRATIFY_NUMERIC_BINS,
            minimum=2,
        )
        split_seed = safe_int(validation_metadata.get("random_state"), default=CANDIDATE_RANDOM_STATE)
        test_fraction = float(validation_metadata.get("test_fraction", 0.2))
        train_idx, test_idx = train_test_split_indices(
            working_df,
            target,
            test_fraction=test_fraction,
            random_state=split_seed,
            stratify=bool(split_stratify_columns),
            stratify_columns=split_stratify_columns,
            stratify_numeric_bins=split_numeric_bins,
        )
        active_df = working_df.loc[train_idx].copy()
        test_df = working_df.loc[test_idx].copy()
        data_key = (
            f"{working_data_key}:train_split:{float(test_fraction):.4f}:"
            f"{int(split_seed)}:{','.join(split_stratify_columns) or 'random'}:"
            f"{split_numeric_bins}:{dataframe_fingerprint(active_df)}"
        )
        metadata["validation"] = {
            **validation_metadata,
            "mode": "split_train_data",
            "test_fraction": float(test_fraction),
            "random_state": int(split_seed),
            "stratify_columns": list(split_stratify_columns),
            "numeric_bins": int(split_numeric_bins),
            "source_rows": int(len(working_df)),
            "train_rows": int(len(active_df)),
            "test_rows": int(len(test_df)),
        }
    elif isinstance(validation_metadata, dict) and validation_metadata.get("mode") == "separate_data_source":
        metadata["validation"] = {
            **validation_metadata,
            "rows": int(len(test_df)) if test_df is not None else 0,
        }
    else:
        test_df = None
        metadata.pop("validation", None)

    return {
        "df": active_df,
        "test_df": test_df,
        "uploaded_name": context.get("uploaded_name"),
        "source_metadata": metadata,
        "source_data_id": context.get("source_data_id"),
        "data_key": data_key,
        "setup_key": context.get("setup_key"),
        "data_source": context.get("data_source"),
        "target": target,
        "target_kind": infer_target_kind(active_df[target]),
        "features": list(context.get("features") or []),
        "positive_class": context.get("positive_class"),
        "split_variable_limit": safe_int(
            context.get("split_variable_limit"),
            default=max(1, len(context.get("features") or [])),
            minimum=1,
        ),
    }


def can_auto_apply_draft_context(context: dict[str, Any]) -> bool:
    if not context.get("_staged"):
        return len(context.get("df", [])) <= AUTO_APPLY_DATA_SETUP_MAX_ROWS
    if len(context.get("_source_df", [])) > AUTO_APPLY_DATA_SETUP_MAX_ROWS:
        return False
    metadata = context.get("source_metadata") if isinstance(context.get("source_metadata"), dict) else {}
    sample_metadata = metadata.get("sample") if isinstance(metadata.get("sample"), dict) else None
    validation_metadata = metadata.get("validation") if isinstance(metadata.get("validation"), dict) else None
    return not (sample_metadata and sample_metadata.get("enabled")) and not validation_metadata


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def deserialize_tree(tree_payload: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(tree_payload, dict):
        return {}

    tree: dict[int, dict[str, Any]] = {}
    for raw_key, raw_node in tree_payload.items():
        if not isinstance(raw_node, dict):
            continue
        node = dict(raw_node)
        node_id = int(node.get("id", raw_key))
        node["id"] = node_id
        node["depth"] = int(node.get("depth", 0))
        node["path"] = str(node.get("path", "root"))
        node["row_idx"] = [int(idx) for idx in node.get("row_idx", [])]
        node["left"] = int_or_none(node.get("left"))
        node["right"] = int_or_none(node.get("right"))
        node["children"] = [
            {"id": int(child["id"]), "label": str(child.get("label", ""))}
            for child in node.get("children", [])
            if isinstance(child, dict) and "id" in child
        ]
        split = node.get("split")
        if isinstance(split, dict):
            split = dict(split)
            split["branch_labels"] = list(split.get("branch_labels", []))
            split["missing_policy"] = str(split.get("missing_policy") or "right")
            node["split"] = split
        else:
            node["split"] = None
        tree[node_id] = node
    return tree


def restore_tree_state_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    state_key: tuple[str, str, int],
    df: pd.DataFrame,
) -> bool:
    if not checkpoint or checkpoint.get("tree_schema_version") != TREE_SCHEMA_VERSION:
        return False
    tree_state = checkpoint.get("tree_state")
    if not isinstance(tree_state, dict):
        return False
    checkpoint_state_key = tree_state.get("state_key")
    if not isinstance(checkpoint_state_key, (list, tuple)) or tuple(checkpoint_state_key) != state_key:
        return False

    tree = deserialize_tree(tree_state.get("tree"))
    if 0 not in tree:
        return False

    valid_index = set(df.index.tolist())
    for node in tree.values():
        if any(idx not in valid_index for idx in node["row_idx"]):
            return False

    st.session_state.tree = tree
    st.session_state.next_node_id = safe_int(
        tree_state.get("next_node_id"),
        default=max(tree) + 1,
        minimum=max(tree) + 1,
    )
    st.session_state.current_node_id = int(tree_state.get("current_node_id", 0))
    if st.session_state.current_node_id not in tree:
        st.session_state.current_node_id = 0
    st.session_state.split_history = [
        int(node_id)
        for node_id in tree_state.get("split_history", [])
        if int(node_id) in tree
    ]
    raw_action_history = tree_state.get("split_action_history")
    split_action_history: list[list[int]] = []
    if isinstance(raw_action_history, list):
        for action in raw_action_history:
            if not isinstance(action, list):
                continue
            node_ids = [int(node_id) for node_id in action if int(node_id) in tree]
            if node_ids:
                split_action_history.append(node_ids)
    if not split_action_history:
        split_action_history = [[node_id] for node_id in st.session_state.split_history]
    st.session_state.split_action_history = split_action_history
    st.session_state.auto_tree_message = str(tree_state.get("auto_tree_message", ""))
    st.session_state.tree_zoom = safe_float(tree_state.get("tree_zoom"), default=recommended_tree_zoom())
    return True


def is_checkpoint_ui_state_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    transient_widget_prefixes = (
        "group_all_",
        "group_single_",
        "group_profile_",
        "group_split_",
        "group_merge_button_",
    )
    if key.startswith(transient_widget_prefixes):
        return False
    prefixes = (
        "category_groups::",
        "node_features_v2::",
        "woe_selected_variables::",
        "manual_thresholds_",
        "manual_missing_policy_",
        "group_source_",
        "group_values_",
        "group_merge_",
    )
    exact_keys = {
        "data_source_choice",
        "validation_source_mode",
        "validation_test_share",
        "validation_split_seed",
        "validation_stratify_columns",
        "validation_stratify_numeric_bins",
        "data_sample_enabled",
        "data_sample_rows",
        "data_sample_seed",
        "data_sample_stratify_columns",
        "data_sample_stratify_numeric_bins",
        "test_data_source_choice",
        "train_session_data_id",
        "test_session_data_id",
        "workspace_mode",
        "feature_include_mode",
        "woe_selected_variables",
        "woe_max_bins",
        "woe_min_bin_size",
        "woe_monotonic_trend",
        "woe_engine",
        "woe_missing_separate",
        "woe_blank_as_missing",
        "woe_replace_existing",
        "_interactive_tree_woe_active_variable",
    }
    return key.startswith(prefixes) or key in exact_keys


def restore_checkpoint_ui_state(checkpoint: dict[str, Any] | None) -> None:
    if not checkpoint:
        return
    ui_state = checkpoint.get("ui_state")
    if not isinstance(ui_state, dict):
        return
    for key, value in ui_state.items():
        if is_checkpoint_ui_state_key(key) and key not in st.session_state:
            st.session_state[key] = value


def checkpoint_ui_state() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in list(st.session_state.keys()):
        if is_checkpoint_ui_state_key(key):
            out[key] = json_safe(st.session_state[key])
    return out


def save_work_checkpoint(
    work_id: str,
    df: pd.DataFrame,
    data_key: str,
    data_source: str,
    uploaded_name: str | None,
    data_id: str | None,
    source_metadata: dict[str, Any] | None,
    target: str,
    selected_features: list[str],
    parameters: dict[str, Any],
    auto_parameters: dict[str, Any],
) -> None:
    data_payload: dict[str, Any] = {
        "source": data_source,
        "name": uploaded_name,
        "data_id": data_id,
        "data_key": data_key,
        "rows": int(len(df)),
        "columns": [str(column) for column in df.columns],
        "metadata": json_safe(source_metadata or {}),
    }
    if data_source == "uploaded" and len(df) <= CHECKPOINT_EMBED_MAX_ROWS:
        data_payload["frame_json"] = df.to_json(orient="split", date_format="iso", default_handler=str)
    elif data_source == "uploaded":
        data_payload["frame_json_omitted"] = True
        data_payload["frame_json_omitted_reason"] = f"row_count_above_{CHECKPOINT_EMBED_MAX_ROWS}"

    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tree_schema_version": TREE_SCHEMA_VERSION,
        "work_id": work_id,
        "data": data_payload,
        "target": target,
        "positive_class": json_safe(st.session_state.get(POSITIVE_CLASS_SESSION_KEY)),
        "selected_features": list(selected_features),
        "parameters": json_safe(parameters),
        "auto_parameters": json_safe(auto_parameters),
        "tree_state": {
            "state_key": json_safe(st.session_state.get("state_key")),
            "tree": json_safe(st.session_state.get("tree", {})),
            "next_node_id": json_safe(st.session_state.get("next_node_id", 1)),
            "current_node_id": json_safe(st.session_state.get("current_node_id", 0)),
            "split_history": json_safe(st.session_state.get("split_history", [])),
            "split_action_history": json_safe(st.session_state.get("split_action_history", [])),
            "auto_tree_message": json_safe(st.session_state.get("auto_tree_message", "")),
            "tree_zoom": json_safe(st.session_state.get("tree_zoom")),
        },
        "woe_projects": json_safe(st.session_state.get(WOE_PROJECTS_KEY, {})),
        "ui_state": checkpoint_ui_state(),
    }

    try:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        path = checkpoint_path(work_id)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        temp_path.replace(path)
    except OSError as exc:
        st.session_state["_checkpoint_error"] = str(exc)


def split_branch_conditions(split: dict[str, Any]) -> list[dict[str, Any]]:
    feature = split["feature"]
    split_type = split["split_type"]
    value = split["value"]

    if split_type == "numeric_le":
        threshold = json_safe(value)
        missing_policy = str(split.get("missing_policy") or "right")
        conditions = [
            {"feature": feature, "operator": "<=", "threshold": threshold, "includes_missing": missing_policy == "left"},
            {"feature": feature, "operator": ">", "threshold": threshold, "includes_missing": missing_policy == "right"},
        ]
        if missing_policy == "separate":
            conditions.append({"feature": feature, "operator": "is_missing"})
        return conditions

    if split_type == "category_eq":
        category_value = json_safe(value)
        return [
            {"feature": feature, "operator": "==", "value": category_value},
            {"feature": feature, "operator": "!=", "value": category_value},
        ]

    if split_type in ("numeric_bins", "numeric_manual_bins"):
        thresholds = [json_safe(x) for x in value]
        missing_policy = str(split.get("missing_policy") or "right")
        conditions: list[dict[str, Any]] = []
        previous_threshold: Any | None = None
        for index, threshold in enumerate(thresholds):
            if previous_threshold is None:
                conditions.append(
                    {
                        "feature": feature,
                        "operator": "<=",
                        "threshold": threshold,
                        "includes_missing": missing_policy == "left",
                    }
                )
            else:
                conditions.append(
                    {
                        "feature": feature,
                        "operator": "range",
                        "lower": previous_threshold,
                        "lower_inclusive": False,
                        "upper": threshold,
                        "upper_inclusive": True,
                    }
                )
            previous_threshold = threshold
        conditions.append(
            {
                "feature": feature,
                "operator": ">",
                "threshold": thresholds[-1],
                "includes_missing": missing_policy == "right",
            }
        )
        if missing_policy == "separate":
            conditions.append({"feature": feature, "operator": "is_missing"})
        return conditions

    if split_type == "category_multi":
        kept_values = [json_safe(x) for x in value]
        conditions = [{"feature": feature, "operator": "==", "value": item} for item in kept_values]
        if len(split.get("branch_labels", [])) > len(kept_values):
            conditions.append({"feature": feature, "operator": "not_in", "values": kept_values})
        return conditions

    if split_type == "category_group":
        selected_values = [json_safe(x) for x in value]
        return [
            {"feature": feature, "operator": "in", "values": selected_values},
            {"feature": feature, "operator": "not_in", "values": selected_values},
        ]

    if split_type in ("category_profile_groups", "category_manual_groups"):
        return [
            {"feature": feature, "operator": "in", "values": [json_safe(item) for item in group]}
            for group in value
        ]

    return [{"feature": feature, "operator": "custom", "label": label} for label in split.get("branch_labels", [])]


def export_target_summary(df: pd.DataFrame, target: str, node: dict[str, Any]) -> dict[str, Any]:
    summary = node_summary(df, target, node["row_idx"])
    out = {
        "prediction": json_safe(summary["prediction"]),
        "impurity_name": summary["impurity_label"],
        "impurity": json_safe(summary["impurity"]),
        "class_distribution": [
            {"value": json_safe(value), "count": int(count)}
            for value, count in summary["class_counts"].items()
        ],
    }
    if summary["target_kind"] == "binary":
        out["positive_class"] = json_safe(summary["positive_class"])
        out["default_rate"] = json_safe(summary.get("default_rate", 0.0))
        out["event_count"] = int(summary.get("event_count", 0))
    elif summary["target_kind"] == "regression":
        out["target_mean"] = json_safe(summary.get("target_mean"))
        out["target_std"] = json_safe(summary.get("target_std"))
    return out


def export_branch_path(parent_path: str, branch_label: Any, condition: dict[str, Any]) -> str:
    feature = condition.get("feature")
    if feature is not None:
        return f"{parent_path} -> {feature} {branch_label}"
    return f"{parent_path} -> {branch_label}"


def compute_tree_export_paths(root_node_id: int = 0) -> dict[int, str]:
    tree = st.session_state.tree
    paths = {root_node_id: "root"}
    stack = [root_node_id]
    while stack:
        node_id = stack.pop()
        node = tree[node_id]
        children = get_node_children(node)
        if not children or node["split"] is None:
            continue

        branch_conditions = split_branch_conditions(node["split"])
        parent_path = paths.get(node_id, str(node.get("path", "root")))
        for index, child in enumerate(children):
            condition = branch_conditions[index] if index < len(branch_conditions) else {}
            paths[child["id"]] = export_branch_path(parent_path, child["label"], condition)
            stack.append(child["id"])
    return paths


def export_node(
    df: pd.DataFrame,
    target: str,
    node: dict[str, Any],
    path_override: str | None = None,
) -> dict[str, Any]:
    children = get_node_children(node)
    target_summary = export_target_summary(df, target, node)
    out: dict[str, Any] = {
        "node_id": node["id"],
        "depth": node["depth"],
        "path": path_override if path_override is not None else node["path"],
        "n": len(node["row_idx"]),
        "is_leaf": node["split"] is None,
        "target_summary": target_summary,
    }

    if node["split"] is None:
        out["leaf"] = {"prediction": target_summary["prediction"]}
        out["branches"] = []
        return out

    split = node["split"]
    branch_conditions = split_branch_conditions(split)
    out["split"] = {
        "feature": split["feature"],
        "type": split["split_type"],
        "label": split["label"],
        "value": json_safe(split["value"]),
        "information_gain": json_safe(split["information_gain"]),
        "weighted_impurity": json_safe(split["weighted_entropy"]),
        "branch_count": split["branch_count"],
        "missing_policy": split.get("missing_policy", "right"),
    }
    out["branches"] = [
        {
            "branch_index": index,
            "label": child["label"],
            "condition": branch_conditions[index] if index < len(branch_conditions) else {"label": child["label"]},
            "child_node_id": child["id"],
        }
        for index, child in enumerate(children)
    ]
    return out


def export_nested_node(
    df: pd.DataFrame,
    target: str,
    node_id: int,
    path_map: dict[int, str],
) -> dict[str, Any]:
    node = st.session_state.tree[node_id]
    exported = export_node(df, target, node, path_override=path_map.get(node_id))
    if node["split"] is None:
        return exported

    nested_branches: list[dict[str, Any]] = []
    for branch in exported["branches"]:
        child_id = branch["child_node_id"]
        nested_branch = branch.copy()
        nested_branch["child"] = export_nested_node(df, target, child_id, path_map)
        nested_branches.append(nested_branch)
    exported["branches"] = nested_branches
    return exported


def tree_export(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    target_kind = infer_target_kind(df[target])
    metrics = arrow_safe_dataframe(model_metrics(df, target)).to_dict("records")
    export_paths = compute_tree_export_paths(root_node_id=0)
    nodes = [
        export_node(df, target, node, path_override=export_paths.get(node["id"]))
        for _, node in sorted(st.session_state.tree.items())
    ]
    node_count = len(nodes)
    leaf_count = sum(1 for node in st.session_state.tree.values() if node["split"] is None)
    split_count = node_count - leaf_count
    return {
        "format": "interactive_entropy_decision_tree",
        "format_version": "1.0",
        "runner_contract": {
            "root_node_id": 0,
            "traversal": "Start at root_node_id. If node.is_leaf is true, return node.leaf.prediction. Otherwise evaluate node.branches in order and move to the matching child_node_id.",
            "missing_values": "Numeric missing routing is stored per split with includes_missing=true or an is_missing branch. Categorical missing values are represented as __MISSING__.",
        },
        "target": target,
        "task": target_kind,
        "positive_class": json_safe(choose_positive_class(df[target])) if target_kind == "binary" else None,
        "features": features,
        "parameters": json_safe(parameters),
        "data_rows": int(len(df)),
        "data_columns": [str(column) for column in df.columns],
        "data_fingerprint": dataframe_fingerprint(df),
        "metrics": json_safe(metrics),
        "root_node_id": 0,
        "node_count": node_count,
        "leaf_count": leaf_count,
        "split_count": split_count,
        "nodes": nodes,
        "nodes_by_id": {str(node["node_id"]): node for node in nodes},
        "tree": export_nested_node(df, target, 0, export_paths),
    }


class TreeImportError(ValueError):
    pass


def read_tree_payload_upload(uploaded: Any) -> dict[str, Any]:
    suffix = Path(uploaded.name).suffix.lower()
    raw = uploaded.getvalue()
    if suffix == ".json":
        payload = json.loads(raw.decode("utf-8"))
    elif suffix in (".pkl", ".pickle"):
        payload = pickle.loads(raw)
    else:
        raise TreeImportError("Unsupported tree artifact type. Use JSON or pickle.")
    if not isinstance(payload, dict):
        raise TreeImportError("Tree artifact must contain a dictionary payload.")
    return payload


def category_values_for_condition(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "__MISSING__")


def export_condition_mask(df: pd.DataFrame, row_idx: list[int], condition: dict[str, Any]) -> pd.Series:
    feature = condition.get("feature")
    if feature not in df.columns:
        raise TreeImportError(f"Split feature is missing from current data: {feature}")

    series = df.loc[row_idx, feature]
    operator = condition.get("operator")

    if operator in ("<=", ">"):
        numeric = pd.to_numeric(series, errors="coerce")
        threshold = float(condition["threshold"])
        mask = numeric <= threshold if operator == "<=" else numeric > threshold
        if condition.get("includes_missing"):
            mask = mask | numeric.isna()
        return mask

    if operator == "range":
        numeric = pd.to_numeric(series, errors="coerce")
        lower = float(condition["lower"])
        upper = float(condition["upper"])
        lower_ok = numeric >= lower if condition.get("lower_inclusive") else numeric > lower
        upper_ok = numeric <= upper if condition.get("upper_inclusive") else numeric < upper
        mask = lower_ok & upper_ok
        if condition.get("includes_missing"):
            mask = mask | numeric.isna()
        return mask

    if operator == "is_missing":
        return series.isna()

    values = category_values_for_condition(series)
    if operator == "==":
        return values == condition.get("value")
    if operator == "!=":
        return values != condition.get("value")
    if operator == "in":
        return values.isin(condition.get("values", []))
    if operator == "not_in":
        return ~values.isin(condition.get("values", []))

    raise TreeImportError(f"Unsupported imported branch operator: {operator}")


def split_value_from_export(split: dict[str, Any], branches: list[dict[str, Any]]) -> Any:
    if "value" in split:
        return split["value"]

    split_type = split.get("type") or split.get("split_type")
    conditions = [branch.get("condition", {}) for branch in branches]
    if split_type == "numeric_le":
        for condition in conditions:
            if condition.get("operator") == "<=":
                return condition.get("threshold")
    if split_type == "category_eq":
        for condition in conditions:
            if condition.get("operator") == "==":
                return condition.get("value")
    if split_type == "category_group":
        for condition in conditions:
            if condition.get("operator") == "in":
                return tuple(condition.get("values", []))
    if split_type in ("category_profile_groups", "category_manual_groups"):
        return tuple(tuple(condition.get("values", [])) for condition in conditions)
    if split_type in ("numeric_bins", "numeric_manual_bins"):
        thresholds: list[Any] = []
        for condition in conditions:
            if condition.get("operator") == "<=":
                thresholds.append(condition.get("threshold"))
            elif condition.get("operator") == "range":
                thresholds.append(condition.get("upper"))
        deduped: list[Any] = []
        for threshold in thresholds:
            if threshold not in deduped:
                deduped.append(threshold)
        return tuple(deduped)

    return None


def missing_policy_from_export(split_type: str, split: dict[str, Any], branches: list[dict[str, Any]]) -> str:
    explicit = split.get("missing_policy")
    if explicit in {"left", "right", "separate"}:
        return str(explicit)
    if split_type not in {"numeric_le", "numeric_bins", "numeric_manual_bins"}:
        return "category_level"
    conditions = [branch.get("condition") or {} for branch in branches if isinstance(branch, dict)]
    if any(condition.get("operator") == "is_missing" for condition in conditions):
        return "separate"
    if conditions and conditions[0].get("includes_missing"):
        return "left"
    return "right"


def branch_indices_from_export(
    df: pd.DataFrame,
    row_idx: list[int],
    branches: list[dict[str, Any]],
    node_id: int,
) -> list[tuple[dict[str, Any], list[int]]]:
    matched_indices: list[int] = []
    out: list[tuple[dict[str, Any], list[int]]] = []

    for branch in branches:
        condition = branch.get("condition")
        if not isinstance(condition, dict):
            raise TreeImportError(f"Node {node_id} has a branch without a valid condition.")
        mask = export_condition_mask(df, row_idx, condition)
        child_idx = df.loc[row_idx].loc[mask].index.tolist()
        expected_child = branch.get("child") or {}
        expected_n = expected_child.get("n")
        if expected_n is not None and int(expected_n) != len(child_idx):
            raise TreeImportError(
                f"Node {node_id} branch '{branch.get('label')}' row mismatch: "
                f"artifact has {expected_n}, current data gives {len(child_idx)}."
            )
        matched_indices.extend(child_idx)
        out.append((branch, child_idx))

    if len(set(matched_indices)) != len(matched_indices) or set(matched_indices) != set(row_idx):
        raise TreeImportError(
            f"Node {node_id} branch conditions do not exactly cover the current node rows."
        )
    return out


def rebuild_editable_tree_from_export(
    df: pd.DataFrame,
    target: str,
    payload: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], int, list[int], list[str]]:
    if payload.get("target") != target:
        raise TreeImportError(
            f"Target mismatch: artifact target is {payload.get('target')!r}, current target is {target!r}."
        )

    root = payload.get("tree")
    if not isinstance(root, dict):
        raise TreeImportError("Tree artifact must contain a nested `tree` object.")

    payload_features = [str(feature) for feature in payload.get("features", [])]
    missing_features = [
        feature for feature in payload_features if feature != target and feature not in df.columns
    ]
    if missing_features:
        raise TreeImportError(f"Current data is missing imported feature(s): {', '.join(missing_features)}")

    expected_rows = payload.get("data_rows") or root.get("n")
    if expected_rows is not None and int(expected_rows) != len(df):
        raise TreeImportError(
            f"Row count mismatch: artifact has {expected_rows}, current data has {len(df)}."
        )
    expected_columns = payload.get("data_columns")
    if isinstance(expected_columns, list):
        missing_columns = [str(column) for column in expected_columns if str(column) not in df.columns]
        if missing_columns:
            raise TreeImportError(
                f"Current data is missing imported column(s): {', '.join(missing_columns)}"
            )
    expected_fingerprint = payload.get("data_fingerprint")
    if expected_fingerprint and str(expected_fingerprint) != dataframe_fingerprint(df):
        raise TreeImportError(
            "Data fingerprint mismatch. Load the same dataframe snapshot before importing this tree."
        )

    tree: dict[int, dict[str, Any]] = {}
    split_history: list[int] = []

    def rebuild_node(node_payload: dict[str, Any], row_idx: list[int], depth: int, path: str) -> int:
        node_id = int(node_payload["node_id"])
        if node_id in tree:
            raise TreeImportError(f"Duplicate node id in artifact: {node_id}")

        expected_n = node_payload.get("n")
        if expected_n is not None and int(expected_n) != len(row_idx):
            raise TreeImportError(
                f"Node {node_id} row mismatch: artifact has {expected_n}, current data gives {len(row_idx)}."
            )

        branches = node_payload.get("branches") or []
        is_leaf = bool(node_payload.get("is_leaf")) or not branches
        node = {
            "id": node_id,
            "depth": depth,
            "path": path,
            "row_idx": row_idx,
            "split": None,
            "children": [],
            "left": None,
            "right": None,
        }
        tree[node_id] = node
        if is_leaf:
            return node_id

        split = node_payload.get("split")
        if not isinstance(split, dict):
            raise TreeImportError(f"Node {node_id} is not a leaf but has no split metadata.")
        split_type = split.get("type") or split.get("split_type")
        if not split_type:
            raise TreeImportError(f"Node {node_id} split type is missing.")
        split_feature = split.get("feature")
        if split_feature not in df.columns:
            raise TreeImportError(f"Node {node_id} split feature is missing from current data: {split_feature}")

        branch_rows = branch_indices_from_export(df, row_idx, branches, node_id)
        branch_labels = [str(branch.get("label", "")) for branch, _ in branch_rows]
        node["split"] = {
            "feature": split_feature,
            "split_type": split_type,
            "value": split_value_from_export(split, branches),
            "label": split.get("label", f"{split_feature} {split_type}"),
            "branch_count": len(branches),
            "branch_labels": branch_labels,
            "information_gain": float(split.get("information_gain", 0.0) or 0.0),
            "weighted_entropy": float(split.get("weighted_impurity", split.get("weighted_entropy", 0.0)) or 0.0),
            "missing_policy": missing_policy_from_export(split_type, split, branches),
        }
        split_history.append(node_id)

        children: list[dict[str, Any]] = []
        for branch, child_idx in branch_rows:
            child_payload = branch.get("child")
            if not isinstance(child_payload, dict):
                raise TreeImportError(f"Node {node_id} branch '{branch.get('label')}' has no child payload.")
            condition = branch.get("condition") or {}
            child_path = export_branch_path(path, branch.get("label", ""), condition)
            child_id = rebuild_node(child_payload, child_idx, depth + 1, child_path)
            children.append({"id": child_id, "label": str(branch.get("label", ""))})

        node["children"] = children
        if len(children) == 2:
            node["left"] = children[0]["id"]
            node["right"] = children[1]["id"]
        return node_id

    rebuild_node(root, df.index.tolist(), 0, "root")
    if 0 not in tree:
        raise TreeImportError("Imported tree must contain root node_id 0.")
    return tree, max(tree) + 1, split_history, payload_features


def truncate_text(value: Any, max_len: int = 70) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def tooltip_text(value: Any, max_len: int = GRAPH_TOOLTIP_LIMIT) -> str:
    return truncate_text(value, max_len)


def incoming_branch_full_label(node: dict[str, Any]) -> str:
    if node["id"] == 0:
        return "root"
    for parent in st.session_state.tree.values():
        for child in get_node_children(parent):
            if child["id"] == node["id"]:
                return str(child["label"])
    return ""


def feature_summary_rows(
    candidates: list[SplitCandidate],
    features: list[str],
    selected_features: list[str] | None = None,
    include_zero_gain: bool = False,
    validation_lookup: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected = set(selected_features or [])
    rows: list[dict[str, Any]] = []
    for feature in features:
        feature_candidates = [c for c in candidates if c.feature == feature]
        if feature_candidates:
            best = max(feature_candidates, key=lambda c: candidate_validation_sort_key(c, validation_lookup))
            total_gain = sum(c.information_gain for c in feature_candidates)
            if not include_zero_gain and best.information_gain <= MIN_INFORMATION_GAIN_EPSILON:
                continue
            row = {
                "selected": "yes" if feature in selected else "no",
                "variable": feature,
                "total_information_gain": total_gain,
                "best_information_gain": best.information_gain,
                "candidate_count": len(feature_candidates),
                "best_split": best.label,
                "best_branches": best.branch_count,
            }
            if validation_lookup:
                row.update(candidate_validation_row_values(validation_lookup.get(id(best))))
            rows.append(row)
        else:
            if not include_zero_gain:
                continue
            row = {
                "selected": "yes" if feature in selected else "no",
                "variable": feature,
                "total_information_gain": 0.0,
                "best_information_gain": 0.0,
                "candidate_count": 0,
                "best_split": "",
                "best_branches": 0,
            }
            if validation_lookup:
                row.update(candidate_validation_row_values(None))
            rows.append(row)
    return rows


def ordered_features_by_gain(features: list[str], feature_stats: dict[str, dict[str, Any]]) -> list[str]:
    eligible = [
        feature
        for feature in features
        if feature_stats.get(feature, {}).get("best_information_gain", 0.0) > MIN_INFORMATION_GAIN_EPSILON
    ]
    validation_mode = any("validation_safe" in stats for stats in feature_stats.values())
    if validation_mode:
        safe_features = [
            feature
            for feature in eligible
            if feature_stats.get(feature, {}).get("validation_safe") == "yes"
        ]
        sort_features = safe_features or eligible
        return sorted(
            sort_features,
            key=lambda feature: (
                feature_stats.get(feature, {}).get("validation_safe") == "yes",
                numeric_sort_value(feature_stats.get(feature, {}).get("test_gini_after"), -np.inf),
                numeric_sort_value(feature_stats.get(feature, {}).get("test_gini_delta"), -np.inf),
                -numeric_sort_value(feature_stats.get(feature, {}).get("gini_gap_after"), np.inf),
                feature_stats.get(feature, {}).get("best_information_gain", 0.0),
            ),
            reverse=True,
        )

    return sorted(
        eligible,
        key=lambda feature: (
            feature_stats.get(feature, {}).get("best_information_gain", 0.0),
            feature_stats.get(feature, {}).get("total_information_gain", 0.0),
            feature_stats.get(feature, {}).get("candidate_count", 0),
        ),
        reverse=True,
    )


def node_feature_key(data_key: str, target: str, node_id: int) -> str:
    return f"node_features_v2::{data_key}::{target}::{node_id}"


def ensure_node_features(key: str, available_features: list[str]) -> None:
    if key not in st.session_state:
        st.session_state[key] = available_features.copy()
        return
    st.session_state[key] = [f for f in st.session_state[key] if f in available_features]


def ensure_node_feature(key: str, available_features: list[str], feature_stats: dict[str, dict[str, Any]]) -> str:
    current = st.session_state.get(key)
    if current in available_features:
        return current
    selected = available_features[0] if available_features else ""
    st.session_state[key] = selected
    return selected


def tree_levels() -> list[list[dict[str, Any]]]:
    levels: dict[int, list[dict[str, Any]]] = {}
    for _, node in sorted(st.session_state.tree.items()):
        levels.setdefault(node["depth"], []).append(node)
    return [levels[depth] for depth in sorted(levels)]


def tree_button_label(df: pd.DataFrame, target: str, node: dict[str, Any], data_key: str) -> str:
    summary = node_summary(df, target, node["row_idx"])
    status = "Split" if node["split"] is not None else "Leaf"
    active_text = "ACTIVE | " if node["id"] == st.session_state.current_node_id else ""
    rate_text = ""
    if summary["target_kind"] == "binary":
        rate_text = f" | DR={summary.get('default_rate', 0.0):.3f}"
    elif summary["target_kind"] == "regression":
        rate_text = f" | mean={summary.get('target_mean', 0.0):.3f}"

    if node["split"] is not None:
        split_text = truncate_text(node["split"]["label"], 34)
        return (
            f"{active_text}{status} {node['id']} | {node['split']['feature']} | "
            f"{split_text} | n={summary['n']}"
        )

    selected_feature = st.session_state.get(node_feature_key(data_key, target, node["id"]), "")
    return (
        f"{active_text}{status} {node['id']} | PREDICT={summary['prediction']} | "
        f"n={summary['n']} | {summary['impurity_label']}={summary['impurity']:.3f}"
        f"{rate_text} | var={selected_feature or '-'}"
    )


def incoming_branch_label(node: dict[str, Any]) -> str:
    if node["id"] == 0:
        return "root"
    for parent in st.session_state.tree.values():
        for child in get_node_children(parent):
            if child["id"] == node["id"]:
                return truncate_text(child["label"], 42)
    return ""


def recommended_tree_zoom() -> float:
    leaf_count = len(current_leaves())
    max_children = max((len(get_node_children(node)) for node in st.session_state.tree.values()), default=1)
    if leaf_count >= 10 or max_children >= 8:
        return 0.65
    if leaf_count >= 7 or max_children >= 6:
        return 0.72
    if leaf_count >= 5 or max_children >= 5:
        return 0.78
    if leaf_count >= 3 or max_children >= 3:
        return 0.9
    return 1.0


def ensure_tree_zoom() -> None:
    if "tree_zoom" not in st.session_state:
        st.session_state.tree_zoom = recommended_tree_zoom()
    st.session_state.tree_zoom = float(min(1.35, max(0.65, st.session_state.tree_zoom)))


def graph_node_label(df: pd.DataFrame, target: str, node: dict[str, Any], data_key: str) -> str:
    summary = node_summary(df, target, node["row_idx"])
    status = "SPLIT" if node["split"] is not None else "LEAF"
    lines = [
        f"Node {node['id']} | {status}",
        f"PRED={summary['prediction']}",
        f"n={summary['n']} | {summary['impurity_label']}={summary['impurity']:.3f}",
    ]

    if summary["target_kind"] == "binary":
        lines.append(f"DR={summary.get('default_rate', 0.0):.3f}")
    elif summary["target_kind"] == "regression":
        lines.append(f"mean={summary.get('target_mean', 0.0):.3f}")

    if node["split"] is not None:
        lines.append(f"var={node['split']['feature']}")
        lines.append(f"IG={node['split'].get('information_gain', 0.0):.3f}")
    else:
        selected_feature = st.session_state.get(node_feature_key(data_key, target, node["id"]), "")
        lines.append(f"try_var={selected_feature or '-'}")

    return "\n".join(lines)


def graph_node_tooltip(df: pd.DataFrame, target: str, node: dict[str, Any], data_key: str) -> str:
    return tooltip_text(graph_node_full_text(df, target, node, data_key))


def graph_node_full_text(df: pd.DataFrame, target: str, node: dict[str, Any], data_key: str) -> str:
    summary = node_summary(df, target, node["row_idx"])
    lines = [
        graph_node_label(df, target, node, data_key),
        f"Path: {node['path']}",
    ]
    incoming = incoming_branch_full_label(node)
    if incoming:
        lines.append(f"Incoming branch: {incoming}")
    if node["split"] is not None:
        split = node["split"]
        lines.extend(
            [
                f"Split: {split['label']}",
                f"Feature: {split['feature']}",
                f"Split type: {split['split_type']}",
            ]
        )
    else:
        selected_feature = st.session_state.get(node_feature_key(data_key, target, node["id"]), "")
        if selected_feature:
            lines.append(f"Selected variable to try: {selected_feature}")
    if summary["target_kind"] == "binary":
        lines.append(f"Positive class: {summary.get('positive_class')}")
        lines.append(f"Event count: {summary.get('event_count', 0)}")
    return "\n".join(lines)


def render_interactive_tree_graph(df: pd.DataFrame, target: str, data_key: str) -> int | None:
    ensure_tree_zoom()
    zoom = st.session_state.tree_zoom
    leaf_count = len(current_leaves())
    max_children = max((len(get_node_children(node)) for node in st.session_state.tree.values()), default=1)
    node_font_size = max(8, min(15, int(round(10 * zoom))))
    edge_font_size = max(8, min(13, int(round(9 * zoom))))
    node_margin = max(5, int(round(8 * zoom)))
    node_min_width = max(88, int(round(128 * zoom)))
    node_max_width = max(118, int(round(165 * zoom)))
    edge_label_width = max(12, int(round(18 * zoom)))
    level_separation = max(145, int(round(160 * zoom + max_children * 8)))
    node_spacing = max(node_max_width + 74, int(round(170 * zoom + max_children * 18)))
    tree_spacing = max(node_max_width + 120, int(round(220 * zoom + leaf_count * 16)))

    graph_nodes: list[Node] = []
    graph_edges: list[Edge] = []
    max_depth = 0

    for node_id, node in sorted(st.session_state.tree.items()):
        max_depth = max(max_depth, node["depth"])
        is_leaf = node["split"] is None
        background = "#f8fafc" if is_leaf else "#e8f1ff"
        border = "#94a3b8" if is_leaf else "#4f83cc"
        graph_node = Node(
            id=str(node_id),
            label=graph_node_label(df, target, node, data_key),
            title=graph_node_tooltip(df, target, node, data_key),
            shape="box",
            color={
                "background": background,
                "border": border,
                "highlight": {"background": "#fff2b3", "border": "#c47f00"},
            },
            font={"face": "Arial", "size": node_font_size, "align": "left"},
            margin=node_margin,
            widthConstraint={"minimum": node_min_width, "maximum": node_max_width},
            level=node["depth"],
        )
        graph_nodes.append(graph_node)

        for child in get_node_children(node):
            full_edge_label = str(child["label"])
            graph_edges.append(
                Edge(
                    source=str(node_id),
                    target=str(child["id"]),
                    label=truncate_text(full_edge_label, edge_label_width),
                    title=tooltip_text(full_edge_label),
                    color={"color": "#94a3b8", "highlight": "#c47f00"},
                    font={"size": edge_font_size, "align": "middle"},
                    arrows={"to": {"enabled": True, "scaleFactor": 0.7}},
                    smooth={"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.35},
                )
            )

    graph_height = max(620, min(1100, int(320 + (max_depth + 1) * level_separation)))
    config = Config(
        height=graph_height,
        width=900,
        directed=True,
        physics=False,
        hierarchical=True,
        direction="UD",
        sortMethod="directed",
        levelSeparation=level_separation,
        nodeSpacing=node_spacing,
        treeSpacing=tree_spacing,
        blockShifting=True,
        edgeMinimization=True,
        parentCentralization=True,
        stabilization=True,
        fit=True,
        interaction={
            "navigationButtons": True,
            "keyboard": True,
            "dragNodes": False,
            "dragView": True,
            "zoomView": False,
            "hover": True,
        },
    )
    config.width = "100%"
    selected_node = agraph(graph_nodes, graph_edges, config)
    if selected_node is None:
        return None
    try:
        selected_node_id = int(selected_node)
    except (TypeError, ValueError):
        return None
    if selected_node_id in st.session_state.tree:
        return selected_node_id
    return None


def render_tree_selector(df: pd.DataFrame, target: str, data_key: str) -> None:
    levels = tree_levels()
    for depth, nodes in enumerate(levels):
        st.markdown(f"**Level {depth}**")
        cols = st.columns(len(nodes), gap="small")
        for col, node in zip(cols, nodes):
            with col:
                st.caption(incoming_branch_label(node))
                button_type = "primary" if node["id"] == st.session_state.current_node_id else "secondary"
                if st.button(
                    tree_button_label(df, target, node, data_key),
                    key=f"tree_node_pick_{data_key}_{target}_{node['id']}",
                    width="stretch",
                    type=button_type,
                ):
                    st.session_state.current_node_id = node["id"]
                    st.rerun()
        if depth < len(levels) - 1:
            st.markdown(
                "<div style='text-align:center;color:#888;margin:-0.35rem 0 0.15rem 0;'>"
                "&#8595;</div>",
                unsafe_allow_html=True,
            )


def render_leaf_buttons(df: pd.DataFrame, target: str, data_key: str) -> None:
    leaves = [
        node
        for _, node in sorted(st.session_state.tree.items())
        if node["split"] is None
    ]
    if not leaves:
        return

    cols = st.columns(min(4, len(leaves)))
    for i, node in enumerate(leaves):
        summary = node_summary(df, target, node["row_idx"])
        selected_feature = st.session_state.get(node_feature_key(data_key, target, node["id"]), "")
        rate_text = ""
        if summary["target_kind"] == "binary":
            rate_text = f" | DR={summary.get('default_rate', 0.0):.3f}"
        elif summary["target_kind"] == "regression":
            rate_text = f" | mean={summary.get('target_mean', 0.0):.3f}"
        active_text = "ACTIVE | " if node["id"] == st.session_state.current_node_id else ""
        label = (
            f"{active_text}Leaf {node['id']} | PREDICT={summary['majority']} | "
            f"n={summary['n']} | {summary['impurity_label']}={summary['impurity']:.3f}"
            f"{rate_text} | var={selected_feature or '-'}"
        )
        button_type = "primary" if node["id"] == st.session_state.current_node_id else "secondary"
        if cols[i % len(cols)].button(
            label,
            key=f"tree_leaf_pick_{node['id']}",
            width="stretch",
            type=button_type,
        ):
            st.session_state.current_node_id = node["id"]
            st.rerun()


def current_leaves() -> list[dict[str, Any]]:
    return [
        node
        for _, node in sorted(st.session_state.tree.items())
        if node["split"] is None
    ]


def tree_ui_signature(data_key: str, target: str) -> str:
    nodes: list[dict[str, Any]] = []
    for node_id, node in sorted(st.session_state.get("tree", {}).items()):
        split = node.get("split") or {}
        nodes.append(
            {
                "id": int(node_id),
                "depth": int(node.get("depth", 0)),
                "rows": int(len(node.get("row_idx") or [])),
                "path": str(node.get("path") or ""),
                "split": {
                    "feature": split.get("feature"),
                    "split_type": split.get("split_type"),
                    "value": json_safe(split.get("value")),
                    "label": split.get("label"),
                    "missing_policy": split.get("missing_policy"),
                },
                "children": [
                    {"id": child.get("id"), "label": child.get("label")}
                    for child in get_node_children(node)
                ],
            }
        )
    payload = {
        "schema": TREE_SCHEMA_VERSION,
        "data_key": str(data_key),
        "target": str(target),
        "nodes": nodes,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def cached_tree_total_gain(df: pd.DataFrame, target: str, data_key: str) -> tuple[float, float, float]:
    cache = session_cache(TREE_UI_METRIC_CACHE_KEY)
    key = ("tree_total_gain", id(df), int(len(df)), str(target), tree_ui_signature(data_key, target))
    if key not in cache:
        bounded_cache_set(cache, key, tree_total_gain(df, target), max_items=64)
    return cache[key]


def tree_total_gain(df: pd.DataFrame, target: str) -> tuple[float, float, float]:
    target_kind = infer_target_kind(df[target])
    root_impurity = target_impurity(df[target], target_kind)
    weighted_leaf_impurity = 0.0
    for leaf in current_leaves():
        y_leaf = df.loc[leaf["row_idx"], target]
        weighted_leaf_impurity += (len(y_leaf) / len(df)) * target_impurity(y_leaf, target_kind)
    return root_impurity - weighted_leaf_impurity, root_impurity, weighted_leaf_impurity


def candidate_total_gain_delta(df: pd.DataFrame, candidate: SplitCandidate, row_idx: list[int]) -> float:
    return (len(row_idx) / len(df)) * candidate.information_gain


def candidate_impact_rows(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    candidate: SplitCandidate,
    current_total_gain: float | None = None,
) -> list[dict[str, Any]]:
    if current_total_gain is None:
        current_total_gain, _, _ = tree_total_gain(df, target)
    total_delta = candidate_total_gain_delta(df, candidate, row_idx)
    score_name = split_score_name(df[target])
    return [
        {"metric": f"selected_variable_{score_name}", "value": candidate.information_gain},
        {"metric": f"weighted_tree_{score_name}_delta", "value": total_delta},
        {"metric": f"tree_total_{score_name}_before", "value": current_total_gain},
        {"metric": f"tree_total_{score_name}_after", "value": current_total_gain + total_delta},
        {"metric": "split_branches", "value": candidate.branch_count},
    ]


def candidate_split_summary_rows(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    candidate: SplitCandidate,
    validation_stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    score_name = split_score_name(df[target])
    row = {
        "split_type": candidate.split_type,
        "split": candidate.label,
        "branches": candidate.branch_count,
        score_name: candidate.information_gain,
        "weighted_tree_delta": candidate_total_gain_delta(df, candidate, row_idx),
        "child_weighted_impurity": candidate.weighted_entropy,
    }
    if validation_stats is not None:
        row.update(candidate_validation_row_values(validation_stats))
    return [row]


def candidate_split_summary_column_config(score_name: str) -> dict[str, Any]:
    return {
        score_name: st.column_config.NumberColumn(format="%.6f"),
        "weighted_tree_delta": st.column_config.NumberColumn(format="%.6f"),
        "child_weighted_impurity": st.column_config.NumberColumn(format="%.6f"),
        "train_gini_after": st.column_config.NumberColumn(format="%.6f"),
        "test_gini_after": st.column_config.NumberColumn(format="%.6f"),
        "test_gini_delta": st.column_config.NumberColumn(format="%.6f"),
        "gini_gap_after": st.column_config.NumberColumn(format="%.6f"),
    }


def format_class_counts(counts: dict[Any, int]) -> str:
    return ", ".join(f"{value}: {count}" for value, count in counts.items())


def candidate_branch_detail_rows(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    candidate: SplitCandidate,
) -> list[dict[str, Any]]:
    target_kind = infer_target_kind(df[target])
    score_name = split_score_name(df[target])
    total_rows = len(row_idx)
    total_delta = candidate_total_gain_delta(df, candidate, row_idx)
    branch_indices = split_branch_indices(df, row_idx, candidate)
    rows: list[dict[str, Any]] = []

    for branch_no, (condition, child_idx) in enumerate(branch_indices, start=1):
        summary = node_summary(df, target, child_idx)
        row_share = len(child_idx) / total_rows if total_rows else 0.0
        row: dict[str, Any] = {
            "split_type": candidate.split_type,
            "split": candidate.label,
            "branch": branch_no,
            "condition": condition,
            "rows": len(child_idx),
            "row_share": row_share,
            "prediction": summary["prediction"],
            score_name: candidate.information_gain,
            "weighted_tree_delta": total_delta,
            "branch_impurity": summary["impurity"],
            "branch_weighted_impurity": row_share * summary["impurity"],
        }
        if target_kind == "binary":
            row["positive_class"] = summary["positive_class"]
            row["positive_class_rate"] = summary.get("default_rate", np.nan)
            row["positive_class_count"] = summary.get("event_count", 0)
        elif target_kind == "regression":
            row["target_mean"] = summary.get("target_mean", np.nan)
            row["target_std"] = summary.get("target_std", np.nan)
        else:
            row["class_distribution"] = format_class_counts(summary["class_counts"])
        rows.append(row)

    return rows


def branch_detail_column_config(score_name: str) -> dict[str, Any]:
    return {
        "row_share": st.column_config.NumberColumn(format="%.2f"),
        score_name: st.column_config.NumberColumn(format="%.6f"),
        "weighted_tree_delta": st.column_config.NumberColumn(format="%.6f"),
        "branch_impurity": st.column_config.NumberColumn(format="%.6f"),
        "branch_weighted_impurity": st.column_config.NumberColumn(format="%.6f"),
        "positive_class_rate": st.column_config.NumberColumn(format="%.6f"),
        "target_mean": st.column_config.NumberColumn(format="%.6f"),
        "target_std": st.column_config.NumberColumn(format="%.6f"),
    }


def parse_threshold_text(text: str) -> list[float]:
    thresholds: list[float] = []
    normalized = text.replace(";", ",").replace("\n", ",")
    for raw in normalized.split(","):
        raw = raw.strip()
        if not raw:
            continue
        thresholds.append(float(raw))
    return sorted({float(x) for x in thresholds})


def manual_numeric_branch_rows(
    df: pd.DataFrame,
    row_idx: list[int],
    feature: str,
    thresholds: list[float],
    missing_policy: str = "right",
) -> list[dict[str, Any]]:
    thresholds = sorted({float(x) for x in thresholds})
    if not thresholds:
        return []

    frame = df.loc[row_idx, [feature]]
    numeric = pd.to_numeric(frame[feature], errors="coerce")
    return [
        {"branch": label, "rows": int(mask.sum())}
        for mask, label in numeric_masks_and_labels(numeric, thresholds, missing_policy)
    ]


def safe_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def binary_auc(y_true: pd.Series, scores: pd.Series) -> float | None:
    frame = pd.DataFrame({"y": y_true, "score": scores}).dropna()
    if frame.empty:
        return None
    positives = int(frame["y"].sum())
    negatives = len(frame) - positives
    if positives == 0 or negatives == 0:
        return None

    sorted_scores = frame["score"].to_numpy()
    order = np.argsort(sorted_scores, kind="mergesort")
    sorted_scores = sorted_scores[order]
    ranks = np.empty(len(sorted_scores), dtype=float)

    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[i:j] = (i + 1 + j) / 2
        i = j

    original_ranks = np.empty(len(sorted_scores), dtype=float)
    original_ranks[order] = ranks
    pos_ranks = original_ranks[frame["y"].to_numpy(dtype=bool)]
    return float((pos_ranks.sum() - positives * (positives + 1) / 2) / (positives * negatives))


def tree_predictions(df: pd.DataFrame, target: str) -> pd.DataFrame:
    target_kind = infer_target_kind(df[target])
    positive_class = choose_positive_class(df[target]) if target_kind == "binary" else None
    out = pd.DataFrame(index=df.index)
    out["leaf_id"] = np.nan

    if target_kind == "regression":
        out["prediction"] = np.nan
    elif target_kind == "binary":
        out["prediction"] = None
        out["positive_rate"] = np.nan
    else:
        out["prediction"] = None

    for leaf in current_leaves():
        idx = leaf["row_idx"]
        y_leaf = df.loc[idx, target]
        summary = node_summary(df, target, idx)
        out.loc[idx, "leaf_id"] = leaf["id"]
        if target_kind == "regression":
            out.loc[idx, "prediction"] = summary.get("target_mean", np.nan)
        elif target_kind == "binary":
            rate = float((y_leaf == positive_class).mean()) if positive_class is not None else np.nan
            out.loc[idx, "positive_rate"] = rate
            out.loc[idx, "prediction"] = positive_class if rate >= 0.5 else [
                cls for cls in df[target].dropna().unique() if cls != positive_class
            ][0]
        else:
            out.loc[idx, "prediction"] = summary["prediction"]

    return out


def tree_predictions_for_dataframe(train_df: pd.DataFrame, eval_df: pd.DataFrame, target: str) -> pd.DataFrame:
    target_kind = infer_target_kind(train_df[target])
    positive_class = choose_positive_class(train_df[target]) if target_kind == "binary" else None
    out = pd.DataFrame(index=eval_df.index)
    out["leaf_id"] = np.nan

    if target_kind == "regression":
        out["prediction"] = np.nan
    elif target_kind == "binary":
        out["prediction"] = None
        out["positive_rate"] = np.nan
    else:
        out["prediction"] = None

    pending: list[tuple[int, list[int]]] = [(0, eval_df.index.tolist())]
    while pending:
        node_id, idx = pending.pop()
        if not idx or node_id not in st.session_state.tree:
            continue
        node = st.session_state.tree[node_id]
        children = get_node_children(node)
        if node["split"] is None or not children:
            summary = node_summary(train_df, target, node["row_idx"])
            out.loc[idx, "leaf_id"] = node["id"]
            if target_kind == "regression":
                out.loc[idx, "prediction"] = summary.get("target_mean", np.nan)
            elif target_kind == "binary":
                rate = float(summary.get("default_rate", np.nan))
                out.loc[idx, "positive_rate"] = rate
                negative_classes = [
                    cls for cls in train_df[target].dropna().unique() if not class_values_equal(cls, positive_class)
                ]
                out.loc[idx, "prediction"] = positive_class if rate >= 0.5 else (
                    negative_classes[0] if negative_classes else None
                )
            else:
                out.loc[idx, "prediction"] = summary["prediction"]
            continue

        conditions = split_branch_conditions(node["split"])
        remaining = pd.Index(idx)
        for child_index, child in enumerate(children):
            if child_index == len(children) - 1:
                child_idx = remaining.tolist()
            else:
                condition = conditions[child_index] if child_index < len(conditions) else {}
                mask = export_condition_mask(eval_df, remaining.tolist(), condition)
                child_idx = eval_df.loc[remaining].loc[mask].index.tolist()
                remaining = remaining.difference(child_idx, sort=False)
            pending.append((child["id"], child_idx))

    return out


def evaluated_tree_total_gain(eval_df: pd.DataFrame, target: str, predictions: pd.DataFrame) -> tuple[float, float, float]:
    target_kind = infer_target_kind(eval_df[target])
    root_impurity = target_impurity(eval_df[target], target_kind)
    weighted_leaf_impurity = 0.0
    valid = predictions["leaf_id"].notna()
    if not valid.any():
        return 0.0, root_impurity, root_impurity
    for _, group in predictions[valid].groupby("leaf_id"):
        y_leaf = eval_df.loc[group.index, target]
        weighted_leaf_impurity += (len(y_leaf) / len(eval_df)) * target_impurity(y_leaf, target_kind)
    return root_impurity - weighted_leaf_impurity, root_impurity, weighted_leaf_impurity


def evaluation_model_metrics(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target: str,
    dataset_name: str,
) -> pd.DataFrame:
    target_kind = infer_target_kind(train_df[target])
    preds = tree_predictions_for_dataframe(train_df, eval_df, target)
    y = eval_df[target]
    valid = preds["prediction"].notna() & y.notna()
    rows: list[dict[str, Any]] = [{"dataset": dataset_name, "metric": "rows", "value": len(eval_df)}]
    rows.append({"dataset": dataset_name, "metric": "scored_rows", "value": int(valid.sum())})

    if target_kind == "binary":
        positive_class = choose_positive_class(train_df[target])
        y_binary = (y == positive_class).astype(int)
        auc = binary_auc(y_binary[valid], preds.loc[valid, "positive_rate"])
        accuracy = float((preds.loc[valid, "prediction"] == y[valid]).mean()) if valid.any() else None
        rows.append({"dataset": dataset_name, "metric": "target_type", "value": "binary"})
        rows.append({"dataset": dataset_name, "metric": "positive_class", "value": positive_class})
        rows.append({"dataset": dataset_name, "metric": "default_rate", "value": float(y_binary.mean())})
        rows.append({"dataset": dataset_name, "metric": "auc", "value": auc})
        rows.append({"dataset": dataset_name, "metric": "gini", "value": None if auc is None else 2 * auc - 1})
        rows.append({"dataset": dataset_name, "metric": "accuracy", "value": accuracy})
    elif target_kind == "regression":
        y_num = pd.to_numeric(y, errors="coerce")
        pred_num = pd.to_numeric(preds["prediction"], errors="coerce")
        mask = y_num.notna() & pred_num.notna()
        residual = y_num[mask] - pred_num[mask]
        baseline = y_num[mask] - y_num[mask].mean()
        sse = float((residual**2).sum())
        sst = float((baseline**2).sum())
        rows.append({"dataset": dataset_name, "metric": "target_type", "value": "regression"})
        rows.append({"dataset": dataset_name, "metric": "rmse", "value": float(np.sqrt((residual**2).mean())) if len(residual) else None})
        rows.append({"dataset": dataset_name, "metric": "mae", "value": float(residual.abs().mean()) if len(residual) else None})
        rows.append({"dataset": dataset_name, "metric": "r2", "value": None if sst == 0 else 1 - sse / sst})
    else:
        accuracy = float((preds.loc[valid, "prediction"] == y[valid]).mean()) if valid.any() else None
        rows.append({"dataset": dataset_name, "metric": "target_type", "value": "classification"})
        rows.append({"dataset": dataset_name, "metric": "accuracy", "value": accuracy})

    total_gain, root_impurity, leaf_impurity = evaluated_tree_total_gain(eval_df, target, preds)
    rows.append({"dataset": dataset_name, "metric": "tree_total_gain", "value": total_gain})
    rows.append({"dataset": dataset_name, "metric": "root_impurity", "value": root_impurity})
    rows.append({"dataset": dataset_name, "metric": "weighted_leaf_impurity", "value": leaf_impurity})
    rows.append({"dataset": dataset_name, "metric": "leaf_count", "value": len(current_leaves())})
    return pd.DataFrame(rows)


def numeric_sort_value(value: Any, default: float) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def gini_gap(train_gini: float | None, test_gini: float | None) -> float | None:
    if train_gini is None or test_gini is None:
        return None
    return abs(float(train_gini) - float(test_gini))


def binary_gini_from_predictions(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target: str,
    predictions: pd.DataFrame,
) -> float | None:
    if infer_target_kind(train_df[target]) != "binary" or "positive_rate" not in predictions.columns:
        return None
    positive_class = choose_positive_class(train_df[target])
    y_binary = (eval_df[target] == positive_class).astype(int)
    valid = predictions["positive_rate"].notna() & eval_df[target].notna()
    auc = binary_auc(y_binary[valid], predictions.loc[valid, "positive_rate"])
    return None if auc is None else float(2 * auc - 1)


def tree_predictions_with_candidate(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target: str,
    node_id: int,
    candidate: SplitCandidate,
    baseline_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    predictions = (
        baseline_predictions.copy()
        if baseline_predictions is not None
        else tree_predictions_for_dataframe(train_df, eval_df, target)
    )
    predictions["leaf_id"] = predictions["leaf_id"].astype("object")
    target_idx = predictions.index[predictions["leaf_id"] == node_id].tolist()
    if not target_idx or node_id not in st.session_state.tree:
        return predictions

    target_kind = infer_target_kind(train_df[target])
    positive_class = choose_positive_class(train_df[target]) if target_kind == "binary" else None
    negative_classes = [
        cls for cls in train_df[target].dropna().unique() if not class_values_equal(cls, positive_class)
    ]
    negative_class = negative_classes[0] if negative_classes else None
    train_node_idx = st.session_state.tree[node_id]["row_idx"]
    train_branches = split_branch_indices(train_df, train_node_idx, candidate)
    eval_branches = split_branch_indices(eval_df, target_idx, candidate)

    for branch_no, ((_, train_child_idx), (_, eval_child_idx)) in enumerate(
        zip(train_branches, eval_branches),
        start=1,
    ):
        if not eval_child_idx:
            continue
        summary = node_summary(train_df, target, train_child_idx)
        virtual_leaf_id = f"{node_id}:{branch_no}"
        predictions.loc[eval_child_idx, "leaf_id"] = virtual_leaf_id
        if target_kind == "regression":
            predictions.loc[eval_child_idx, "prediction"] = summary.get("target_mean", np.nan)
        elif target_kind == "binary":
            rate = float(summary.get("default_rate", np.nan))
            predictions.loc[eval_child_idx, "positive_rate"] = rate
            predictions.loc[eval_child_idx, "prediction"] = positive_class if rate >= 0.5 else negative_class
        else:
            predictions.loc[eval_child_idx, "prediction"] = summary["prediction"]

    return predictions


def candidate_validation_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    node_id: int,
    candidate: SplitCandidate,
    max_gini_gap: float,
    baseline_train_predictions: pd.DataFrame | None = None,
    baseline_test_predictions: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    if test_df is None or infer_target_kind(train_df[target]) != "binary":
        return None

    train_before_preds = (
        baseline_train_predictions
        if baseline_train_predictions is not None
        else tree_predictions_for_dataframe(train_df, train_df, target)
    )
    test_before_preds = (
        baseline_test_predictions
        if baseline_test_predictions is not None
        else tree_predictions_for_dataframe(train_df, test_df, target)
    )
    train_after_preds = tree_predictions_with_candidate(
        train_df,
        train_df,
        target,
        node_id,
        candidate,
        baseline_predictions=train_before_preds,
    )
    test_after_preds = tree_predictions_with_candidate(
        train_df,
        test_df,
        target,
        node_id,
        candidate,
        baseline_predictions=test_before_preds,
    )

    train_gini_before = binary_gini_from_predictions(train_df, train_df, target, train_before_preds)
    test_gini_before = binary_gini_from_predictions(train_df, test_df, target, test_before_preds)
    train_gini_after = binary_gini_from_predictions(train_df, train_df, target, train_after_preds)
    test_gini_after = binary_gini_from_predictions(train_df, test_df, target, test_after_preds)
    gap_before = gini_gap(train_gini_before, test_gini_before)
    gap_after = gini_gap(train_gini_after, test_gini_after)
    test_delta = None if test_gini_before is None or test_gini_after is None else test_gini_after - test_gini_before
    train_delta = None if train_gini_before is None or train_gini_after is None else train_gini_after - train_gini_before

    if gap_after is None or test_delta is None:
        validation_safe = True
    else:
        allowed_gap = max(float(max_gini_gap), float(gap_before or 0.0))
        validation_safe = (
            test_delta >= -MIN_INFORMATION_GAIN_EPSILON
            and gap_after <= allowed_gap + MIN_INFORMATION_GAIN_EPSILON
        )

    return {
        "train_gini_before": train_gini_before,
        "test_gini_before": test_gini_before,
        "train_gini_after": train_gini_after,
        "test_gini_after": test_gini_after,
        "train_gini_delta": train_delta,
        "test_gini_delta": test_delta,
        "gini_gap_before": gap_before,
        "gini_gap_after": gap_after,
        "max_gini_gap": max_gini_gap,
        "validation_safe": bool(validation_safe),
    }


def candidate_validation_row_values(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {
            "validation_safe": "",
            "train_gini_after": None,
            "test_gini_after": None,
            "test_gini_delta": None,
            "gini_gap_after": None,
        }
    return {
        "validation_safe": "yes" if stats.get("validation_safe") else "no",
        "train_gini_after": stats.get("train_gini_after"),
        "test_gini_after": stats.get("test_gini_after"),
        "test_gini_delta": stats.get("test_gini_delta"),
        "gini_gap_after": stats.get("gini_gap_after"),
    }


def candidate_validation_sort_key(
    candidate: SplitCandidate,
    validation_lookup: dict[int, dict[str, Any]] | None = None,
) -> tuple[Any, ...]:
    stats = validation_lookup.get(id(candidate)) if validation_lookup else None
    if not stats:
        return (
            0,
            -np.inf,
            -np.inf,
            -np.inf,
            candidate.information_gain,
        )
    return (
        1 if stats.get("validation_safe") else 0,
        numeric_sort_value(stats.get("test_gini_after"), -np.inf),
        numeric_sort_value(stats.get("test_gini_delta"), -np.inf),
        -numeric_sort_value(stats.get("gini_gap_after"), np.inf),
        candidate.information_gain,
    )


def candidate_passes_validation(
    candidate: SplitCandidate,
    validation_lookup: dict[int, dict[str, Any]] | None,
) -> bool:
    if not validation_lookup:
        return True
    stats = validation_lookup.get(id(candidate))
    return bool(stats and stats.get("validation_safe"))


def candidates_selected_for_validation(
    candidates: list[SplitCandidate],
    limit: int | None,
) -> list[SplitCandidate]:
    if limit is None:
        return [candidate for candidate in candidates if candidate.information_gain > MIN_INFORMATION_GAIN_EPSILON]

    selected: list[SplitCandidate] = []
    covered_features: set[str] = set()
    for candidate in candidates:
        if candidate.information_gain <= MIN_INFORMATION_GAIN_EPSILON:
            continue
        if len(selected) < limit or candidate.feature not in covered_features:
            selected.append(candidate)
            covered_features.add(candidate.feature)
    return selected


def build_candidate_validation_lookup(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    node_id: int,
    candidates: list[SplitCandidate],
    max_gini_gap: float,
    candidate_limit: int | None = None,
) -> dict[int, dict[str, Any]]:
    if test_df is None or infer_target_kind(train_df[target]) != "binary":
        return {}

    baseline_train_predictions = tree_predictions_for_dataframe(train_df, train_df, target)
    baseline_test_predictions = tree_predictions_for_dataframe(train_df, test_df, target)
    lookup: dict[int, dict[str, Any]] = {}
    for candidate in candidates_selected_for_validation(candidates, candidate_limit):
        stats = candidate_validation_stats(
            train_df=train_df,
            test_df=test_df,
            target=target,
            node_id=node_id,
            candidate=candidate,
            max_gini_gap=max_gini_gap,
            baseline_train_predictions=baseline_train_predictions,
            baseline_test_predictions=baseline_test_predictions,
        )
        if stats is not None:
            lookup[id(candidate)] = stats
    return lookup


def model_performance_wide_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or not {"dataset", "metric", "value"}.issubset(metrics.columns):
        return metrics

    dataset_order = list(dict.fromkeys(metrics["dataset"].astype(str).tolist()))
    metric_order = list(dict.fromkeys(metrics["metric"].astype(str).tolist()))
    rows: list[dict[str, Any]] = []
    for metric in metric_order:
        row: dict[str, Any] = {"metric": metric}
        metric_rows = metrics[metrics["metric"].astype(str) == metric]
        for dataset in dataset_order:
            values = metric_rows.loc[metric_rows["dataset"].astype(str) == dataset, "value"]
            row[dataset] = values.iloc[0] if not values.empty else None
        rows.append(row)
    return pd.DataFrame(rows, columns=["metric", *dataset_order])


def model_metrics(df: pd.DataFrame, target: str) -> pd.DataFrame:
    metrics = evaluation_model_metrics(df, df, target, "Train")
    return metrics.drop(columns=["dataset"])


def leaf_performance_rows(
    train_df: pd.DataFrame,
    target: str,
    data_key: str,
    eval_df: pd.DataFrame | None = None,
    dataset_name: str = "Train",
) -> list[dict[str, Any]]:
    target_kind = infer_target_kind(train_df[target])
    positive_class = choose_positive_class(train_df[target]) if target_kind == "binary" else None
    measurement_df = eval_df if eval_df is not None else train_df
    eval_predictions = None if eval_df is None else tree_predictions_for_dataframe(train_df, measurement_df, target)
    rows: list[dict[str, Any]] = []
    for leaf in current_leaves():
        if eval_predictions is None:
            measurement_idx = leaf["row_idx"]
        else:
            measurement_idx = eval_predictions.index[eval_predictions["leaf_id"] == leaf["id"]].tolist()
        summary = target_series_summary(
            measurement_df.loc[measurement_idx, target],
            target_kind,
            positive_class,
        )
        train_summary = node_summary(train_df, target, leaf["row_idx"])
        selected_feature = st.session_state.get(node_feature_key(data_key, target, leaf["id"]), "")
        row = {
            "dataset": dataset_name,
            "leaf": leaf["id"],
            "selected_variable": selected_feature,
            "n": summary["n"],
            "predict": train_summary["prediction"],
            "impurity": summary["impurity"],
            "path": leaf["path"],
        }
        if target_kind == "binary":
            row["positive_class"] = summary["positive_class"]
            row["default_rate"] = summary.get("default_rate", 0.0)
            row["event_count"] = summary.get("event_count", 0)
        elif target_kind == "regression":
            row["target_mean"] = summary.get("target_mean")
            row["target_std"] = summary.get("target_std")
        rows.append(row)
    return rows


def cached_leaf_performance_rows(
    train_df: pd.DataFrame,
    target: str,
    data_key: str,
    eval_df: pd.DataFrame | None = None,
    dataset_name: str = "Train",
) -> list[dict[str, Any]]:
    measurement_df = eval_df if eval_df is not None else train_df
    cache = session_cache(TREE_UI_METRIC_CACHE_KEY)
    key = (
        "leaf_performance_rows",
        id(train_df),
        int(len(train_df)),
        id(measurement_df),
        int(len(measurement_df)),
        str(target),
        str(dataset_name),
        tree_ui_signature(data_key, target),
    )
    if key not in cache:
        bounded_cache_set(
            cache,
            key,
            leaf_performance_rows(train_df, target, data_key, eval_df=eval_df, dataset_name=dataset_name),
            max_items=64,
        )
    return [dict(row) for row in cache[key]]


def cached_model_performance_wide_table(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    data_key: str,
) -> pd.DataFrame:
    cache = session_cache(TREE_UI_METRIC_CACHE_KEY)
    key = (
        "model_performance_wide",
        id(train_df),
        int(len(train_df)),
        id(test_df) if test_df is not None else None,
        int(len(test_df)) if test_df is not None else 0,
        str(target),
        tree_ui_signature(data_key, target),
    )
    if key not in cache:
        train_metrics = model_metrics(train_df, target)
        train_metrics.insert(0, "dataset", "Train")
        metric_frames = [train_metrics]
        if test_df is not None:
            metric_frames.append(evaluation_model_metrics(train_df, test_df, target, "Test"))
        performance_metrics = pd.concat(metric_frames, ignore_index=True)
        bounded_cache_set(cache, key, model_performance_wide_table(performance_metrics), max_items=32)
    return cache[key].copy()


def main() -> None:
    st.set_page_config(page_title="Interactive entropy tree", layout="wide")
    st.title("Interactive entropy decision tree")

    work_id = ensure_work_id()
    checkpoint = load_work_checkpoint(work_id)
    restore_checkpoint_ui_state(checkpoint)
    if (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("woe_projects"), dict)
        and WOE_PROJECTS_KEY not in st.session_state
    ):
        st.session_state[WOE_PROJECTS_KEY] = checkpoint["woe_projects"]

    query_session = load_session_dataframe_from_query()
    query_data_id = query_session[2] if query_session is not None else None

    source_options = ["Session DataFrame", "CSV / Excel Upload", "SQL", "Demo"]
    default_source = "Session DataFrame" if query_session is not None else "Demo"
    remembered_source = st.session_state.get("data_source_choice")
    if query_data_id and st.session_state.get("_last_query_data_id") != query_data_id:
        st.session_state.pop("data_source_choice", None)
        st.session_state.pop(APPLIED_DATA_CONTEXT_KEY, None)
        remembered_source = "Session DataFrame"
        st.session_state["_last_query_data_id"] = query_data_id
    elif remembered_source == "CSV Upload":
        st.session_state.pop("data_source_choice", None)
        remembered_source = "CSV / Excel Upload"
    elif remembered_source not in source_options:
        st.session_state.pop("data_source_choice", None)
        remembered_source = default_source
    checkpoint_ui_state = checkpoint.get("ui_state") if isinstance(checkpoint, dict) else {}
    if (
        st.session_state.get(APPLIED_DATA_CONTEXT_KEY) is None
        and isinstance(checkpoint_ui_state, dict)
        and checkpoint_ui_state.get("workspace_mode") in {"Tree Builder", "WOE Binning"}
    ):
        restored_context = restore_applied_context_from_checkpoint(checkpoint, query_session)
        if restored_context is not None:
            st.session_state[APPLIED_DATA_CONTEXT_KEY] = restored_context
    workspace_options = ["Data Setup", "Tree Builder", "WOE Binning"]
    workspace_default = "Tree Builder" if st.session_state.get(APPLIED_DATA_CONTEXT_KEY) is not None else "Data Setup"
    workspace_mode = st.sidebar.radio(
        "Workspace",
        workspace_options,
        index=workspace_options.index(st.session_state.get("workspace_mode", workspace_default))
        if st.session_state.get("workspace_mode", workspace_default) in workspace_options
        else workspace_options.index(workspace_default),
        key="workspace_mode",
    )
    show_data_setup = workspace_mode == "Data Setup" or st.session_state.get(APPLIED_DATA_CONTEXT_KEY) is None

    train_source = None
    if show_data_setup:
        train_panel = st.sidebar.expander("Data source", expanded=True)
        source_choice = train_panel.radio(
            "Train data source",
            source_options,
            index=source_options.index(str(remembered_source)),
            key="data_source_choice",
        )

        train_source = render_dataframe_source_loader(
            train_panel,
            role="train",
            source_choice=source_choice,
            query_session=query_session,
            checkpoint=checkpoint,
            allow_checkpoint_restore=True,
            update_query_params=True,
        )
    else:
        train_panel = st.sidebar.expander("Active dataset", expanded=True)
    st.sidebar.caption(f"Autosave work id: {work_id}")
    if st.session_state.get("_checkpoint_error"):
        st.sidebar.warning(f"Autosave failed: {st.session_state['_checkpoint_error']}")

    draft_context: dict[str, Any] | None = None
    apply_clicked = False
    if show_data_setup and train_source is None:
        train_panel.info("Configure a draft train source, then click Apply data setup.")
    elif show_data_setup:
        draft_source_df = train_source.df
        draft_uploaded_name = train_source.name
        draft_source_metadata = dict(train_source.metadata)
        draft_source_data_id = train_source.data_id
        draft_source_data_key = train_source.data_key
        draft_data_source = train_source.source
        restored_upload = train_source.restored_upload
        st.subheader("Data setup")
        st.caption("Configure the active dataset here. Tree rendering and split ranking stay idle on this page.")
        data_setup_form = st.container()
        with data_setup_form:

            data_setup_form.caption(
                f"Draft train source rows: {len(draft_source_df):,} | Columns: {len(draft_source_df.columns):,}"
            )
            if restored_upload and draft_uploaded_name is not None:
                data_setup_form.caption(f"Restored uploaded data: {draft_uploaded_name}")
            if draft_source_data_id:
                data_setup_form.caption(f"Draft data id: {draft_source_data_id}")

            checkpoint_data = checkpoint.get("data") if isinstance(checkpoint, dict) else {}
            checkpoint_data_key = checkpoint_data.get("data_key") if isinstance(checkpoint_data, dict) else None
            draft_checkpoint_source_match = checkpoint_data_key == draft_source_data_key or str(
                checkpoint_data_key
            ).startswith(f"{draft_source_data_key}:")
            saved_target = checkpoint.get("target") if isinstance(checkpoint, dict) and draft_checkpoint_source_match else None
            draft_saved_parameters = (
                checkpoint.get("parameters")
                if isinstance(checkpoint, dict) and draft_checkpoint_source_match
                else {}
            )
            if not isinstance(draft_saved_parameters, dict):
                draft_saved_parameters = {}
            source_target = source_metadata_value(draft_source_metadata, "target")
            target_options = draft_source_df.columns.tolist()
            default_target_index = (
                target_options.index("risk_flag") if "risk_flag" in target_options else len(target_options) - 1
            )
            if saved_target in target_options:
                default_target_index = target_options.index(saved_target)
            elif source_target in target_options:
                default_target_index = target_options.index(source_target)
            draft_target = data_setup_form.selectbox(
                "Target",
                options=target_options,
                index=default_target_index,
            )
            draft_target_kind = infer_target_kind(draft_source_df[draft_target])
            draft_positive_class = None
            if draft_target_kind == "binary":
                positive_class_options = target_unique_values(draft_source_df[draft_target])
                positive_class_key = f"positive_class::{draft_source_data_key}::{draft_target}"
                checkpoint_positive_class = (
                    checkpoint.get("positive_class")
                    if isinstance(checkpoint, dict) and draft_checkpoint_source_match
                    else None
                )
                remembered_positive_class = st.session_state.get(positive_class_key, checkpoint_positive_class)
                default_positive_class = choose_positive_class(
                    draft_source_df[draft_target],
                    preferred=remembered_positive_class,
                    use_session_default=False,
                )
                draft_positive_class = data_setup_form.selectbox(
                    "Positive class",
                    options=positive_class_options,
                    index=class_option_index(positive_class_options, default_positive_class),
                    key=positive_class_key,
                    format_func=lambda value: str(value),
                )

            default_features = [c for c in draft_source_df.columns if c != draft_target]
            checkpoint_features = (
                checkpoint.get("selected_features")
                if isinstance(checkpoint, dict) and draft_checkpoint_source_match
                else None
            )
            source_features = source_metadata_value(draft_source_metadata, "features")
            if isinstance(checkpoint_features, list):
                default_selected_features = [
                    str(feature) for feature in checkpoint_features if str(feature) in default_features
                ]
            elif isinstance(source_features, list):
                default_selected_features = [str(feature) for feature in source_features if str(feature) in default_features]
            else:
                default_selected_features = default_features
            draft_features, draft_split_variable_limit = render_feature_manager_fragment(
                draft_source_df,
                default_features,
                default_selected_features,
                data_key=draft_source_data_key,
                target=draft_target,
                saved_parameters=draft_saved_parameters,
            )

            draft_separate_test_df: pd.DataFrame | None = None
            sample_metadata: dict[str, Any] | None = None
            validation_metadata: dict[str, Any] | None = None
            data_sample_tab, validation_tab = data_setup_form.tabs(["Sample", "Test / validation"])

            default_stratify_columns = [draft_target] if draft_target_kind != "regression" else []
            with data_sample_tab:
                sample_enabled = st.checkbox(
                    "Use sampled working data",
                    value=False,
                    key="data_sample_enabled",
                    help="Samples the loaded train source before tree building or train/test splitting.",
                )
                sample_col1, sample_col2 = st.columns(2)
                sample_rows_input = sample_col1.number_input(
                    "Sample rows",
                    min_value=1,
                    max_value=max(1, len(draft_source_df)),
                    value=min(len(draft_source_df), DEFAULT_DATA_SAMPLE_ROWS),
                    step=10_000,
                    format="%d",
                    key="data_sample_rows",
                )
                sample_seed_input = sample_col2.number_input(
                    "Sample seed",
                    value=CANDIDATE_RANDOM_STATE,
                    step=1,
                    key="data_sample_seed",
                )
                sample_stratify_columns = st.multiselect(
                    "Sample stratify columns",
                    options=draft_source_df.columns.tolist(),
                    default=default_stratify_columns,
                    key="data_sample_stratify_columns",
                    help=(
                        "Categorical columns are used as-is. Numeric columns are converted into quantile bins "
                        "before the stratified sample is drawn."
                    ),
                )
                sample_numeric_bins_input = st.number_input(
                    "Numeric stratify bins",
                    min_value=2,
                    max_value=50,
                    value=DEFAULT_STRATIFY_NUMERIC_BINS,
                    step=1,
                    format="%d",
                    key="data_sample_stratify_numeric_bins",
                )
                sample_rows = min(
                    len(draft_source_df),
                    safe_int(sample_rows_input, default=len(draft_source_df), minimum=1),
                )
                sample_seed = int(sample_seed_input)
                sample_numeric_bins = safe_int(
                    sample_numeric_bins_input,
                    default=DEFAULT_STRATIFY_NUMERIC_BINS,
                    minimum=2,
                )
                sample_stratify_columns = normalize_stratify_columns(draft_source_df, sample_stratify_columns)
                if sample_enabled and sample_rows < len(draft_source_df):
                    sample_metadata = {
                        "enabled": True,
                        "source_rows": int(len(draft_source_df)),
                        "sample_rows": int(sample_rows),
                        "random_state": sample_seed,
                        "stratify_columns": list(sample_stratify_columns),
                        "numeric_bins": sample_numeric_bins,
                    }
                    draft_source_metadata = {**draft_source_metadata, "sample": sample_metadata}
                else:
                    draft_source_metadata = {key: value for key, value in draft_source_metadata.items() if key != "sample"}
                st.caption("Sample settings are staged only; sampling runs after Apply data setup.")

            validation_mode = validation_tab.radio(
                "Test / validation source",
                ["No test data", "Split train data", "Separate data source"],
                key="validation_source_mode",
                help="Train datasini kullan, ayni tabloyu train/test bol veya ayri bir test kaynagi yukle.",
            )
            draft_error: str | None = None
            if validation_mode == "Split train data":
                validation_tab.caption("Split settings below are used only when Test / validation source is Split train data.")
                split_col1, split_col2 = validation_tab.columns(2)
                test_fraction = split_col1.slider(
                    "Test share",
                    min_value=0.05,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    key="validation_test_share",
                )
                split_seed = split_col2.number_input(
                    "Split seed",
                    value=CANDIDATE_RANDOM_STATE,
                    step=1,
                    key="validation_split_seed",
                )
                split_stratify_columns = validation_tab.multiselect(
                    "Split stratify columns",
                    options=draft_source_df.columns.tolist(),
                    default=default_stratify_columns,
                    key="validation_stratify_columns",
                    help=(
                        "Categorical columns are used as-is. Numeric columns are converted into quantile bins "
                        "before the train/test split is drawn."
                    ),
                )
                split_numeric_bins_input = validation_tab.number_input(
                    "Split numeric stratify bins",
                    min_value=2,
                    max_value=50,
                    value=DEFAULT_STRATIFY_NUMERIC_BINS,
                    step=1,
                    format="%d",
                    key="validation_stratify_numeric_bins",
                )
                split_stratify_columns = normalize_stratify_columns(draft_source_df, split_stratify_columns)
                split_numeric_bins = safe_int(
                    split_numeric_bins_input,
                    default=DEFAULT_STRATIFY_NUMERIC_BINS,
                    minimum=2,
                )
                validation_metadata = {
                    "mode": "split_train_data",
                    "test_fraction": float(test_fraction),
                    "random_state": int(split_seed),
                    "stratify_columns": list(split_stratify_columns),
                    "numeric_bins": split_numeric_bins,
                }
                draft_source_metadata = {
                    **draft_source_metadata,
                    "validation": validation_metadata,
                }
                validation_tab.caption("Split train data settings are staged only; split runs after Apply data setup.")
            elif validation_mode == "Separate data source":
                test_source_choice = validation_tab.radio(
                    "Test data source",
                    source_options,
                    index=source_options.index("CSV / Excel Upload"),
                    key="test_data_source_choice",
                )
                test_source = render_dataframe_source_loader(
                    validation_tab,
                    role="test",
                    source_choice=test_source_choice,
                    query_session=None,
                    checkpoint=None,
                    allow_checkpoint_restore=False,
                    update_query_params=False,
                    use_sql_form=False,
                )
                if test_source is not None:
                    missing_test_columns = validate_test_dataframe(test_source.df, draft_target, draft_features)
                    if missing_test_columns:
                        draft_error = f"Test data is missing column(s): {', '.join(missing_test_columns)}"
                        validation_tab.error(draft_error)
                    else:
                        draft_separate_test_df = test_source.df
                        validation_metadata = {
                            "mode": "separate_data_source",
                            "source": test_source.source,
                            "name": test_source.name,
                            "data_id": test_source.data_id,
                            "rows": int(len(draft_separate_test_df)),
                        }
                        draft_source_metadata = {
                            **draft_source_metadata,
                            "validation": validation_metadata,
                        }
                        validation_tab.caption(
                            "Separate test source is staged; active rows update after Apply data setup."
                        )
                else:
                    draft_error = "Separate test source is not loaded."
                    validation_tab.info("Choose and load a separate test source to enable Apply data setup.")
            else:
                draft_source_metadata = {
                    key: value for key, value in draft_source_metadata.items() if key != "validation"
                }
                validation_tab.caption("No test data will be used after Apply data setup.")

            if draft_separate_test_df is not None:
                missing_test_columns = validate_test_dataframe(draft_separate_test_df, draft_target, draft_features)
                if missing_test_columns:
                    draft_error = f"Test data is missing column(s): {', '.join(missing_test_columns)}"
                    validation_tab.error(draft_error)

            if draft_error is None:
                draft_setup_key = staged_setup_key(
                    draft_source_data_key,
                    sample_metadata,
                    validation_metadata,
                )
                draft_data_key = staged_data_key(
                    draft_source_data_key,
                    sample_metadata,
                    validation_metadata,
                )
                draft_context = {
                    "_staged": True,
                    "_source_df": draft_source_df,
                    "_separate_test_df": draft_separate_test_df,
                    "df": draft_source_df,
                    "test_df": draft_separate_test_df,
                    "uploaded_name": draft_uploaded_name,
                    "source_metadata": draft_source_metadata,
                    "source_data_id": draft_source_data_id,
                    "source_data_key": draft_source_data_key,
                    "data_key": draft_data_key,
                    "setup_key": draft_setup_key,
                    "data_source": draft_data_source,
                    "target": draft_target,
                    "target_kind": draft_target_kind,
                    "features": list(draft_features),
                    "positive_class": draft_positive_class,
                    "split_variable_limit": draft_split_variable_limit,
                }

            apply_clicked = data_setup_form.button(
                "Apply data setup",
                width="stretch",
                type="primary",
                disabled=draft_context is None,
                help="Only after this click does the configured source/sample/test setup become active for the tree.",
            )
    if apply_clicked and draft_context is not None:
        with st.status("Applying data setup", expanded=True) as status:
            st.write("Materializing sample and validation data...")
            applied_draft_context = materialize_staged_data_context(draft_context)
            status.update(label="Data setup applied", state="complete", expanded=False)
        st.session_state[APPLIED_DATA_CONTEXT_KEY] = applied_draft_context
        st.session_state["_data_setup_message"] = (
            f"Applied train rows: {len(applied_draft_context['df']):,}"
            + (
                f" | Test rows: {len(applied_draft_context['test_df']):,}"
                if applied_draft_context.get("test_df") is not None
                else ""
            )
        )
        applied_data_id = normalize_data_id(applied_draft_context.get("source_data_id"))
        if applied_data_id is not None:
            st.query_params[DATA_ID_QUERY_PARAM] = applied_data_id
        clear_candidate_cache()
        st.rerun()

    applied_context = st.session_state.get(APPLIED_DATA_CONTEXT_KEY)
    if (
        applied_context is None
        and draft_context is not None
        and can_auto_apply_draft_context(draft_context)
    ):
        applied_context = materialize_staged_data_context(draft_context)
        st.session_state[APPLIED_DATA_CONTEXT_KEY] = applied_context
    if applied_context is None:
        st.info("Configure a data source and click Apply data setup to start the tree.")
        st.stop()

    if draft_context is not None and data_context_signature(draft_context) != data_context_signature(applied_context):
        train_panel.warning("Draft data setup has unapplied changes. The tree still uses the last applied data.")
    if st.session_state.get("_data_setup_message"):
        train_panel.caption(st.session_state["_data_setup_message"])

    df = applied_context["df"]
    test_df = applied_context.get("test_df")
    uploaded_name = applied_context.get("uploaded_name")
    source_metadata = dict(applied_context.get("source_metadata") or {})
    source_data_id = applied_context.get("source_data_id")
    data_key = str(applied_context["data_key"])
    data_source = str(applied_context.get("data_source") or "data")
    target = str(applied_context["target"])
    features = [str(feature) for feature in applied_context.get("features", []) if str(feature) in df.columns]
    target_kind = infer_target_kind(df[target])
    if target_kind == "binary":
        st.session_state[POSITIVE_CLASS_SESSION_KEY] = applied_context.get("positive_class")
    else:
        st.session_state[POSITIVE_CLASS_SESSION_KEY] = None

    active_caption = f"Active train rows: {len(df):,}"
    if test_df is not None:
        active_caption += f" | Test rows: {len(test_df):,}"
    train_panel.caption(active_caption)
    validation_guard_enabled = test_df is not None and target_kind == "binary"

    if workspace_mode == "Data Setup":
        st.subheader("Active dataset")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Train rows", f"{len(df):,}")
        metric_cols[1].metric("Test rows", f"{len(test_df):,}" if test_df is not None else "0")
        metric_cols[2].metric("Split variables", f"{len(features):,}")
        metric_cols[3].metric("Target", target)
        summary_rows = [
            {"setting": "data_source", "value": str(data_source)},
            {"setting": "data_key", "value": str(data_key)},
            {"setting": "positive_class", "value": str(st.session_state.get(POSITIVE_CLASS_SESSION_KEY))},
            {"setting": "split_variable_limit", "value": str(applied_context.get("split_variable_limit"))},
        ]
        sample_metadata = source_metadata.get("sample") if isinstance(source_metadata, dict) else None
        validation_metadata = source_metadata.get("validation") if isinstance(source_metadata, dict) else None
        if sample_metadata:
            summary_rows.append({"setting": "sample", "value": json.dumps(sample_metadata, default=str)})
        if validation_metadata:
            summary_rows.append({"setting": "validation", "value": json.dumps(validation_metadata, default=str)})
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")
        with st.expander("Active split variables", expanded=False):
            st.dataframe(pd.DataFrame({"variable": features}), hide_index=True, width="stretch")
        st.info("Switch to Tree Builder when this active dataset is ready for split ranking or tree edits.")
        st.stop()

    if workspace_mode == "WOE Binning":
        render_woe_workspace(
            df=df,
            test_df=test_df,
            target=target,
            features=features,
            positive_class=st.session_state.get(POSITIVE_CLASS_SESSION_KEY),
            data_key=data_key,
        )
        save_work_checkpoint(
            work_id=work_id,
            df=df,
            data_key=data_key,
            data_source=data_source,
            uploaded_name=uploaded_name,
            data_id=source_data_id,
            source_metadata=source_metadata,
            target=target,
            selected_features=features,
            parameters={"workspace": "WOE Binning"},
            auto_parameters={},
        )
        st.stop()

    checkpoint_data = checkpoint.get("data") if isinstance(checkpoint, dict) else {}
    checkpoint_data_key = checkpoint_data.get("data_key") if isinstance(checkpoint_data, dict) else None
    checkpoint_source_match = checkpoint_data_key == data_key or str(checkpoint_data_key).startswith(f"{data_key}:")

    saved_parameters = (
        checkpoint.get("parameters")
        if isinstance(checkpoint, dict) and checkpoint_data_key == data_key
        else {}
    )
    if not isinstance(saved_parameters, dict):
        saved_parameters = {}
    min_leaf_input = st.sidebar.number_input(
        "Minimum leaf rows",
        value=safe_int(saved_parameters.get("min_leaf"), default=20, minimum=1),
        step=1,
        format="%d",
    )
    max_thresholds_input = st.sidebar.number_input(
        "Numeric candidate thresholds",
        value=safe_int(saved_parameters.get("max_thresholds"), default=40, minimum=1),
        step=1,
        format="%d",
    )
    max_numeric_bins_input = st.sidebar.number_input(
        "Numeric multiway max bins",
        value=safe_int(saved_parameters.get("max_numeric_bins"), default=4, minimum=2),
        step=1,
        format="%d",
    )
    max_categories_input = st.sidebar.number_input(
        "Categorical candidate levels",
        value=safe_int(saved_parameters.get("max_categories"), default=20, minimum=1),
        step=1,
        format="%d",
    )
    max_category_groups_input = st.sidebar.number_input(
        "Categorical grouped max branches",
        value=safe_int(saved_parameters.get("max_category_groups"), default=5, minimum=2),
        step=1,
        format="%d",
    )
    parallel_workers_input = st.sidebar.number_input(
        "Parallel split workers",
        value=safe_int(saved_parameters.get("parallel_workers"), default=DEFAULT_PARALLEL_WORKERS, minimum=1),
        min_value=1,
        max_value=CPU_COUNT,
        step=1,
        format="%d",
        help="Split candidates are scored feature-by-feature in parallel. Thread workers avoid copying the full DataFrame.",
    )

    min_leaf = safe_int(min_leaf_input, default=20, minimum=1)
    max_thresholds = safe_int(max_thresholds_input, default=40, minimum=1)
    max_numeric_bins = safe_int(max_numeric_bins_input, default=4, minimum=2)
    max_categories = safe_int(max_categories_input, default=20, minimum=1)
    max_category_groups = safe_int(max_category_groups_input, default=5, minimum=2)
    parallel_workers = safe_int(parallel_workers_input, default=DEFAULT_PARALLEL_WORKERS, minimum=1)
    split_variable_limit = min(
        len(features),
        safe_int(applied_context.get("split_variable_limit"), default=min(50, max(1, len(features))), minimum=1),
    ) if features else 0
    ranking_features = features[:split_variable_limit]

    state_key = (data_key, target, TREE_SCHEMA_VERSION)
    if "state_key" not in st.session_state or st.session_state.state_key != state_key:
        st.session_state.state_key = state_key
        if not restore_tree_state_from_checkpoint(checkpoint, state_key, df):
            init_tree(df)

    parameters = {
        "min_leaf": min_leaf,
        "max_thresholds": max_thresholds,
        "max_numeric_bins": max_numeric_bins,
        "max_categories": max_categories,
        "max_category_groups": max_category_groups,
        "parallel_workers": parallel_workers,
        "split_variable_limit": split_variable_limit,
    }
    saved_auto_parameters = (
        checkpoint.get("auto_parameters")
        if isinstance(checkpoint, dict) and checkpoint_data_key == data_key
        else {}
    )
    if not isinstance(saved_auto_parameters, dict):
        saved_auto_parameters = {}
    auto_parameters: dict[str, Any] = {}

    def save_and_rerun(selected_features_override: list[str] | None = None) -> None:
        save_work_checkpoint(
            work_id=work_id,
            df=df,
            data_key=data_key,
            data_source=data_source,
            uploaded_name=uploaded_name,
            data_id=source_data_id,
            source_metadata=source_metadata,
            target=target,
            selected_features=selected_features_override or features,
            parameters=parameters,
            auto_parameters=auto_parameters,
        )
        st.rerun()

    with st.sidebar.expander("Optimal tree", expanded=True):
        st.caption("These settings are used only when you press Build optimal tree.")
        has_existing_splits = any(
            node.get("split") is not None for node in st.session_state.get("tree", {}).values()
        )
        auto_max_depth_input = st.number_input(
            "Max depth",
            value=safe_int(saved_auto_parameters.get("max_depth"), default=3, minimum=1),
            step=1,
            format="%d",
        )
        auto_max_leaves_input = st.number_input(
            "Max leaves",
            value=safe_int(saved_auto_parameters.get("max_leaves"), default=12, minimum=2),
            step=1,
            format="%d",
        )
        auto_min_gain_input = st.number_input(
            "Auto tree minimum information gain",
            value=safe_float(saved_auto_parameters.get("min_information_gain"), default=0.005),
            step=0.001,
            format="%.6f",
            help=(
                "Only affects Build optimal tree. Splits below this gain are skipped. "
                "Manual split preview and selected-leaf ranking are not filtered by this value."
            ),
        )
        if validation_guard_enabled:
            max_validation_gini_gap_input = st.number_input(
                "Max Gini gap",
                value=safe_float(
                    saved_auto_parameters.get("max_validation_gini_gap"),
                    default=DEFAULT_MAX_VALIDATION_GINI_GAP,
                ),
                min_value=0.0,
                step=0.01,
                format="%.4f",
                help=(
                    "When Test data exists, auto split choices must keep Test Gini from falling and must not move "
                    "Train/Test Gini gap above this value."
                ),
            )
        else:
            max_validation_gini_gap_input = DEFAULT_MAX_VALIDATION_GINI_GAP
        auto_max_depth = safe_int(auto_max_depth_input, default=3, minimum=1)
        auto_max_leaves = safe_int(auto_max_leaves_input, default=12, minimum=2)
        auto_min_gain = safe_float(auto_min_gain_input, default=0.005)
        auto_candidate_rows = len(df)
        max_validation_gini_gap = safe_float(
            max_validation_gini_gap_input,
            default=DEFAULT_MAX_VALIDATION_GINI_GAP,
        )
        auto_parameters = {
            "max_depth": auto_max_depth,
            "max_leaves": auto_max_leaves,
            "min_information_gain": auto_min_gain,
            "parallel_workers": parallel_workers,
            "max_validation_gini_gap": max_validation_gini_gap,
        }

        build_from_root_requested = st.button(
            "Build from root",
            width="stretch",
            disabled=not ranking_features,
            help="Clears the current tree and builds the optimal tree from the root node.",
        )
        continue_requested = st.button(
            "Continue from current tree",
            width="stretch",
            disabled=not ranking_features or not has_existing_splits,
            help="Keeps the current manual tree and adds optimal splits only to remaining eligible leaves.",
        )

        if build_from_root_requested or continue_requested:
            split_count = build_optimal_tree(
                df=df,
                target=target,
                features=ranking_features,
                test_df=test_df,
                min_leaf=min_leaf,
                max_thresholds=max_thresholds,
                max_categories=max_categories,
                max_numeric_bins=max_numeric_bins,
                max_category_groups=max_category_groups,
                max_depth=auto_max_depth,
                max_leaves=auto_max_leaves,
                min_information_gain=auto_min_gain,
                candidate_rows=auto_candidate_rows,
                parallel_workers=parallel_workers,
                max_validation_gini_gap=max_validation_gini_gap,
                reset_tree=build_from_root_requested,
            )
            if continue_requested:
                st.session_state.auto_tree_message = (
                    f"Optimal tree continued from current tree with {split_count} added split(s)."
                )
            else:
                st.session_state.auto_tree_message = f"Optimal tree rebuilt from root with {split_count} split(s)."
            save_and_rerun()
        if st.session_state.get("auto_tree_message"):
            st.caption(st.session_state.auto_tree_message)

    with st.sidebar.expander("Import editable tree", expanded=False):
        st.caption("Upload a tree JSON/pickle exported from this app after loading the same data.")
        imported_tree_file = st.file_uploader(
            "Tree artifact",
            type=["json", "pkl", "pickle"],
            key="tree_import_upload",
            help="Pickle import is intended only for local files exported by this app.",
        )
        if st.button("Import tree into current data", width="stretch", disabled=imported_tree_file is None):
            try:
                import_payload = read_tree_payload_upload(imported_tree_file)
                imported_tree, next_node_id, split_history, imported_features = rebuild_editable_tree_from_export(
                    df,
                    target,
                    import_payload,
                )
            except (TreeImportError, json.JSONDecodeError, pickle.PickleError, OSError, ValueError) as exc:
                st.error(f"Import failed: {exc}")
            else:
                imported_selected_features = [
                    feature for feature in imported_features if feature in df.columns and feature != target
                ]
                st.session_state.tree = imported_tree
                st.session_state.next_node_id = next_node_id
                st.session_state.split_history = split_history
                st.session_state.split_action_history = [[node_id] for node_id in split_history]
                st.session_state.current_node_id = 0
                st.session_state.tree_zoom = recommended_tree_zoom()
                st.session_state["_tree_import_message"] = (
                    f"Imported editable tree with {len(imported_tree)} node(s)."
                )
                clear_candidate_cache(data_key, target)
                save_and_rerun(imported_selected_features or features)
        if st.session_state.get("_tree_import_message"):
            st.caption(st.session_state["_tree_import_message"])

    if st.sidebar.button("Reset tree", width="stretch"):
        init_tree(df)
        clear_candidate_cache(data_key, target)
        save_and_rerun()

    undo_available = bool(st.session_state.get("split_action_history") or st.session_state.get("split_history"))
    if st.sidebar.button(
        "Undo last action",
        width="stretch",
        disabled=not undo_available,
        help="Reverts the last manual split, or all splits added by the last optimal-tree run.",
    ):
        undo_last_split()
        save_and_rerun()

    tree = st.session_state.tree

    if st.session_state.current_node_id not in tree:
        st.session_state.current_node_id = 0

    with st.expander("Tree view", expanded=True):
        st.caption("Click a connected node in the tree to edit it. The selected node details appear below.")
        selected_graph_node_id = render_interactive_tree_graph(df, target, data_key)
        if selected_graph_node_id is not None and selected_graph_node_id != st.session_state.current_node_id:
            st.session_state.current_node_id = selected_graph_node_id
        leaf_eval_df = test_df if test_df is not None else None
        leaf_dataset_name = "Test" if test_df is not None else "Train"
        leaf_rows = cached_leaf_performance_rows(
            df,
            target,
            data_key,
            eval_df=leaf_eval_df,
            dataset_name=leaf_dataset_name,
        )
        if leaf_rows:
            st.caption(
                f"Leaf performance is measured on {leaf_dataset_name}. Splits and predictions are fitted on Train."
            )
            st.dataframe(
                arrow_safe_dataframe(pd.DataFrame(leaf_rows)),
                hide_index=True,
                width="stretch",
                column_config={
                    "default_rate": st.column_config.NumberColumn(format="%.6f"),
                    "impurity": st.column_config.NumberColumn(format="%.6f"),
                    "target_mean": st.column_config.NumberColumn(format="%.6f"),
                    "target_std": st.column_config.NumberColumn(format="%.6f"),
                },
            )

    if st.session_state.current_node_id not in tree:
        st.session_state.current_node_id = 0
    current = tree[st.session_state.current_node_id]
    summary = node_summary(df, target, current["row_idx"])
    current_row_count = len(current["row_idx"])
    candidate_features = ranking_features
    compute_candidates = False
    if current["split"] is None and features:
        train_panel.markdown("**Split ranking**")
        train_panel.caption(
            split_ranking_scope_caption(
                candidate_feature_count=len(candidate_features),
                total_feature_count=len(features),
                selected_node_rows=current_row_count,
                active_train_rows=len(df),
            )
        )
        sidebar_candidate_key = candidate_cache_key(
            data_key=data_key,
            target=target,
            node_id=current["id"],
            features=candidate_features,
            row_count=current_row_count,
            parameters=parameters,
            max_rows=current_row_count,
        )
        sidebar_cached_meta = cached_candidate_meta(data_key, target, current["id"])
        sidebar_cached_candidates = get_cached_candidates(
            data_key,
            target,
            current["id"],
            sidebar_candidate_key,
        )
        if sidebar_cached_candidates is not None:
            analyzed_rows = int(sidebar_cached_meta.get("analyzed_rows", current_row_count) or current_row_count)
            train_panel.success(
                cached_ranking_ready_message(
                    candidate_count=len(sidebar_cached_candidates),
                    analyzed_rows=analyzed_rows,
                    selected_node_rows=current_row_count,
                    active_train_rows=len(df),
                )
            )
        elif sidebar_cached_meta:
            train_panel.caption("Cached ranking exists for another scope; recompute to use current settings.")
        compute_candidates = train_panel.button(
            "Recompute split ranking" if sidebar_cached_candidates is not None else "Compute split ranking",
            key=f"compute_split_ranking::{data_key}::{target}::{current['id']}",
            type="primary",
            width="stretch",
            disabled=not candidate_features,
            help=(
                "Scores candidate splits for the selected leaf. Results are cached until scope, parameters, "
                "target, data, or tree state changes."
            ),
        )
    elif features:
        train_panel.markdown("**Split ranking**")
        train_panel.caption("Select a leaf node to compute split ranking.")

    with st.expander("Model performance", expanded=True):
        st.dataframe(
            arrow_safe_dataframe(cached_model_performance_wide_table(df, test_df, target, data_key)),
            hide_index=True,
            width="stretch",
        )

    left_col, right_col = st.columns([1.1, 1.5])

    with left_col:
        panel_title = "Selected leaf" if current["split"] is None else "Selected split"
        st.subheader(f"{panel_title} {current['id']}")
        st.write(f"Path: `{current['path']}`")
        st.metric("Rows", f"{summary['n']:,}")
        st.metric(summary["impurity_label"], f"{summary['impurity']:.4f}")
        if summary["target_kind"] == "binary":
            st.metric("Default rate", f"{summary.get('default_rate', 0.0):.4f}")
        elif summary["target_kind"] == "regression":
            st.metric("Target mean", f"{summary.get('target_mean', 0.0):.4f}")
        st.write("Prediction:", summary["prediction"])
        st.dataframe(
            arrow_safe_dataframe(pd.DataFrame(
                [{"class": str(k), "count": int(v)} for k, v in summary["class_counts"].items()]
            )),
            hide_index=True,
            width="stretch",
        )

        if current["split"] is not None:
            st.info(f"Already split on {current['split']['feature']}: {current['split']['label']}")
            for child in get_node_children(current):
                if st.button(f"Go {child['label']}", key=f"go_child_{current['id']}_{child['id']}", width="stretch"):
                    st.session_state.current_node_id = child["id"]
                    save_and_rerun()
            if st.button("Revise this split", width="stretch"):
                prune_node(current["id"])
                save_and_rerun()

    with right_col:
        if current["split"] is None:
            if len(set(df.loc[current["row_idx"], target].astype(str))) <= 1:
                st.success("Pure leaf: target has one class here.")
            elif not features:
                st.warning("Select at least one feature.")
            elif not candidate_features:
                st.warning("Increase Split ranking variable limit to at least 1.")
            else:
                large_leaf = current_row_count > AUTO_COMPUTE_CANDIDATE_ROWS
                if not candidate_features:
                    st.warning("Select at least one variable for split ranking.")
                    st.stop()
                candidate_max_rows = current_row_count

                candidate_key = candidate_cache_key(
                    data_key=data_key,
                    target=target,
                    node_id=current["id"],
                    features=candidate_features,
                    row_count=current_row_count,
                    parameters=parameters,
                    max_rows=candidate_max_rows,
                )
                cached_meta = cached_candidate_meta(data_key, target, current["id"])
                cached_candidates = get_cached_candidates(data_key, target, current["id"], candidate_key)
                has_candidate_cache = cached_candidates is not None
                all_candidates = cached_candidates if cached_candidates is not None else []
                compute_caption = (
                    f"Compute split ranking on all {current_row_count:,} rows "
                    f"across {len(candidate_features):,} variable(s)."
                )
                if large_leaf and not cached_meta:
                    st.warning(
                        "Large leaf mode: split ranking is not recomputed automatically. "
                        "Press Compute split ranking when ready."
                    )
                if compute_candidates:
                    sampled_row_idx = analysis_row_idx(current["row_idx"], candidate_max_rows, df=df, target=target)
                    with st.spinner(compute_caption):
                        all_candidates = candidate_splits(
                            df=df,
                            target=target,
                            features=candidate_features,
                            row_idx=sampled_row_idx,
                            min_leaf=int(min_leaf),
                            max_thresholds=int(max_thresholds),
                            max_categories=int(max_categories),
                            max_numeric_bins=int(max_numeric_bins),
                            max_category_groups=int(max_category_groups),
                            parallel_workers=parallel_workers,
                        )
                    store_cached_candidates(
                        data_key,
                        target,
                        current["id"],
                        candidate_key,
                        all_candidates,
                        analyzed_rows=len(sampled_row_idx),
                        full_rows=current_row_count,
                    )
                    st.rerun()

                if not all_candidates:
                    if has_candidate_cache:
                        st.warning("No valid split found for the current variables and split settings.")
                    else:
                        st.info("Press Compute split ranking for this leaf.")
                    st.stop()

                meta = cached_candidate_meta(data_key, target, current["id"])
                analyzed_rows = int(meta.get("analyzed_rows", current_row_count) or current_row_count)
                if analyzed_rows < current_row_count:
                    st.caption(
                        f"Split ranking used {analyzed_rows:,} / {current_row_count:,} rows. "
                        "Applied splits still use the full leaf."
                    )

                validation_lookup = build_candidate_validation_lookup(
                    train_df=df,
                    test_df=test_df,
                    target=target,
                    node_id=current["id"],
                    candidates=all_candidates,
                    max_gini_gap=max_validation_gini_gap,
                    candidate_limit=VALIDATION_CANDIDATE_LIMIT,
                ) if validation_guard_enabled else {}
                if validation_lookup:
                    safe_count = sum(1 for stats in validation_lookup.values() if stats.get("validation_safe"))
                    st.caption(
                        f"Validation guard: {safe_count:,} / {len(validation_lookup):,} evaluated split(s) keep Test Gini stable "
                        f"and Train/Test Gini gap within {max_validation_gini_gap:.4f}."
                    )

                feature_rows = feature_summary_rows(
                    all_candidates,
                    candidate_features,
                    validation_lookup=validation_lookup,
                )
                feature_stats = {row["variable"]: row for row in feature_rows}
                leaf_feature_options = ordered_features_by_gain(candidate_features, feature_stats)
                positive_candidates = [
                    candidate
                    for candidate in all_candidates
                    if candidate.information_gain > MIN_INFORMATION_GAIN_EPSILON
                ]
                if validation_lookup:
                    positive_candidates = [
                        candidate for candidate in positive_candidates if candidate_passes_validation(candidate, validation_lookup)
                    ]
                if not leaf_feature_options or not positive_candidates:
                    st.warning(
                        "No validation-safe split with positive information gain found for this leaf."
                        if validation_lookup
                        else "No split with positive information gain found for this leaf."
                    )
                    st.dataframe(
                        arrow_safe_dataframe(
                            pd.DataFrame(
                                feature_summary_rows(
                                    all_candidates,
                                    candidate_features,
                                    include_zero_gain=True,
                                    validation_lookup=validation_lookup,
                                )
                            )
                        ),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "total_information_gain": st.column_config.NumberColumn(format="%.6f"),
                            "best_information_gain": st.column_config.NumberColumn(format="%.6f"),
                            "train_gini_after": st.column_config.NumberColumn(format="%.6f"),
                            "test_gini_after": st.column_config.NumberColumn(format="%.6f"),
                            "test_gini_delta": st.column_config.NumberColumn(format="%.6f"),
                            "gini_gap_after": st.column_config.NumberColumn(format="%.6f"),
                        },
                    )
                    st.stop()

                leaf_feature_key = node_feature_key(data_key, target, current["id"])
                ensure_node_feature(leaf_feature_key, leaf_feature_options, feature_stats)

                st.subheader("Variable for this leaf")
                selected_feature = st.selectbox(
                    "Choose one variable to try under this leaf",
                    options=leaf_feature_options,
                    key=leaf_feature_key,
                    format_func=lambda feature: (
                        f"{feature} | best={feature_stats[feature]['best_information_gain']:.6f}"
                        + (
                            f" | Test Gini={feature_stats[feature].get('test_gini_after'):.4f}"
                            f" | Gap={feature_stats[feature].get('gini_gap_after'):.4f}"
                            if validation_lookup and feature_stats[feature].get("test_gini_after") is not None
                            else f" | total IG={feature_stats[feature]['total_information_gain']:.6f}"
                        )
                    ),
                )

                selected_stats = feature_stats[selected_feature]
                current_total_gain, _, _ = cached_tree_total_gain(df, target, data_key)
                score_name = split_score_name(df[target])
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric(f"Tree total {score_name}", f"{current_total_gain:.6f}")
                metric_col2.metric(f"{selected_feature} total", f"{selected_stats['total_information_gain']:.6f}")
                metric_col3.metric(f"{selected_feature} best", f"{selected_stats['best_information_gain']:.6f}")

                st.dataframe(
                    arrow_safe_dataframe(
                        pd.DataFrame(
                            feature_summary_rows(
                                all_candidates,
                                candidate_features,
                                [selected_feature],
                                validation_lookup=validation_lookup,
                            )
                        )
                    ),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "total_information_gain": st.column_config.NumberColumn(format="%.6f"),
                        "best_information_gain": st.column_config.NumberColumn(format="%.6f"),
                        "train_gini_after": st.column_config.NumberColumn(format="%.6f"),
                        "test_gini_after": st.column_config.NumberColumn(format="%.6f"),
                        "test_gini_delta": st.column_config.NumberColumn(format="%.6f"),
                        "gini_gap_after": st.column_config.NumberColumn(format="%.6f"),
                    },
                )

                feature_candidates = [
                    c
                    for c in all_candidates
                    if c.feature == selected_feature and c.information_gain > MIN_INFORMATION_GAIN_EPSILON
                ]
                st.subheader(f"Strongest auto split for {selected_feature}")

                if not feature_candidates:
                    st.warning("No valid split found for this variable.")
                else:
                    selected_candidate = max(
                        feature_candidates,
                        key=lambda c: candidate_validation_sort_key(c, validation_lookup),
                    )
                    selected_validation_stats = validation_lookup.get(id(selected_candidate))
                    st.dataframe(
                        arrow_safe_dataframe(
                            pd.DataFrame(
                                candidate_split_summary_rows(
                                    df,
                                    target,
                                    current["row_idx"],
                                    selected_candidate,
                                    validation_stats=selected_validation_stats,
                                )
                            )
                        ),
                        hide_index=True,
                        width="stretch",
                        column_config=candidate_split_summary_column_config(score_name),
                    )

                    st.caption("Branch details")
                    st.dataframe(
                        arrow_safe_dataframe(
                            pd.DataFrame(
                                candidate_branch_detail_rows(df, target, current["row_idx"], selected_candidate)
                            )
                        ),
                        hide_index=True,
                        width="stretch",
                        column_config=branch_detail_column_config(score_name),
                    )

                    st.dataframe(
                        arrow_safe_dataframe(
                            pd.DataFrame(
                                candidate_impact_rows(
                                    df,
                                    target,
                                    current["row_idx"],
                                    selected_candidate,
                                    current_total_gain=current_total_gain,
                                )
                            )
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                    if st.button(
                        f"Apply auto {selected_feature} split",
                        key=f"apply_split_{current['id']}_{selected_feature}",
                        type="primary",
                        width="stretch",
                        disabled=bool(validation_lookup) and not candidate_passes_validation(selected_candidate, validation_lookup),
                    ):
                        apply_split(df, selected_candidate)
                        save_and_rerun()

                st.subheader("Manual split")
                selected_series = df.loc[current["row_idx"], selected_feature]
                manual_candidate: SplitCandidate | None = None

                if pd.api.types.is_numeric_dtype(selected_series):
                    manual_text = st.text_input(
                        "Manual numeric threshold(s)",
                        value="",
                        placeholder="Example: 42000 or 30000, 60000",
                        key=f"manual_thresholds_{current['id']}_{selected_feature}",
                    )
                    selected_numeric = pd.to_numeric(selected_series, errors="coerce")
                    missing_policy_choice = "right"
                    if selected_numeric.isna().any():
                        missing_policy_choice = st.selectbox(
                            "Manual numeric missing routing",
                            options=["auto", "right", "left", "separate"],
                            index=0,
                            key=f"manual_missing_policy_{current['id']}_{selected_feature}",
                            format_func=lambda value: {
                                "auto": "Auto best",
                                "right": "Send missing to upper/right branch",
                                "left": "Send missing to lower/left branch",
                                "separate": "Keep missing as separate branch",
                            }[str(value)],
                        )
                    if manual_text.strip():
                        try:
                            thresholds = parse_threshold_text(manual_text)
                            if missing_policy_choice == "auto":
                                manual_candidates = [
                                    candidate
                                    for policy in numeric_missing_policies(selected_numeric)
                                    for candidate in [
                                        score_numeric_manual_bins(
                                            df=df,
                                            target=target,
                                            row_idx=current["row_idx"],
                                            feature=selected_feature,
                                            thresholds=thresholds,
                                            min_leaf=int(min_leaf),
                                            missing_policy=policy,
                                        )
                                    ]
                                    if candidate is not None
                                ]
                                manual_candidate = max(
                                    manual_candidates,
                                    key=lambda candidate: candidate.information_gain,
                                    default=None,
                                )
                            else:
                                manual_candidate = score_numeric_manual_bins(
                                    df=df,
                                    target=target,
                                    row_idx=current["row_idx"],
                                    feature=selected_feature,
                                    thresholds=thresholds,
                                    min_leaf=int(min_leaf),
                                    missing_policy=str(missing_policy_choice),
                                )
                            branch_rows = manual_numeric_branch_rows(
                                df,
                                current["row_idx"],
                                selected_feature,
                                thresholds,
                                missing_policy=manual_candidate.missing_policy if manual_candidate else str(
                                    "right" if missing_policy_choice == "auto" else missing_policy_choice
                                ),
                            )
                            if manual_candidate is None:
                                low_branches = [
                                    row
                                    for row in branch_rows
                                    if int(row["rows"]) < int(min_leaf)
                                ]
                                if low_branches:
                                    st.dataframe(
                                        arrow_safe_dataframe(pd.DataFrame(branch_rows)),
                                        hide_index=True,
                                        width="stretch",
                                    )
                                    branch_text = ", ".join(
                                        f"{row['branch']} n={row['rows']}" for row in low_branches
                                    )
                                    st.warning(
                                        "Manual split is not valid because every branch must have "
                                        f"at least {int(min_leaf)} rows. Low-count branch(es): {branch_text}."
                                    )
                                else:
                                    st.warning("Manual split is not valid for the current node.")
                        except ValueError:
                            st.warning("Manual thresholds must be numeric values separated by comma.")
                else:
                    levels = (
                        selected_series.astype("object")
                        .where(selected_series.notna(), "__MISSING__")
                        .drop_duplicates()
                        .tolist()
                    )
                    levels = sorted(levels, key=lambda x: str(x))
                    level_texts = [str(level) for level in levels]
                    text_to_value = {str(level): level for level in levels}
                    group_key = category_group_state_key(data_key, target, current["id"], selected_feature)

                    if group_key not in st.session_state:
                        st.session_state[group_key] = [level_texts.copy()]
                    st.session_state[group_key] = normalize_category_groups(st.session_state[group_key], level_texts)
                    groups = st.session_state[group_key]

                    st.caption("Select values, split them into a new group, or merge groups back together.")
                    seed_col1, seed_col2, seed_col3 = st.columns(3)
                    if seed_col1.button("All in one", key=f"group_all_{group_key}", type="primary", width="stretch"):
                        st.session_state[group_key] = [level_texts.copy()]
                        save_and_rerun()
                    if seed_col2.button("One per value", key=f"group_single_{group_key}", type="primary", width="stretch"):
                        st.session_state[group_key] = [[value] for value in level_texts]
                        save_and_rerun()
                    if seed_col3.button("Profile groups", key=f"group_profile_{group_key}", type="primary", width="stretch"):
                        st.session_state[group_key] = profile_group_texts(
                            df=df,
                            target=target,
                            row_idx=current["row_idx"],
                            feature=selected_feature,
                            max_groups=int(max_category_groups),
                        )
                        save_and_rerun()

                    st.dataframe(
                        arrow_safe_dataframe(
                            pd.DataFrame(
                                category_level_rows(df, target, current["row_idx"], selected_feature)
                            )
                        ),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "default_rate": st.column_config.NumberColumn(format="%.6f"),
                            "impurity": st.column_config.NumberColumn(format="%.6f"),
                            "target_mean": st.column_config.NumberColumn(format="%.6f"),
                            "target_std": st.column_config.NumberColumn(format="%.6f"),
                        },
                    )

                    st.dataframe(
                        arrow_safe_dataframe(
                            pd.DataFrame(
                                category_group_rows(
                                    df=df,
                                    target=target,
                                    row_idx=current["row_idx"],
                                    feature=selected_feature,
                                    groups=groups,
                                    text_to_value=text_to_value,
                                )
                            )
                        ),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "default_rate": st.column_config.NumberColumn(format="%.6f"),
                            "impurity": st.column_config.NumberColumn(format="%.6f"),
                            "target_mean": st.column_config.NumberColumn(format="%.6f"),
                            "target_std": st.column_config.NumberColumn(format="%.6f"),
                        },
                    )

                    source_group_index = st.selectbox(
                        "Group to split",
                        range(len(groups)),
                        format_func=lambda i: f"G{i + 1}: {', '.join(groups[i])}",
                        key=f"group_source_{group_key}",
                    )
                    values_to_split = st.multiselect(
                        "Values to move into a new group",
                        options=groups[int(source_group_index)],
                        key=f"group_values_{group_key}",
                    )
                    if st.button(
                        "Move selected to new group",
                        key=f"group_split_{group_key}",
                        type="primary",
                        width="stretch",
                    ):
                        source_index = int(source_group_index)
                        selected_values = set(values_to_split)
                        if selected_values and selected_values != set(groups[source_index]):
                            st.session_state[group_key][source_index] = [
                                value for value in groups[source_index] if value not in selected_values
                            ]
                            st.session_state[group_key].append(list(values_to_split))
                            st.session_state[group_key] = normalize_category_groups(
                                st.session_state[group_key], level_texts
                            )
                            save_and_rerun()
                        else:
                            st.warning("Select some, but not all, values from the source group.")

                    merge_options = list(range(len(groups)))
                    groups_to_merge = st.multiselect(
                        "Groups to merge",
                        options=merge_options,
                        format_func=lambda i: f"G{i + 1}: {', '.join(groups[i])}",
                        key=f"group_merge_{group_key}",
                    )
                    if st.button(
                        "Merge selected groups",
                        key=f"group_merge_button_{group_key}",
                        type="primary",
                        width="stretch",
                    ):
                        selected_group_ids = sorted(set(int(i) for i in groups_to_merge))
                        if len(selected_group_ids) >= 2:
                            merged: list[str] = []
                            next_groups: list[list[str]] = []
                            for i, group in enumerate(groups):
                                if i in selected_group_ids:
                                    merged.extend(group)
                                else:
                                    next_groups.append(group)
                            next_groups.append(merged)
                            st.session_state[group_key] = normalize_category_groups(next_groups, level_texts)
                            save_and_rerun()
                        else:
                            st.warning("Select at least two groups to merge.")

                    actual_groups = [
                        [text_to_value[value] for value in group if value in text_to_value]
                        for group in st.session_state[group_key]
                    ]
                    manual_candidate = score_category_manual_groups(
                        df=df,
                        target=target,
                        row_idx=current["row_idx"],
                        feature=selected_feature,
                        groups=actual_groups,
                        min_leaf=int(min_leaf),
                    )

                manual_validation_stats = None
                if manual_candidate is not None and validation_guard_enabled:
                    manual_validation_stats = candidate_validation_stats(
                        train_df=df,
                        test_df=test_df,
                        target=target,
                        node_id=current["id"],
                        candidate=manual_candidate,
                        max_gini_gap=max_validation_gini_gap,
                    )

                if manual_candidate is None:
                    st.caption("Enter a valid manual split to preview its impact.")
                else:
                    manual_validation_safe = (
                        manual_validation_stats is None
                        or bool(manual_validation_stats.get("validation_safe"))
                    )
                    if not manual_validation_safe:
                        st.warning(
                            "Manual split is blocked by validation guard because it would reduce Test Gini "
                            "or increase the Train/Test Gini gap beyond the allowed limit."
                        )
                    st.caption("Manual split summary")
                    st.dataframe(
                        arrow_safe_dataframe(
                            pd.DataFrame(
                                candidate_split_summary_rows(
                                    df,
                                    target,
                                    current["row_idx"],
                                    manual_candidate,
                                    validation_stats=manual_validation_stats,
                                )
                            )
                        ),
                        hide_index=True,
                        width="stretch",
                        column_config=candidate_split_summary_column_config(score_name),
                    )
                    st.caption("Manual branch details")
                    st.dataframe(
                        arrow_safe_dataframe(
                            pd.DataFrame(
                                candidate_branch_detail_rows(df, target, current["row_idx"], manual_candidate)
                            )
                        ),
                        hide_index=True,
                        width="stretch",
                        column_config=branch_detail_column_config(score_name),
                    )
                    st.dataframe(
                        arrow_safe_dataframe(
                            pd.DataFrame(
                                candidate_impact_rows(
                                    df,
                                    target,
                                    current["row_idx"],
                                    manual_candidate,
                                    current_total_gain=current_total_gain,
                                )
                            )
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                    if st.button(
                        f"Apply manual {selected_feature} split",
                        key=f"apply_manual_split_{current['id']}_{selected_feature}",
                        type="primary",
                        width="stretch",
                        disabled=not manual_validation_safe,
                    ):
                        apply_split(df, manual_candidate)
                        save_and_rerun()
        else:
            st.info("Use 'Revise this split' to clear this node's children. Then choose a new variable or manual split for the same node.")

    st.divider()
    st.subheader("Tree export")
    export_payload = tree_export(
        df=df,
        target=target,
        features=features,
        parameters=parameters,
    )
    export_col1, export_col2, export_col3 = st.columns(3)
    export_col1.metric("Export nodes", export_payload["node_count"])
    export_col2.metric("Export splits", export_payload["split_count"])
    export_col3.metric("Export leaves", export_payload["leaf_count"])
    st.download_button(
        "Download runnable tree JSON",
        data=json.dumps(export_payload, indent=2, default=str),
        file_name="interactive_entropy_tree_runnable.json",
        mime="application/json",
        key=f"download_tree_json_{export_payload['node_count']}_{export_payload['split_count']}_{st.session_state.next_node_id}",
    )
    st.download_button(
        "Download runnable tree pickle",
        data=pickle.dumps(export_payload, protocol=pickle.HIGHEST_PROTOCOL),
        file_name="interactive_entropy_tree_runnable.pkl",
        mime="application/octet-stream",
        key=f"download_tree_pickle_{export_payload['node_count']}_{export_payload['split_count']}_{st.session_state.next_node_id}",
    )
    st.json(export_payload, expanded=False)
    save_work_checkpoint(
        work_id=work_id,
        df=df,
        data_key=data_key,
        data_source=data_source,
        uploaded_name=uploaded_name,
        data_id=source_data_id,
        source_metadata=source_metadata,
        target=target,
        selected_features=features,
        parameters=parameters,
        auto_parameters=auto_parameters,
    )


if __name__ == "__main__":
    main()
