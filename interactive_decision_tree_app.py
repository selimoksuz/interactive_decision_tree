from __future__ import annotations

import hashlib
import io
import json
import pickle
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


TREE_SCHEMA_VERSION = 4
CHECKPOINT_SCHEMA_VERSION = 1
WORK_ID_QUERY_PARAM = "work_id"
CHECKPOINT_DIR = Path(__file__).with_name(".tree_checkpoints")
POSITIVE_CLASS_SESSION_KEY = "_interactive_tree_positive_class"
MIN_INFORMATION_GAIN_EPSILON = 1e-12


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


def make_demo_data(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    age = rng.integers(21, 72, size=n)
    income = rng.normal(52_000, 18_000, size=n).clip(12_000, 140_000)
    tenure = rng.integers(0, 120, size=n)
    segment = rng.choice(
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
        size=n,
        p=[0.11, 0.1, 0.1, 0.09, 0.09, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06],
    )
    channel = rng.choice(
        ["branch", "mobile", "web", "call_center", "agent", "atm"],
        size=n,
        p=[0.18, 0.32, 0.24, 0.1, 0.09, 0.07],
    )
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
    )
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
    )

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


def infer_target_kind(y: pd.Series) -> str:
    non_missing = y.dropna()
    unique_count = non_missing.nunique()
    if pd.api.types.is_numeric_dtype(non_missing) and unique_count > 10:
        return "regression"
    if unique_count == 2:
        return "binary"
    return "classification"


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
    classes = list(y.dropna().unique())
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
    y = df.loc[row_idx, target]
    target_kind = infer_target_kind(df[target])
    counts = y.value_counts(dropna=False)
    positive_class = choose_positive_class(df[target]) if target_kind == "binary" else None
    numeric = pd.to_numeric(y, errors="coerce") if target_kind == "regression" else None
    prediction = float(numeric.mean()) if target_kind == "regression" and numeric is not None else (
        counts.index[0] if not counts.empty else None
    )
    out = {
        "n": len(row_idx),
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


def score_split(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    split_type: str,
    value: Any,
    min_leaf: int,
) -> SplitCandidate | None:
    frame = df.loc[row_idx, [feature, target]]

    if split_type == "numeric_le":
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        masks_and_labels = [
            (numeric <= float(value), f"<= {float(value):.6g}"),
            (~(numeric <= float(value)), f"> {float(value):.6g}"),
        ]
        label = f"{feature} <= {float(value):.6g}"
    elif split_type == "category_eq":
        values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__")
        masks_and_labels = [
            (values == value, f"== {value}"),
            (values != value, f"!= {value}"),
        ]
        label = f"{feature} == {value}"
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    scored = score_branch_split(frame, target, masks_and_labels, min_leaf, infer_target_kind(df[target]))
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
    )


def score_numeric_multiway_split(
    df: pd.DataFrame,
    target: str,
    row_idx: list[int],
    feature: str,
    bin_count: int,
    min_leaf: int,
) -> SplitCandidate | None:
    frame = df.loc[row_idx, [feature, target]]
    numeric = pd.to_numeric(frame[feature], errors="coerce")
    thresholds = numeric_bin_thresholds(numeric, bin_count)
    if len(thresholds) != bin_count - 1:
        return None

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

    last_threshold = thresholds[-1]
    masks_and_labels.append((numeric > last_threshold, f"> {last_threshold:.6g}"))

    if numeric.isna().any():
        non_missing_union = pd.Series(False, index=frame.index)
        for mask, _ in masks_and_labels:
            non_missing_union = non_missing_union | mask
        masks_and_labels[-1] = (
            masks_and_labels[-1][0] | ~non_missing_union,
            f"{masks_and_labels[-1][1]} or missing",
        )

    scored = score_branch_split(frame, target, masks_and_labels, min_leaf, infer_target_kind(df[target]))
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
        label=f"{feature} into {bin_count} bins",
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


def category_profile_order(df: pd.DataFrame, target: str, row_idx: list[int], feature: str) -> list[Any]:
    frame = df.loc[row_idx, [feature, target]]
    values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__")
    target_kind = infer_target_kind(df[target])

    if target_kind == "binary":
        positive_class = choose_positive_class(df[target])
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
) -> list[SplitCandidate]:
    frame = df.loc[row_idx, [feature, target]]
    values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__")
    ordered_values = category_profile_order(df, target, row_idx, feature)
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

        scored = score_branch_split(frame, target, masks_and_labels, min_leaf, infer_target_kind(df[target]))
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
) -> SplitCandidate | None:
    thresholds = sorted({float(x) for x in thresholds})
    if not thresholds:
        return None

    frame = df.loc[row_idx, [feature, target]]
    numeric = pd.to_numeric(frame[feature], errors="coerce")
    masks_and_labels: list[tuple[pd.Series, str]] = []
    previous_threshold: float | None = None
    covered = pd.Series(False, index=frame.index)

    for threshold in thresholds:
        if previous_threshold is None:
            mask = numeric <= threshold
            branch_label = f"<= {threshold:.6g}"
        else:
            mask = (numeric > previous_threshold) & (numeric <= threshold)
            branch_label = f"> {previous_threshold:.6g} and <= {threshold:.6g}"
        covered = covered | mask
        masks_and_labels.append((mask, branch_label))
        previous_threshold = threshold

    last_mask = numeric > thresholds[-1]
    if numeric.isna().any():
        last_mask = last_mask | ~covered
    masks_and_labels.append((last_mask, f"> {thresholds[-1]:.6g}"))

    scored = score_branch_split(frame, target, masks_and_labels, min_leaf, infer_target_kind(df[target]))
    if scored is None:
        return None
    parent_impurity, weighted_impurity, branch_labels, branch_ns, branch_entropies = scored
    split_type = "numeric_le" if len(thresholds) == 1 else "numeric_manual_bins"
    label = (
        f"{feature} <= {thresholds[0]:.6g}"
        if len(thresholds) == 1
        else f"{feature} manual bins: {', '.join(f'{x:.6g}' for x in thresholds)}"
    )

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
) -> list[SplitCandidate]:
    candidates: list[SplitCandidate] = []

    for feature in features:
        s = df.loc[row_idx, feature]

        if pd.api.types.is_numeric_dtype(s):
            for threshold in numeric_thresholds(s, max_thresholds):
                candidate = score_split(
                    df=df,
                    target=target,
                    row_idx=row_idx,
                    feature=feature,
                    split_type="numeric_le",
                    value=threshold,
                    min_leaf=min_leaf,
                )
                if candidate is not None:
                    candidates.append(candidate)
            for bin_count in range(3, max_numeric_bins + 1):
                candidate = score_numeric_multiway_split(
                    df=df,
                    target=target,
                    row_idx=row_idx,
                    feature=feature,
                    bin_count=bin_count,
                    min_leaf=min_leaf,
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
                )
                if candidate is not None:
                    candidates.append(candidate)
            grouped_candidates = score_category_profile_groups(
                df=df,
                target=target,
                row_idx=row_idx,
                feature=feature,
                max_groups=max_category_groups,
                min_leaf=min_leaf,
            )
            candidates.extend(grouped_candidates)

    feature_rank = {feature: i for i, feature in enumerate(features)}
    return sorted(candidates, key=lambda x: (-x.information_gain, feature_rank.get(x.feature, 9999), x.label))


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
    st.session_state.auto_tree_message = ""


def split_branch_indices(
    df: pd.DataFrame,
    row_idx: list[int],
    candidate: SplitCandidate,
) -> list[tuple[str, list[int]]]:
    frame = df.loc[row_idx, [candidate.feature]]

    if candidate.split_type == "numeric_le":
        numeric = pd.to_numeric(frame[candidate.feature], errors="coerce")
        return [
            (candidate.branch_labels[0], frame[numeric <= float(candidate.value)].index.tolist()),
            (candidate.branch_labels[1], frame[~(numeric <= float(candidate.value))].index.tolist()),
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
        branches: list[tuple[str, list[int]]] = []
        previous_threshold: float | None = None
        covered = pd.Series(False, index=frame.index)
        for i, threshold in enumerate(thresholds):
            if previous_threshold is None:
                mask = numeric <= threshold
            else:
                mask = (numeric > previous_threshold) & (numeric <= threshold)
            covered = covered | mask
            branches.append((candidate.branch_labels[i], frame[mask].index.tolist()))
            previous_threshold = threshold

        last_mask = numeric > thresholds[-1]
        if numeric.isna().any():
            last_mask = last_mask | ~covered
        branches.append((candidate.branch_labels[-1], frame[last_mask].index.tolist()))
        return branches
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


def split_node(df: pd.DataFrame, node_id: int, candidate: SplitCandidate, select_first_child: bool = True) -> None:
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
    st.session_state.split_history.append(node_id)
    st.session_state.current_node_id = child_ids[0] if select_first_child else node_id


def apply_split(df: pd.DataFrame, candidate: SplitCandidate) -> None:
    split_node(df, st.session_state.current_node_id, candidate, select_first_child=True)


def prune_node(node_id: int) -> None:
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
    st.session_state.current_node_id = node_id


def undo_last_split() -> bool:
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
    min_leaf: int,
    max_thresholds: int,
    max_categories: int,
    max_numeric_bins: int,
    max_category_groups: int,
    max_depth: int,
    max_leaves: int,
    min_information_gain: float,
) -> int:
    init_tree(df)
    split_count = 0

    while True:
        if len(current_leaves()) >= max_leaves:
            break

        best_node_id: int | None = None
        best_candidate: SplitCandidate | None = None
        best_weighted_delta = -1.0

        leaf_count_now = len(current_leaves())
        for leaf in current_leaves():
            if leaf["depth"] >= max_depth:
                continue

            candidates = candidate_splits(
                df=df,
                target=target,
                features=features,
                row_idx=leaf["row_idx"],
                min_leaf=min_leaf,
                max_thresholds=max_thresholds,
                max_categories=max_categories,
                max_numeric_bins=max_numeric_bins,
                max_category_groups=max_category_groups,
            )
            for candidate in candidates:
                if candidate.information_gain < min_information_gain:
                    break
                if leaf_count_now + candidate.branch_count - 1 > max_leaves:
                    continue

                weighted_delta = candidate_total_gain_delta(df, candidate, leaf["row_idx"])
                if weighted_delta > best_weighted_delta:
                    best_weighted_delta = weighted_delta
                    best_candidate = candidate
                    best_node_id = leaf["id"]
                break

        if best_node_id is None or best_candidate is None:
            break

        split_node(df, best_node_id, best_candidate, select_first_child=False)
        split_count += 1

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


def render_sql_source_loader() -> None:
    connections = secret_sql_connections()
    connection_mode_options = ["Manual SQLAlchemy URL"]
    if connections:
        connection_mode_options.insert(0, "Saved secret connection")

    with st.sidebar.form("sql_source_form"):
        connection_mode = st.selectbox("SQL connection", connection_mode_options)
        if connection_mode == "Saved secret connection":
            selected_connection = st.selectbox("Saved connection", sorted(connections))
            connection_url = connections[selected_connection]
        else:
            connection_url = st.text_input("SQLAlchemy connection URL", type="password")

        sql_mode = st.radio("SQL mode", ["Table", "Query"], horizontal=True)
        table_name = ""
        query_text = ""
        if sql_mode == "Table":
            table_name = st.text_input("Table name")
        else:
            query_text = st.text_area("SQL query", height=120)

        full_table = st.checkbox("Load full result", value=False)
        limit_value = st.number_input(
            "Row limit",
            value=DEFAULT_SQL_LIMIT,
            min_value=1,
            step=1000,
            disabled=full_table,
            format="%d",
        )
        sample_n_value = st.number_input("Optional sample rows", value=0, min_value=0, step=100, format="%d")
        submitted = st.form_submit_button("Load SQL data", width="stretch")

    if not submitted:
        st.info("Choose a SQL source in the sidebar and load data.")
        st.stop()

    if not connection_url:
        st.sidebar.error("SQL connection URL is required.")
        st.stop()

    try:
        df = read_sql_dataframe(
            connection_url,
            table=table_name.strip() or None,
            query=query_text.strip() or None,
            limit=None if full_table else int(limit_value),
            sample_n=int(sample_n_value) or None,
            full_table=bool(full_table),
        )
        save_source_session(
            df,
            source="sql",
            name=table_name.strip() or "SQL query",
            metadata={
                "sql": {
                    "mode": sql_mode.lower(),
                    "table": table_name.strip() or None,
                    "limit": None if full_table else int(limit_value),
                    "sample_n": int(sample_n_value) or None,
                    "full_table": bool(full_table),
                }
            },
        )
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"SQL load failed: {exc}")
        st.stop()


def restore_checkpoint_dataframe(checkpoint: dict[str, Any] | None) -> tuple[pd.DataFrame, str, str] | None:
    if not checkpoint:
        return None
    data = checkpoint.get("data")
    if not isinstance(data, dict) or data.get("source") != "uploaded":
        return None
    frame_json = data.get("frame_json")
    if not isinstance(frame_json, str) or not frame_json:
        return None
    try:
        df = pd.read_json(io.StringIO(frame_json), orient="split")
    except ValueError:
        return None
    uploaded_name = str(data.get("name") or "restored_upload.csv")
    data_key = str(data.get("data_key") or uploaded_data_key(uploaded_name, df))
    return df, uploaded_name, data_key


def restore_checkpoint_ui_state(checkpoint: dict[str, Any] | None) -> None:
    if not checkpoint:
        return
    ui_state = checkpoint.get("ui_state")
    if not isinstance(ui_state, dict):
        return
    for key, value in ui_state.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
    if tuple(tree_state.get("state_key", [])) != state_key:
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
    st.session_state.auto_tree_message = str(tree_state.get("auto_tree_message", ""))
    st.session_state.tree_zoom = safe_float(tree_state.get("tree_zoom"), default=recommended_tree_zoom())
    return True


def checkpoint_ui_state() -> dict[str, Any]:
    prefixes = (
        "category_groups::",
        "node_features_v2::",
        "manual_thresholds_",
        "group_source_",
        "group_values_",
        "group_merge_",
    )
    out: dict[str, Any] = {}
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(prefixes):
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
    if "tree" not in st.session_state:
        return

    data_payload: dict[str, Any] = {
        "source": data_source,
        "name": uploaded_name,
        "data_id": data_id,
        "data_key": data_key,
        "rows": int(len(df)),
        "columns": [str(column) for column in df.columns],
        "metadata": json_safe(source_metadata or {}),
    }
    if data_source == "uploaded":
        data_payload["frame_json"] = df.to_json(orient="split", date_format="iso", default_handler=str)

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
            "auto_tree_message": json_safe(st.session_state.get("auto_tree_message", "")),
            "tree_zoom": json_safe(st.session_state.get("tree_zoom")),
        },
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
        return [
            {"feature": feature, "operator": "<=", "threshold": threshold},
            {"feature": feature, "operator": ">", "threshold": threshold, "includes_missing": True},
        ]

    if split_type == "category_eq":
        category_value = json_safe(value)
        return [
            {"feature": feature, "operator": "==", "value": category_value},
            {"feature": feature, "operator": "!=", "value": category_value},
        ]

    if split_type in ("numeric_bins", "numeric_manual_bins"):
        thresholds = [json_safe(x) for x in value]
        conditions: list[dict[str, Any]] = []
        previous_threshold: Any | None = None
        for threshold in thresholds:
            if previous_threshold is None:
                conditions.append({"feature": feature, "operator": "<=", "threshold": threshold})
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
                "includes_missing": True,
            }
        )
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
            "missing_values": "Numeric missing values are routed to the final greater-than branch when exported with includes_missing=true. Categorical missing values are represented as __MISSING__.",
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


def feature_summary_rows(
    candidates: list[SplitCandidate],
    features: list[str],
    selected_features: list[str] | None = None,
    include_zero_gain: bool = False,
) -> list[dict[str, Any]]:
    selected = set(selected_features or [])
    rows: list[dict[str, Any]] = []
    for feature in features:
        feature_candidates = [c for c in candidates if c.feature == feature]
        if feature_candidates:
            best = max(feature_candidates, key=lambda c: c.information_gain)
            total_gain = sum(c.information_gain for c in feature_candidates)
            if not include_zero_gain and best.information_gain <= MIN_INFORMATION_GAIN_EPSILON:
                continue
            rows.append(
                {
                    "selected": "yes" if feature in selected else "no",
                    "variable": feature,
                    "total_information_gain": total_gain,
                    "best_information_gain": best.information_gain,
                    "candidate_count": len(feature_candidates),
                    "best_split": best.label,
                    "best_branches": best.branch_count,
                }
            )
        else:
            if not include_zero_gain:
                continue
            rows.append(
                {
                    "selected": "yes" if feature in selected else "no",
                    "variable": feature,
                    "total_information_gain": 0.0,
                    "best_information_gain": 0.0,
                    "candidate_count": 0,
                    "best_split": "",
                    "best_branches": 0,
                }
            )
    return rows


def ordered_features_by_gain(features: list[str], feature_stats: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        [
            feature
            for feature in features
            if feature_stats.get(feature, {}).get("best_information_gain", 0.0) > MIN_INFORMATION_GAIN_EPSILON
        ],
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
            title="",
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
        graph_node.title = ""
        graph_nodes.append(graph_node)

        for child in get_node_children(node):
            graph_edges.append(
                Edge(
                    source=str(node_id),
                    target=str(child["id"]),
                    label=truncate_text(child["label"], edge_label_width),
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
) -> list[dict[str, Any]]:
    current_total_gain, _, _ = tree_total_gain(df, target)
    total_delta = candidate_total_gain_delta(df, candidate, row_idx)
    score_name = split_score_name(df[target])
    return [
        {"metric": f"selected_variable_{score_name}", "value": candidate.information_gain},
        {"metric": f"weighted_tree_{score_name}_delta", "value": total_delta},
        {"metric": f"tree_total_{score_name}_before", "value": current_total_gain},
        {"metric": f"tree_total_{score_name}_after", "value": current_total_gain + total_delta},
        {"metric": "split_branches", "value": candidate.branch_count},
        {"metric": "branch_rows", "value": " / ".join(str(n) for n in candidate.branch_ns)},
        {"metric": "branch_impurity", "value": " / ".join(f"{x:.6f}" for x in candidate.branch_entropies)},
    ]


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
) -> list[dict[str, Any]]:
    thresholds = sorted({float(x) for x in thresholds})
    if not thresholds:
        return []

    frame = df.loc[row_idx, [feature]]
    numeric = pd.to_numeric(frame[feature], errors="coerce")
    rows: list[dict[str, Any]] = []
    previous_threshold: float | None = None
    covered = pd.Series(False, index=frame.index)

    for threshold in thresholds:
        if previous_threshold is None:
            mask = numeric <= threshold
            label = f"<= {threshold:.6g}"
        else:
            mask = (numeric > previous_threshold) & (numeric <= threshold)
            label = f"> {previous_threshold:.6g} and <= {threshold:.6g}"
        covered = covered | mask
        rows.append({"branch": label, "rows": int(mask.sum())})
        previous_threshold = threshold

    last_mask = numeric > thresholds[-1]
    if numeric.isna().any():
        last_mask = last_mask | ~covered
    rows.append({"branch": f"> {thresholds[-1]:.6g}", "rows": int(last_mask.sum())})
    return rows


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


def model_metrics(df: pd.DataFrame, target: str) -> pd.DataFrame:
    target_kind = infer_target_kind(df[target])
    preds = tree_predictions(df, target)
    y = df[target]
    rows: list[dict[str, Any]] = []

    if target_kind == "binary":
        positive_class = choose_positive_class(y)
        y_binary = (y == positive_class).astype(int)
        auc = binary_auc(y_binary, preds["positive_rate"])
        accuracy = float((preds["prediction"] == y).mean())
        rows.append({"metric": "target_type", "value": "binary"})
        rows.append({"metric": "positive_class", "value": positive_class})
        rows.append({"metric": "default_rate", "value": float(y_binary.mean())})
        rows.append({"metric": "auc", "value": auc})
        rows.append({"metric": "gini", "value": None if auc is None else 2 * auc - 1})
        rows.append({"metric": "accuracy", "value": accuracy})
    elif target_kind == "regression":
        y_num = pd.to_numeric(y, errors="coerce")
        pred_num = pd.to_numeric(preds["prediction"], errors="coerce")
        mask = y_num.notna() & pred_num.notna()
        residual = y_num[mask] - pred_num[mask]
        baseline = y_num[mask] - y_num[mask].mean()
        sse = float((residual**2).sum())
        sst = float((baseline**2).sum())
        rows.append({"metric": "target_type", "value": "regression"})
        rows.append({"metric": "rmse", "value": float(np.sqrt((residual**2).mean())) if len(residual) else None})
        rows.append({"metric": "mae", "value": float(residual.abs().mean()) if len(residual) else None})
        rows.append({"metric": "r2", "value": None if sst == 0 else 1 - sse / sst})
    else:
        rows.append({"metric": "target_type", "value": "classification"})
        rows.append({"metric": "accuracy", "value": float((preds["prediction"] == y).mean())})

    total_gain, root_impurity, leaf_impurity = tree_total_gain(df, target)
    rows.append({"metric": "tree_total_gain", "value": total_gain})
    rows.append({"metric": "root_impurity", "value": root_impurity})
    rows.append({"metric": "weighted_leaf_impurity", "value": leaf_impurity})
    rows.append({"metric": "leaf_count", "value": len(current_leaves())})
    return pd.DataFrame(rows)


def leaf_performance_rows(df: pd.DataFrame, target: str, data_key: str) -> list[dict[str, Any]]:
    target_kind = infer_target_kind(df[target])
    rows: list[dict[str, Any]] = []
    for leaf in current_leaves():
        summary = node_summary(df, target, leaf["row_idx"])
        selected_feature = st.session_state.get(node_feature_key(data_key, target, leaf["id"]), "")
        row = {
            "leaf": leaf["id"],
            "selected_variable": selected_feature,
            "n": summary["n"],
            "predict": summary["prediction"],
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


def main() -> None:
    st.set_page_config(page_title="Interactive entropy tree", layout="wide")
    st.title("Interactive entropy decision tree")

    work_id = ensure_work_id()
    checkpoint = load_work_checkpoint(work_id)
    restore_checkpoint_ui_state(checkpoint)

    query_session = load_session_dataframe_from_query()
    query_data_id = query_session[2] if query_session is not None else None
    if query_data_id and st.session_state.get("_last_query_data_id") != query_data_id:
        st.session_state["data_source_choice"] = "Session DataFrame"
        st.session_state["_last_query_data_id"] = query_data_id

    source_options = ["Session DataFrame", "CSV / Excel Upload", "SQL", "Demo"]
    default_source = "Session DataFrame" if query_session is not None else "Demo"
    if st.session_state.get("data_source_choice") == "CSV Upload":
        st.session_state["data_source_choice"] = "CSV / Excel Upload"
    if st.session_state.get("data_source_choice") not in source_options:
        st.session_state["data_source_choice"] = default_source
    source_choice = st.sidebar.radio(
        "Data source",
        source_options,
        index=source_options.index(default_source),
        key="data_source_choice",
    )

    uploaded_name: str | None = None
    source_metadata: dict[str, Any] = {}
    source_data_id: str | None = None
    data_source = "demo"
    restored_upload = False
    if source_choice == "Session DataFrame":
        if query_session is None:
            st.info("No valid session DataFrame was found. Choose another data source in the sidebar.")
            st.stop()
        df, source_metadata, source_data_id, data_key = query_session
        uploaded_name = str(source_metadata.get("name") or "Session DataFrame")
        data_source = str(source_metadata.get("source") or "session")
    elif source_choice == "CSV / Excel Upload":
        uploaded = st.sidebar.file_uploader("CSV veya Excel yukle", type=["csv", "xlsx", "xls"])
        if uploaded is None:
            restored_dataframe = restore_checkpoint_dataframe(checkpoint)
            if restored_dataframe is not None:
                df, uploaded_name, data_key = restored_dataframe
                data_source = "uploaded"
                restored_upload = True
            else:
                st.info("Upload a CSV or Excel file from the sidebar.")
                st.stop()
        else:
            try:
                df = read_uploaded_table(uploaded)
            except Exception as exc:
                st.sidebar.error(f"File load failed: {exc}")
                st.stop()
            uploaded_name = uploaded.name
            data_source = "uploaded"
            upload_key = uploaded_data_key(uploaded.name, df)
            source_metadata = {
                "source": "uploaded",
                "name": uploaded.name,
                "upload_name": uploaded.name,
                "fingerprint": dataframe_fingerprint(df),
            }
            cache_key = "_uploaded_session"
            cached_upload = st.session_state.get(cache_key, {})
            if cached_upload.get("upload_key") == upload_key:
                source_data_id = cached_upload.get("data_id")
            else:
                source_data_id = save_source_session(
                    df,
                    source="uploaded",
                    name=uploaded.name,
                    metadata={"upload_name": uploaded.name},
                )
                st.session_state[cache_key] = {"upload_key": upload_key, "data_id": source_data_id}
            data_key = session_data_key(str(source_data_id), df, source_metadata)
            if normalize_data_id(st.query_params.get(DATA_ID_QUERY_PARAM)) != source_data_id:
                st.query_params[DATA_ID_QUERY_PARAM] = source_data_id
                st.rerun()
    elif source_choice == "SQL":
        render_sql_source_loader()
    else:
        df = make_demo_data()
        data_key = "demo"

    st.sidebar.caption(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
    if restored_upload and uploaded_name is not None:
        st.sidebar.caption(f"Restored uploaded data: {uploaded_name}")
    if source_data_id:
        st.sidebar.caption(f"Data id: {source_data_id}")
    st.sidebar.caption(f"Autosave work id: {work_id}")
    if st.session_state.get("_checkpoint_error"):
        st.sidebar.warning(f"Autosave failed: {st.session_state['_checkpoint_error']}")

    checkpoint_data = checkpoint.get("data") if isinstance(checkpoint, dict) else {}
    checkpoint_data_key = checkpoint_data.get("data_key") if isinstance(checkpoint_data, dict) else None
    saved_target = checkpoint.get("target") if isinstance(checkpoint, dict) and checkpoint_data_key == data_key else None
    source_target = source_metadata_value(source_metadata, "target")
    target_options = df.columns.tolist()
    default_target_index = target_options.index("risk_flag") if "risk_flag" in target_options else len(target_options) - 1
    if saved_target in target_options:
        default_target_index = target_options.index(saved_target)
    elif source_target in target_options:
        default_target_index = target_options.index(source_target)
    target = st.sidebar.selectbox(
        "Target",
        options=target_options,
        index=default_target_index,
    )
    target_kind = infer_target_kind(df[target])
    if target_kind == "binary":
        positive_class_options = list(df[target].dropna().unique())
        positive_class_key = f"positive_class::{data_key}::{target}"
        checkpoint_positive_class = (
            checkpoint.get("positive_class")
            if isinstance(checkpoint, dict) and checkpoint_data_key == data_key
            else None
        )
        remembered_positive_class = st.session_state.get(positive_class_key, checkpoint_positive_class)
        default_positive_class = choose_positive_class(
            df[target],
            preferred=remembered_positive_class,
            use_session_default=False,
        )
        positive_class = st.sidebar.selectbox(
            "Positive class",
            options=positive_class_options,
            index=class_option_index(positive_class_options, default_positive_class),
            key=positive_class_key,
            format_func=lambda value: str(value),
        )
        st.session_state[POSITIVE_CLASS_SESSION_KEY] = positive_class
    else:
        st.session_state[POSITIVE_CLASS_SESSION_KEY] = None

    default_features = [c for c in df.columns if c != target]
    checkpoint_features = (
        checkpoint.get("selected_features")
        if isinstance(checkpoint, dict) and checkpoint_data_key == data_key
        else None
    )
    source_features = source_metadata_value(source_metadata, "features")
    if isinstance(checkpoint_features, list):
        default_selected_features = [str(feature) for feature in checkpoint_features if str(feature) in default_features]
    elif isinstance(source_features, list):
        default_selected_features = [str(feature) for feature in source_features if str(feature) in default_features]
    else:
        default_selected_features = default_features
    selected_features = st.sidebar.multiselect(
        "Available split variables",
        default_features,
        default=default_selected_features,
    )
    features = selected_features

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

    min_leaf = safe_int(min_leaf_input, default=20, minimum=1)
    max_thresholds = safe_int(max_thresholds_input, default=40, minimum=1)
    max_numeric_bins = safe_int(max_numeric_bins_input, default=4, minimum=2)
    max_categories = safe_int(max_categories_input, default=20, minimum=1)
    max_category_groups = safe_int(max_category_groups_input, default=5, minimum=2)

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
            "Minimum split information gain",
            value=safe_float(saved_auto_parameters.get("min_information_gain"), default=0.005),
            step=0.001,
            format="%.6f",
        )
        auto_max_depth = safe_int(auto_max_depth_input, default=3, minimum=1)
        auto_max_leaves = safe_int(auto_max_leaves_input, default=12, minimum=2)
        auto_min_gain = safe_float(auto_min_gain_input, default=0.005)
        auto_parameters = {
            "max_depth": auto_max_depth,
            "max_leaves": auto_max_leaves,
            "min_information_gain": auto_min_gain,
        }

        if st.button("Build optimal tree", width="stretch", disabled=not features):
            split_count = build_optimal_tree(
                df=df,
                target=target,
                features=features,
                min_leaf=min_leaf,
                max_thresholds=max_thresholds,
                max_categories=max_categories,
                max_numeric_bins=max_numeric_bins,
                max_category_groups=max_category_groups,
                max_depth=auto_max_depth,
                max_leaves=auto_max_leaves,
                min_information_gain=auto_min_gain,
            )
            st.session_state.auto_tree_message = f"Optimal tree built with {split_count} split(s)."
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
                st.session_state.current_node_id = 0
                st.session_state.tree_zoom = recommended_tree_zoom()
                st.session_state["_tree_import_message"] = (
                    f"Imported editable tree with {len(imported_tree)} node(s)."
                )
                save_and_rerun(imported_selected_features or features)
        if st.session_state.get("_tree_import_message"):
            st.caption(st.session_state["_tree_import_message"])

    if st.sidebar.button("Reset tree", width="stretch"):
        init_tree(df)
        save_and_rerun()

    if st.sidebar.button("Undo last split", width="stretch", disabled=not st.session_state.get("split_history")):
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
        leaf_rows = leaf_performance_rows(df, target, data_key)
        if leaf_rows:
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

    with st.expander("Model performance", expanded=True):
        st.dataframe(arrow_safe_dataframe(model_metrics(df, target)), hide_index=True, width="stretch")

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
            else:
                all_candidates = candidate_splits(
                    df=df,
                    target=target,
                    features=features,
                    row_idx=current["row_idx"],
                    min_leaf=int(min_leaf),
                    max_thresholds=int(max_thresholds),
                    max_categories=int(max_categories),
                    max_numeric_bins=int(max_numeric_bins),
                    max_category_groups=int(max_category_groups),
                )

                if not all_candidates:
                    st.warning("No valid split found for current settings.")
                else:
                    feature_rows = feature_summary_rows(all_candidates, features)
                    feature_stats = {row["variable"]: row for row in feature_rows}
                    leaf_feature_options = ordered_features_by_gain(features, feature_stats)
                    positive_candidates = [
                        candidate
                        for candidate in all_candidates
                        if candidate.information_gain > MIN_INFORMATION_GAIN_EPSILON
                    ]
                    if not leaf_feature_options or not positive_candidates:
                        st.warning("No split with positive information gain found for this leaf.")
                        st.dataframe(
                            arrow_safe_dataframe(
                                pd.DataFrame(
                                    feature_summary_rows(
                                        all_candidates,
                                        features,
                                        include_zero_gain=True,
                                    )
                                )
                            ),
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "total_information_gain": st.column_config.NumberColumn(format="%.6f"),
                                "best_information_gain": st.column_config.NumberColumn(format="%.6f"),
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
                            f"{feature} | total IG={feature_stats[feature]['total_information_gain']:.6f} | "
                            f"best={feature_stats[feature]['best_information_gain']:.6f}"
                        ),
                    )

                    selected_stats = feature_stats[selected_feature]
                    current_total_gain, _, _ = tree_total_gain(df, target)
                    score_name = split_score_name(df[target])
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric(f"Tree total {score_name}", f"{current_total_gain:.6f}")
                    metric_col2.metric(f"{selected_feature} total", f"{selected_stats['total_information_gain']:.6f}")
                    metric_col3.metric(f"{selected_feature} best", f"{selected_stats['best_information_gain']:.6f}")

                    st.dataframe(
                        arrow_safe_dataframe(pd.DataFrame(feature_summary_rows(all_candidates, features, [selected_feature]))),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "total_information_gain": st.column_config.NumberColumn(format="%.6f"),
                            "best_information_gain": st.column_config.NumberColumn(format="%.6f"),
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
                        selected_candidate = max(feature_candidates, key=lambda c: c.information_gain)
                        rows = [
                            {
                                "split_type": selected_candidate.split_type,
                                "split": selected_candidate.label,
                                "branches": selected_candidate.branch_count,
                                score_name: selected_candidate.information_gain,
                                "weighted_tree_delta": candidate_total_gain_delta(df, selected_candidate, current["row_idx"]),
                                "child_weighted_impurity": selected_candidate.weighted_entropy,
                                "branch_rows": " / ".join(str(n) for n in selected_candidate.branch_ns),
                                "branch_impurity": " / ".join(f"{x:.3f}" for x in selected_candidate.branch_entropies),
                            }
                        ]

                        st.dataframe(
                            arrow_safe_dataframe(pd.DataFrame(rows)),
                            hide_index=True,
                            width="stretch",
                            column_config={
                                score_name: st.column_config.NumberColumn(format="%.6f"),
                                "weighted_tree_delta": st.column_config.NumberColumn(format="%.6f"),
                                "child_weighted_impurity": st.column_config.NumberColumn(format="%.6f"),
                            },
                        )

                        st.dataframe(
                            arrow_safe_dataframe(pd.DataFrame(candidate_impact_rows(df, target, current["row_idx"], selected_candidate))),
                            hide_index=True,
                            width="stretch",
                        )
                        if st.button(
                            f"Apply auto {selected_feature} split",
                            key=f"apply_split_{current['id']}_{selected_feature}",
                            type="primary",
                            width="stretch",
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
                        if manual_text.strip():
                            try:
                                thresholds = parse_threshold_text(manual_text)
                                branch_rows = manual_numeric_branch_rows(
                                    df,
                                    current["row_idx"],
                                    selected_feature,
                                    thresholds,
                                )
                                if branch_rows:
                                    st.dataframe(
                                        arrow_safe_dataframe(pd.DataFrame(branch_rows)),
                                        hide_index=True,
                                        width="stretch",
                                    )
                                manual_candidate = score_numeric_manual_bins(
                                    df=df,
                                    target=target,
                                    row_idx=current["row_idx"],
                                    feature=selected_feature,
                                    thresholds=thresholds,
                                    min_leaf=int(min_leaf),
                                )
                                if manual_candidate is None:
                                    low_branches = [
                                        row
                                        for row in branch_rows
                                        if int(row["rows"]) < int(min_leaf)
                                    ]
                                    if low_branches:
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
                        if seed_col1.button("All in one", key=f"group_all_{group_key}", width="stretch"):
                            st.session_state[group_key] = [level_texts.copy()]
                            save_and_rerun()
                        if seed_col2.button("One per value", key=f"group_single_{group_key}", width="stretch"):
                            st.session_state[group_key] = [[value] for value in level_texts]
                            save_and_rerun()
                        if seed_col3.button("Profile groups", key=f"group_profile_{group_key}", width="stretch"):
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
                        if st.button("Move selected to new group", key=f"group_split_{group_key}", width="stretch"):
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
                        if st.button("Merge selected groups", key=f"group_merge_button_{group_key}", width="stretch"):
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

                    if manual_candidate is None:
                        st.caption("Enter a valid manual split to preview its impact.")
                    else:
                        st.dataframe(
                            arrow_safe_dataframe(pd.DataFrame(candidate_impact_rows(df, target, current["row_idx"], manual_candidate))),
                            hide_index=True,
                            width="stretch",
                        )
                        if st.button(
                            f"Apply manual {selected_feature} split",
                            key=f"apply_manual_split_{current['id']}_{selected_feature}",
                            width="stretch",
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
