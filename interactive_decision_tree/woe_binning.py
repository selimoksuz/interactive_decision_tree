from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


WOE_SCHEMA_VERSION = 1
WOE_EPSILON = 1e-8


@dataclass(frozen=True)
class WoeBuildConfig:
    max_bins: int = 6
    min_bin_size: float = 0.05
    monotonic_trend: str = "auto"
    missing_separate: bool = True
    blank_as_missing: bool = True
    special_values: tuple[str, ...] = ()
    protected_special: bool = True
    engine: str = "auto"


def parse_special_values(text: str | None) -> list[str]:
    if not text:
        return []
    raw_values: list[str] = []
    for chunk in str(text).replace("\n", ",").split(","):
        value = chunk.strip()
        if value:
            raw_values.append(value)
    return list(dict.fromkeys(raw_values))


def json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def json_safe_list(values: list[Any] | tuple[Any, ...]) -> list[Any]:
    return [json_safe_value(value) for value in values]


def infer_feature_kind(series: pd.Series) -> str:
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"


def safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def is_blank_like(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == "object":
        return series.astype("string").str.strip().fillna("") == ""
    return pd.Series(False, index=series.index)


def missing_mask(series: pd.Series, blank_as_missing: bool = True) -> pd.Series:
    mask = series.isna()
    if blank_as_missing:
        mask = mask | is_blank_like(series)
    return mask.fillna(False)


def special_mask(series: pd.Series, special_values: list[str] | tuple[str, ...]) -> pd.Series:
    if not special_values:
        return pd.Series(False, index=series.index)
    if pd.api.types.is_numeric_dtype(series):
        numeric_values = [
            parsed
            for parsed in (safe_float_or_none(value) for value in special_values)
            if parsed is not None
        ]
        if not numeric_values:
            return pd.Series(False, index=series.index)
        numeric_series = pd.to_numeric(series, errors="coerce")
        return numeric_series.isin(numeric_values).fillna(False)
    text_values = {str(value) for value in special_values}
    return series.astype("string").isin(text_values).fillna(False)


def make_bin_id(index: int) -> str:
    return f"b{index:03d}"


def format_bound(value: float | None, fallback: str) -> str:
    if value is None:
        return fallback
    if abs(value) >= 1000:
        return f"{value:,.4g}"
    return f"{value:.6g}"


def numeric_bin_label(lower: float | None, upper: float | None) -> str:
    if lower is None and upper is None:
        return "all values"
    if lower is None:
        return f"(-inf, {format_bound(upper, 'inf')}]"
    if upper is None:
        return f"({format_bound(lower, '-inf')}, inf)"
    return f"({format_bound(lower, '-inf')}, {format_bound(upper, 'inf')}]"


def quantile_splits(values: pd.Series, max_bins: int, min_bin_size: float) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.nunique() <= 1:
        return []
    max_bins = max(2, int(max_bins))
    quantiles = np.linspace(0, 1, max_bins + 1)[1:-1]
    splits = numeric.quantile(quantiles).dropna().unique().tolist()
    splits = sorted({float(value) for value in splits if math.isfinite(float(value))})
    if not splits:
        return []

    minimum_rows = max(1, int(math.ceil(len(numeric) * float(min_bin_size))))
    accepted: list[float] = []
    previous = -np.inf
    for split in splits:
        rows = int(((numeric > previous) & (numeric <= split)).sum())
        if rows >= minimum_rows:
            accepted.append(split)
            previous = split
    if accepted:
        last_rows = int((numeric > accepted[-1]).sum())
        if last_rows < minimum_rows:
            accepted = accepted[:-1]
    return accepted


def optbinning_numeric_splits(
    values: pd.Series,
    y_event: pd.Series,
    feature: str,
    config: WoeBuildConfig,
) -> list[float] | None:
    if config.engine not in {"auto", "optbinning"}:
        return None
    try:
        from optbinning import OptimalBinning  # type: ignore
    except Exception:
        return None

    x = pd.to_numeric(values, errors="coerce")
    valid = x.notna() & y_event.notna()
    if valid.sum() < 2 or x[valid].nunique() <= 1:
        return []

    monotonic = None if config.monotonic_trend in {"auto", "none"} else config.monotonic_trend
    try:
        optb = OptimalBinning(
            name=str(feature),
            dtype="numerical",
            max_n_bins=max(2, int(config.max_bins)),
            min_bin_size=float(config.min_bin_size),
            monotonic_trend=monotonic,
        )
        optb.fit(x[valid].to_numpy(), y_event[valid].astype(int).to_numpy())
        return sorted(
            {
                float(split)
                for split in getattr(optb, "splits", [])
                if split is not None and math.isfinite(float(split))
            }
        )
    except Exception:
        if config.engine == "optbinning":
            raise
        return None


def numeric_bins_from_splits(splits: list[float], start_index: int = 1) -> list[dict[str, Any]]:
    edges: list[tuple[float | None, float | None]] = []
    if not splits:
        edges.append((None, None))
    else:
        edges.append((None, splits[0]))
        for left, right in zip(splits, splits[1:]):
            edges.append((left, right))
        edges.append((splits[-1], None))

    bins: list[dict[str, Any]] = []
    for offset, (lower, upper) in enumerate(edges, start=start_index):
        bins.append(
            {
                "bin_id": make_bin_id(offset),
                "kind": "normal",
                "label": numeric_bin_label(lower, upper),
                "lower": lower,
                "upper": upper,
                "values": [],
                "assigned_woe": None,
                "protected": False,
                "note": "",
            }
        )
    return bins


def category_event_profile(frame: pd.DataFrame, feature: str, target: str, positive_class: Any) -> pd.DataFrame:
    work = frame[[feature, target]].copy()
    work = work[work[target].notna()]
    work["_value"] = work[feature].astype("string")
    work["_event"] = (work[target] == positive_class).astype(int)
    grouped = work.groupby("_value", dropna=False)["_event"].agg(["count", "sum"]).reset_index()
    grouped["event_rate"] = grouped["sum"] / grouped["count"].replace(0, np.nan)
    return grouped.sort_values(["event_rate", "count", "_value"], ascending=[True, False, True])


def categorical_bins_from_profile(profile: pd.DataFrame, max_bins: int, start_index: int = 1) -> list[dict[str, Any]]:
    if profile.empty:
        return []
    max_bins = max(1, int(max_bins))
    values = profile["_value"].astype(str).tolist()
    if len(values) <= max_bins:
        groups = [[value] for value in values]
    else:
        groups = [list(chunk) for chunk in np.array_split(values, max_bins) if len(chunk)]

    bins: list[dict[str, Any]] = []
    for offset, group in enumerate(groups, start=start_index):
        label = ", ".join(group[:4]) + (" ..." if len(group) > 4 else "")
        bins.append(
            {
                "bin_id": make_bin_id(offset),
                "kind": "normal",
                "label": label,
                "lower": None,
                "upper": None,
                "values": list(group),
                "assigned_woe": None,
                "protected": False,
                "note": "",
            }
        )
    return bins


def special_bin(values: list[str], protected: bool = True) -> dict[str, Any]:
    label = "Special: " + ", ".join(values[:6]) + (" ..." if len(values) > 6 else "")
    return {
        "bin_id": "special_001",
        "kind": "special",
        "label": label,
        "lower": None,
        "upper": None,
        "values": list(values),
        "assigned_woe": None,
        "protected": bool(protected),
        "note": "",
    }


def missing_bin() -> dict[str, Any]:
    return {
        "bin_id": "missing",
        "kind": "missing",
        "label": "Missing",
        "lower": None,
        "upper": None,
        "values": [],
        "assigned_woe": None,
        "protected": True,
        "note": "",
    }


def build_initial_spec(
    df: pd.DataFrame,
    target: str,
    feature: str,
    positive_class: Any,
    config: WoeBuildConfig | None = None,
) -> dict[str, Any]:
    config = config or WoeBuildConfig()
    feature_kind = infer_feature_kind(df[feature])
    y_event = (df[target] == positive_class).astype(int)
    miss = missing_mask(df[feature], config.blank_as_missing)
    special = special_mask(df[feature], config.special_values) & ~miss
    normal_frame = df.loc[~miss & ~special & df[target].notna(), [feature, target]]

    bins: list[dict[str, Any]] = []
    if config.missing_separate and bool(miss.any()):
        bins.append(missing_bin())
    if config.special_values:
        bins.append(special_bin(list(config.special_values), config.protected_special))

    engine_used = "fallback"
    if feature_kind == "numeric":
        opt_splits = optbinning_numeric_splits(normal_frame[feature], y_event.loc[normal_frame.index], feature, config)
        if opt_splits is not None:
            splits = opt_splits
            engine_used = "optbinning"
        else:
            splits = quantile_splits(normal_frame[feature], config.max_bins, config.min_bin_size)
        bins.extend(numeric_bins_from_splits(splits, start_index=1))
    else:
        profile = category_event_profile(normal_frame, feature, target, positive_class)
        bins.extend(categorical_bins_from_profile(profile, config.max_bins, start_index=1))

    return {
        "schema_version": WOE_SCHEMA_VERSION,
        "feature": str(feature),
        "feature_kind": feature_kind,
        "positive_class": json_safe_value(positive_class),
        "config": {
            "max_bins": int(config.max_bins),
            "min_bin_size": float(config.min_bin_size),
            "monotonic_trend": str(config.monotonic_trend),
            "missing_separate": bool(config.missing_separate),
            "blank_as_missing": bool(config.blank_as_missing),
            "special_values": list(config.special_values),
            "protected_special": bool(config.protected_special),
            "engine": str(config.engine),
            "engine_used": engine_used,
        },
        "bins": bins,
    }


def copy_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(spec)


def bin_contains_value(series: pd.Series, spec: dict[str, Any], bin_spec: dict[str, Any]) -> pd.Series:
    kind = bin_spec.get("kind")
    if kind == "missing":
        return missing_mask(series, bool(spec.get("config", {}).get("blank_as_missing", True)))
    if kind == "special":
        return special_mask(series, [str(value) for value in bin_spec.get("values", [])])

    feature_kind = spec.get("feature_kind")
    if feature_kind == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        lower = safe_float_or_none(bin_spec.get("lower"))
        upper = safe_float_or_none(bin_spec.get("upper"))
        mask = numeric.notna()
        if lower is not None:
            mask = mask & (numeric > lower)
        if upper is not None:
            mask = mask & (numeric <= upper)
        return mask.fillna(False)

    values = {str(value) for value in bin_spec.get("values", [])}
    if not values:
        return pd.Series(False, index=series.index)
    return series.astype("string").isin(values).fillna(False)


def assign_bins(series: pd.Series, spec: dict[str, Any]) -> pd.Series:
    assigned = pd.Series(pd.NA, index=series.index, dtype="object")
    for bin_spec in spec.get("bins", []):
        bin_id = str(bin_spec.get("bin_id"))
        mask = bin_contains_value(series, spec, bin_spec) & assigned.isna()
        assigned.loc[mask] = bin_id
    assigned = assigned.astype("object")
    assigned.loc[assigned.isna()] = "__unmapped__"
    return assigned


def bin_order(spec: dict[str, Any], assigned: pd.Series | None = None) -> list[str]:
    ids = [str(bin_spec.get("bin_id")) for bin_spec in spec.get("bins", [])]
    if assigned is not None and bool((assigned == "__unmapped__").any()):
        ids.append("__unmapped__")
    return ids


def assigned_woe_value(bin_spec: dict[str, Any]) -> float | None:
    return safe_float_or_none(bin_spec.get("assigned_woe"))


def build_bin_table(
    df: pd.DataFrame,
    target: str,
    spec: dict[str, Any],
    positive_class: Any,
    dataset_name: str = "Train",
) -> pd.DataFrame:
    feature = str(spec["feature"])
    assigned = assign_bins(df[feature], spec)
    y_valid = df[target].notna()
    event = (df[target] == positive_class) & y_valid
    non_event = (~event) & y_valid
    total_events = int(event.sum())
    total_non_events = int(non_event.sum())
    total_rows = int(y_valid.sum())
    order = bin_order(spec, assigned)
    spec_by_id = {str(bin_spec.get("bin_id")): bin_spec for bin_spec in spec.get("bins", [])}
    rows: list[dict[str, Any]] = []

    for position, bin_id in enumerate(order, start=1):
        bin_spec = spec_by_id.get(bin_id, {})
        mask = (assigned == bin_id) & y_valid
        event_count = int((mask & event).sum())
        non_event_count = int((mask & non_event).sum())
        count = event_count + non_event_count
        event_dist = (event_count + WOE_EPSILON) / (total_events + WOE_EPSILON * max(1, len(order)))
        non_event_dist = (non_event_count + WOE_EPSILON) / (
            total_non_events + WOE_EPSILON * max(1, len(order))
        )
        calculated_woe = float(math.log(non_event_dist / event_dist))
        assigned_woe = assigned_woe_value(bin_spec)
        export_woe = calculated_woe if assigned_woe is None else assigned_woe
        rows.append(
            {
                "dataset": dataset_name,
                "variable": feature,
                "bin_id": bin_id,
                "bin_order": position,
                "kind": str(bin_spec.get("kind", "unmapped")),
                "label": str(bin_spec.get("label", "Unmapped")),
                "lower": bin_spec.get("lower"),
                "upper": bin_spec.get("upper"),
                "values": ", ".join(str(value) for value in bin_spec.get("values", [])),
                "count": count,
                "event_count": event_count,
                "non_event_count": non_event_count,
                "event_rate": None if count == 0 else event_count / count,
                "all_concentration": None if total_rows == 0 else count / total_rows,
                "event_concentration": None if total_events == 0 else event_count / total_events,
                "non_event_concentration": None if total_non_events == 0 else non_event_count / total_non_events,
                "calculated_woe": calculated_woe,
                "assigned_woe": assigned_woe,
                "export_woe": export_woe,
                "calculated_iv": float((non_event_dist - event_dist) * calculated_woe),
                "export_iv": float((non_event_dist - event_dist) * export_woe),
                "protected": bool(bin_spec.get("protected", False)),
                "note": str(bin_spec.get("note", "")),
            }
        )
    return pd.DataFrame(rows)


def binary_auc(y_true: pd.Series, score: pd.Series) -> float | None:
    valid = y_true.notna() & score.notna()
    if not bool(valid.any()):
        return None
    y = y_true[valid].astype(int)
    scores = score[valid].astype(float)
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = scores.rank(method="average")
    rank_sum = float(ranks[y == 1].sum())
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def monotonicity_from_table(table: pd.DataFrame) -> dict[str, Any]:
    normal = table[table["kind"] == "normal"].copy()
    rates = pd.to_numeric(normal["event_rate"], errors="coerce").dropna().tolist()
    if len(rates) <= 2:
        return {"direction": "not_enough_bins", "is_monotonic": True, "violation_count": 0}
    diffs = np.diff(rates)
    increasing_violations = int((diffs < -1e-12).sum())
    decreasing_violations = int((diffs > 1e-12).sum())
    if increasing_violations == 0:
        return {"direction": "increasing", "is_monotonic": True, "violation_count": 0}
    if decreasing_violations == 0:
        return {"direction": "decreasing", "is_monotonic": True, "violation_count": 0}
    return {
        "direction": "mixed",
        "is_monotonic": False,
        "violation_count": min(increasing_violations, decreasing_violations),
    }


def evaluate_spec(
    df: pd.DataFrame,
    target: str,
    spec: dict[str, Any],
    positive_class: Any,
    dataset_name: str = "Train",
) -> dict[str, Any]:
    table = build_bin_table(df, target, spec, positive_class, dataset_name=dataset_name)
    feature = str(spec["feature"])
    assigned = assign_bins(df[feature], spec)
    calc_map = dict(zip(table["bin_id"], table["calculated_woe"]))
    export_map = dict(zip(table["bin_id"], table["export_woe"]))
    y = (df[target] == positive_class).astype(int)
    calc_score = -assigned.map(calc_map).astype(float)
    export_score = -assigned.map(export_map).astype(float)
    calc_auc = binary_auc(y, calc_score)
    export_auc = binary_auc(y, export_score)
    monotonicity = monotonicity_from_table(table)
    metrics = {
        "dataset": dataset_name,
        "variable": feature,
        "rows": int(len(df)),
        "scored_rows": int(df[target].notna().sum()),
        "bin_count": int(len(table)),
        "normal_bin_count": int((table["kind"] == "normal").sum()),
        "calculated_iv": float(table["calculated_iv"].sum()),
        "export_iv": float(table["export_iv"].sum()),
        "calculated_auc": calc_auc,
        "calculated_gini": None if calc_auc is None else float(2 * calc_auc - 1),
        "export_auc": export_auc,
        "export_gini": None if export_auc is None else float(2 * export_auc - 1),
        "monotonic_direction": monotonicity["direction"],
        "is_monotonic": bool(monotonicity["is_monotonic"]),
        "monotonic_violation_count": int(monotonicity["violation_count"]),
        "manual_woe_bins": int(table["assigned_woe"].notna().sum()),
        "engine_used": str(spec.get("config", {}).get("engine_used", "fallback")),
    }
    return {"table": table, "metrics": metrics}


def evaluate_original_current(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    target: str,
    original_spec: dict[str, Any],
    current_spec: dict[str, Any],
    positive_class: Any,
) -> dict[str, Any]:
    result = {
        "original_train": evaluate_spec(train_df, target, original_spec, positive_class, "Train"),
        "current_train": evaluate_spec(train_df, target, current_spec, positive_class, "Train"),
        "original_test": None,
        "current_test": None,
    }
    if test_df is not None:
        result["original_test"] = evaluate_spec(test_df, target, original_spec, positive_class, "Test")
        result["current_test"] = evaluate_spec(test_df, target, current_spec, positive_class, "Test")
    return result


def set_assigned_woe_from_table(spec: dict[str, Any], edited_table: pd.DataFrame) -> dict[str, Any]:
    updated = copy_spec(spec)
    if "bin_id" not in edited_table.columns:
        return updated
    edited_by_id = edited_table.set_index("bin_id").to_dict("index")
    for bin_spec in updated.get("bins", []):
        bin_id = str(bin_spec.get("bin_id"))
        if bin_id not in edited_by_id:
            continue
        row = edited_by_id[bin_id]
        bin_spec["assigned_woe"] = safe_float_or_none(row.get("assigned_woe"))
        if "note" in row:
            bin_spec["note"] = str(row.get("note") or "")
    return updated


def normal_bins(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [bin_spec for bin_spec in spec.get("bins", []) if bin_spec.get("kind") == "normal"]


def special_prefix_bins(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [copy.deepcopy(bin_spec) for bin_spec in spec.get("bins", []) if bin_spec.get("kind") != "normal"]


def renumber_normal_bins(prefix_bins: list[dict[str, Any]], normal: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = copy.deepcopy(prefix_bins)
    for idx, bin_spec in enumerate(normal, start=1):
        new_bin = copy.deepcopy(bin_spec)
        new_bin["bin_id"] = make_bin_id(idx)
        out.append(new_bin)
    return out


def apply_numeric_cutpoints(spec: dict[str, Any], cutpoints: list[float]) -> dict[str, Any]:
    if spec.get("feature_kind") != "numeric":
        return copy_spec(spec)
    updated = copy_spec(spec)
    clean_cutpoints = sorted({float(value) for value in cutpoints if math.isfinite(float(value))})
    prefix = special_prefix_bins(updated)
    updated["bins"] = renumber_normal_bins(prefix, numeric_bins_from_splits(clean_cutpoints, start_index=1))
    return updated


def merge_with_next(spec: dict[str, Any], bin_id: str) -> dict[str, Any]:
    updated = copy_spec(spec)
    prefix = special_prefix_bins(updated)
    normal = normal_bins(updated)
    index = next((idx for idx, bin_spec in enumerate(normal) if str(bin_spec.get("bin_id")) == str(bin_id)), None)
    if index is None or index >= len(normal) - 1:
        return updated
    left = copy.deepcopy(normal[index])
    right = normal[index + 1]
    if updated.get("feature_kind") == "numeric":
        left["upper"] = right.get("upper")
        left["label"] = numeric_bin_label(safe_float_or_none(left.get("lower")), safe_float_or_none(left.get("upper")))
        left["assigned_woe"] = None
    else:
        left["values"] = list(left.get("values", [])) + list(right.get("values", []))
        left["label"] = ", ".join(str(value) for value in left["values"][:4]) + (
            " ..." if len(left["values"]) > 4 else ""
        )
        left["assigned_woe"] = None
    merged = normal[:index] + [left] + normal[index + 2 :]
    updated["bins"] = renumber_normal_bins(prefix, merged)
    return updated


def apply_categorical_groups(spec: dict[str, Any], groups: list[list[str]]) -> dict[str, Any]:
    if spec.get("feature_kind") != "categorical":
        return copy_spec(spec)
    updated = copy_spec(spec)
    prefix = special_prefix_bins(updated)
    normal: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        clean_group = [str(value).strip() for value in group if str(value).strip()]
        clean_group = [value for value in clean_group if value not in seen]
        if not clean_group:
            continue
        seen.update(clean_group)
        label = ", ".join(clean_group[:4]) + (" ..." if len(clean_group) > 4 else "")
        normal.append(
            {
                "bin_id": "",
                "kind": "normal",
                "label": label,
                "lower": None,
                "upper": None,
                "values": clean_group,
                "assigned_woe": None,
                "protected": False,
                "note": "",
            }
        )
    updated["bins"] = renumber_normal_bins(prefix, normal)
    return updated


def cutpoints_from_spec(spec: dict[str, Any]) -> list[float]:
    cutpoints: list[float] = []
    for bin_spec in normal_bins(spec):
        upper = safe_float_or_none(bin_spec.get("upper"))
        if upper is not None:
            cutpoints.append(upper)
    return sorted(set(cutpoints))


def categorical_groups_from_spec(spec: dict[str, Any]) -> list[list[str]]:
    return [
        [str(value) for value in bin_spec.get("values", [])]
        for bin_spec in normal_bins(spec)
    ]


def parse_cutpoints(text: str | None) -> list[float]:
    if not text:
        return []
    cutpoints: list[float] = []
    for chunk in str(text).replace("\n", ",").split(","):
        value = safe_float_or_none(chunk.strip())
        if value is not None:
            cutpoints.append(value)
    return sorted(set(cutpoints))


def parse_category_groups(text: str | None) -> list[list[str]]:
    groups: list[list[str]] = []
    for line in str(text or "").splitlines():
        values = [value.strip() for value in line.split(",") if value.strip()]
        if values:
            groups.append(values)
    return groups
