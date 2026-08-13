from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest


WOE_SCHEMA_VERSION = 1
WOE_EPSILON = 1e-8
WOE_BINOMIAL_CONFIDENCE_LEVEL = 0.95
WOE_BINOMIAL_ALTERNATIVE = "two-sided"
WOE_BINOMIAL_ONE_TAIL_ALTERNATIVE = "greater"
WOE_BINOMIAL_METHOD = "central_exact_binomial"
WOE_BINOMIAL_ONE_TAIL_METHOD = "exact_binomial_upper_tail"
WOE_BINOMIAL_ADJUSTMENT = "bonferroni"
WOE_MAX_BINS = 100
WOE_HHI_LOW_THRESHOLD = 0.15
WOE_HHI_MODERATE_THRESHOLD = 0.25


@dataclass(frozen=True)
class WoeBuildConfig:
    max_bins: int = 6
    min_bin_size: float = 0.05
    monotonic_trend: str = "auto"
    binomial_confidence_level: float = WOE_BINOMIAL_CONFIDENCE_LEVEL
    missing_separate: bool = True
    blank_as_missing: bool = True
    special_values: tuple[str, ...] = ()
    protected_special: bool = True
    engine: str = "optbinning"


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
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def numeric_bin_label(lower: float | None, upper: float | None) -> str:
    if lower is None and upper is None:
        return "all values"
    if lower is None:
        return f"(-inf, {format_bound(upper, 'inf')}]"
    if upper is None:
        return f"({format_bound(lower, '-inf')}, inf)"
    return f"({format_bound(lower, '-inf')}, {format_bound(upper, 'inf')}]"


def bin_display_label(spec: dict[str, Any], bin_spec: dict[str, Any]) -> str:
    if spec.get("feature_kind") == "numeric" and bin_spec.get("kind") == "normal":
        return numeric_bin_label(
            safe_float_or_none(bin_spec.get("lower")),
            safe_float_or_none(bin_spec.get("upper")),
        )
    return str(bin_spec.get("label", "Unmapped"))


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


def package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def version_pair(version_text: str | None) -> tuple[int, int] | None:
    if not version_text:
        return None
    parts = str(version_text).split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def optbinning_compatibility_error() -> str | None:
    optbinning_version = package_version("optbinning")
    sklearn_version = package_version("scikit-learn")
    optbinning_pair = version_pair(optbinning_version)
    sklearn_pair = version_pair(sklearn_version)
    if optbinning_pair is not None and sklearn_pair is not None:
        if optbinning_pair <= (0, 20) and sklearn_pair >= (1, 8):
            return (
                f"optbinning {optbinning_version} is incompatible with scikit-learn {sklearn_version}. "
                "Install scikit-learn>=1.3,<1.8 or use a compatible optbinning/OR-Tools/Python combination."
            )
    return None


@lru_cache(maxsize=1)
def _load_optimal_binning_class() -> Any:
    try:
        from optbinning import OptimalBinning  # type: ignore
    except Exception as exc:
        raise RuntimeError("optbinning is not installed in the active Python environment.") from exc

    compatibility_error = optbinning_compatibility_error()
    if compatibility_error:
        raise RuntimeError(compatibility_error)
    return OptimalBinning


def optbinning_status() -> dict[str, Any]:
    try:
        _load_optimal_binning_class()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "optbinning_version": package_version("optbinning"),
            "sklearn_version": package_version("scikit-learn"),
        }
    return {
        "available": True,
        "error": None,
        "optbinning_version": package_version("optbinning"),
        "sklearn_version": package_version("scikit-learn"),
    }


def optbinning_numeric_splits(
    values: pd.Series,
    y_event: pd.Series,
    feature: str,
    config: WoeBuildConfig,
) -> tuple[list[float] | None, str | None]:
    if config.engine not in {"auto", "optbinning"}:
        return None, None
    try:
        OptimalBinning = _load_optimal_binning_class()
    except Exception as exc:
        message = str(exc)
        raise RuntimeError(message) from exc

    x = pd.to_numeric(values, errors="coerce")
    valid = x.notna() & y_event.notna()
    if valid.sum() < 2 or x[valid].nunique() <= 1:
        return [], None

    monotonic = None if config.monotonic_trend in {"auto", "none"} else config.monotonic_trend
    try:
        optb = OptimalBinning(
            name=str(feature),
            dtype="numerical",
            max_n_bins=max(2, int(config.max_bins)),
            min_bin_size=float(config.min_bin_size),
            monotonic_trend=monotonic,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'force_all_finite' was renamed to 'ensure_all_finite'.*",
                category=FutureWarning,
            )
            optb.fit(x[valid].to_numpy(), y_event[valid].astype(int).to_numpy())
        return sorted(
            {
                float(split)
                for split in getattr(optb, "splits", [])
                if split is not None and math.isfinite(float(split))
            }
        ), None
    except Exception as exc:
        message = f"optbinning failed for {feature}: {type(exc).__name__}: {exc}"
        raise RuntimeError(message) from exc


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


def categorical_bins_from_groups(groups: list[list[str]], start_index: int = 1) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for offset, group in enumerate(groups, start=start_index):
        clean_group = [str(value) for value in group if str(value) != ""]
        if not clean_group:
            continue
        label = ", ".join(clean_group[:4]) + (" ..." if len(clean_group) > 4 else "")
        bins.append(
            {
                "bin_id": make_bin_id(offset),
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
    return bins


def optbinning_categorical_groups(
    values: pd.Series,
    y_event: pd.Series,
    feature: str,
    config: WoeBuildConfig,
) -> tuple[list[list[str]] | None, str | None]:
    if config.engine not in {"auto", "optbinning"}:
        return None, None
    try:
        OptimalBinning = _load_optimal_binning_class()
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    x = values.astype("string")
    valid = x.notna() & y_event.notna()
    valid_values = x[valid]
    if valid.sum() == 0:
        return [], None
    if valid_values.nunique() <= 1:
        return [[str(value) for value in valid_values.dropna().unique().tolist()]], None

    try:
        optb = OptimalBinning(
            name=str(feature),
            dtype="categorical",
            max_n_bins=max(2, int(config.max_bins)),
            min_bin_size=float(config.min_bin_size),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'force_all_finite' was renamed to 'ensure_all_finite'.*",
                category=FutureWarning,
            )
            optb.fit(valid_values.to_numpy(), y_event[valid].astype(int).to_numpy())
        groups: list[list[str]] = []
        for raw_group in getattr(optb, "splits", []) or []:
            if isinstance(raw_group, (str, bytes)):
                group_values = [str(raw_group)]
            else:
                group_values = [str(value) for value in list(raw_group)]
            if group_values:
                groups.append(group_values)
        if not groups:
            groups = [[str(value) for value in valid_values.drop_duplicates().tolist()]]
        return groups, None
    except Exception as exc:
        message = f"optbinning failed for {feature}: {type(exc).__name__}: {exc}"
        raise RuntimeError(message) from exc


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


def count_event_rows(mask: pd.Series, event: pd.Series) -> dict[str, int]:
    count = int(mask.sum())
    event_count = int((mask & event).sum())
    return {
        "count": count,
        "event_count": event_count,
        "non_event_count": count - event_count,
    }


def build_evaluation_profile(
    df: pd.DataFrame,
    target: str,
    feature: str,
    positive_class: Any,
    feature_kind: str,
    missing: pd.Series,
    special: pd.Series,
) -> dict[str, Any]:
    y_valid = df[target].notna()
    event = (df[target] == positive_class) & y_valid
    normal = ~missing & ~special & y_valid
    profile: dict[str, Any] = {
        "schema_version": WOE_SCHEMA_VERSION,
        "dataset_name": "Train",
        "dataframe_id": id(df),
        "rows": int(len(df)),
        "target": str(target),
        "feature": str(feature),
        "positive_class": json_safe_value(positive_class),
        "target_valid_rows": int(y_valid.sum()),
        "total_events": int(event.sum()),
        "total_non_events": int((y_valid & ~event).sum()),
        "missing": count_event_rows(missing & y_valid, event),
        "special": count_event_rows(special & y_valid, event),
    }
    if feature_kind == "categorical":
        value_frame = pd.DataFrame(
            {
                "_value": df.loc[normal, feature].astype("string"),
                "_event": event.loc[normal].astype(int),
            }
        )
        if value_frame.empty:
            profile["categorical_values"] = []
        else:
            grouped = value_frame.groupby("_value", dropna=False)["_event"].agg(["count", "sum"]).reset_index()
            profile["categorical_values"] = [
                {
                    "value": str(row["_value"]),
                    "count": int(row["count"]),
                    "event_count": int(row["sum"]),
                    "non_event_count": int(row["count"] - row["sum"]),
                }
                for row in grouped.to_dict("records")
            ]
    return profile


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
    evaluation_profile = build_evaluation_profile(
        df,
        target,
        feature,
        positive_class,
        feature_kind,
        miss,
        special,
    )
    normal_frame = df.loc[~miss & ~special & df[target].notna(), [feature, target]]

    bins: list[dict[str, Any]] = []
    if config.missing_separate and bool(miss.any()):
        bins.append(missing_bin())
    if config.special_values:
        bins.append(special_bin(list(config.special_values), config.protected_special))

    engine_used = "fallback"
    engine_fallback_reason = None
    if feature_kind == "numeric":
        opt_splits, engine_fallback_reason = optbinning_numeric_splits(
            normal_frame[feature],
            y_event.loc[normal_frame.index],
            feature,
            config,
        )
        if opt_splits is not None:
            splits = opt_splits
            engine_used = "optbinning"
            engine_fallback_reason = None
        else:
            splits = quantile_splits(normal_frame[feature], config.max_bins, config.min_bin_size)
        bins.extend(numeric_bins_from_splits(splits, start_index=1))
    else:
        opt_groups, engine_fallback_reason = optbinning_categorical_groups(
            normal_frame[feature],
            y_event.loc[normal_frame.index],
            feature,
            config,
        )
        if opt_groups is not None:
            engine_used = "optbinning"
            engine_fallback_reason = None
            bins.extend(categorical_bins_from_groups(opt_groups, start_index=1))
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
            "binomial_confidence_level": float(config.binomial_confidence_level),
            "missing_separate": bool(config.missing_separate),
            "blank_as_missing": bool(config.blank_as_missing),
            "special_values": list(config.special_values),
            "protected_special": bool(config.protected_special),
            "engine": str(config.engine),
            "engine_used": engine_used,
            "engine_fallback_reason": engine_fallback_reason,
        },
        "evaluation_profile": evaluation_profile,
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


def spec_binomial_confidence_level(spec: dict[str, Any]) -> float:
    raw_value = spec.get("config", {}).get("binomial_confidence_level", WOE_BINOMIAL_CONFIDENCE_LEVEL)
    try:
        confidence_level = float(raw_value)
    except (TypeError, ValueError):
        return WOE_BINOMIAL_CONFIDENCE_LEVEL
    if not 0.0 < confidence_level < 1.0:
        return WOE_BINOMIAL_CONFIDENCE_LEVEL
    return confidence_level


def central_exact_two_tail_p_value(events: int, trials: int, expected_rate: float) -> float:
    left_tail = float(binom.cdf(int(events), int(trials), float(expected_rate)))
    right_tail = float(binom.sf(int(events) - 1, int(trials), float(expected_rate)))
    return float(min(1.0, 2.0 * min(left_tail, right_tail)))


def exact_upper_tail_p_value(events: int, trials: int, expected_rate: float) -> float:
    return float(binom.sf(int(events) - 1, int(trials), float(expected_rate)))


def append_binomial_hhi_tests(
    table: pd.DataFrame,
    variable_is_probability: bool,
    confidence_level: float = WOE_BINOMIAL_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    out = table.copy()
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("binomial_confidence_level must be between 0 and 1")
    alpha = 1.0 - float(confidence_level)
    counts = pd.to_numeric(out.get("count"), errors="coerce").fillna(0).astype(int)
    events = pd.to_numeric(out.get("event_count"), errors="coerce").fillna(0).astype(int)
    variable_averages = pd.to_numeric(out.get("variable_avg_value"), errors="coerce")
    normal_bins = out.get("kind", pd.Series("", index=out.index)).astype(str).eq("normal")
    applicable = (
        bool(variable_is_probability)
        & normal_bins
        & counts.gt(0)
        & variable_averages.notna()
        & variable_averages.between(0.0, 1.0, inclusive="both")
    )
    family_size = int(applicable.sum())
    effective_alpha = None if family_size == 0 else float(alpha) / family_size
    expected_counts: list[float | None] = []
    count_deltas: list[float | None] = []
    two_tail_p_values: list[float | None] = []
    two_tail_adjusted_p_values: list[float | None] = []
    two_tail_pass_values: list[bool | None] = []
    one_tail_p_values: list[float | None] = []
    one_tail_adjusted_p_values: list[float | None] = []
    one_tail_pass_values: list[bool | None] = []
    ci_lowers: list[float | None] = []
    ci_uppers: list[float | None] = []

    for count, event_count, raw_expected_rate, is_applicable in zip(
        counts.tolist(),
        events.tolist(),
        variable_averages.tolist(),
        applicable.tolist(),
    ):
        if not is_applicable or effective_alpha is None:
            expected_counts.append(None)
            count_deltas.append(None)
            two_tail_p_values.append(None)
            two_tail_adjusted_p_values.append(None)
            two_tail_pass_values.append(None)
            one_tail_p_values.append(None)
            one_tail_adjusted_p_values.append(None)
            one_tail_pass_values.append(None)
            ci_lowers.append(None)
            ci_uppers.append(None)
            continue

        expected_rate = float(raw_expected_rate)
        expected_count = float(count * expected_rate)
        two_tail_p_value = central_exact_two_tail_p_value(event_count, count, expected_rate)
        one_tail_p_value = exact_upper_tail_p_value(event_count, count, expected_rate)
        two_tail_adjusted_p = float(min(1.0, two_tail_p_value * family_size))
        one_tail_adjusted_p = float(min(1.0, one_tail_p_value * family_size))
        interval = binomtest(int(event_count), int(count)).proportion_ci(
            confidence_level=1.0 - effective_alpha,
            method="exact",
        )
        ci_lower = float(interval.low)
        ci_upper = float(interval.high)
        expected_counts.append(expected_count)
        count_deltas.append(float(event_count - expected_count))
        two_tail_p_values.append(two_tail_p_value)
        two_tail_adjusted_p_values.append(two_tail_adjusted_p)
        two_tail_pass_values.append(bool(two_tail_p_value >= effective_alpha))
        one_tail_p_values.append(one_tail_p_value)
        one_tail_adjusted_p_values.append(one_tail_adjusted_p)
        one_tail_pass_values.append(bool(one_tail_p_value >= effective_alpha))
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)

    out["binomial_applicable"] = applicable.astype(bool)
    out["binomial_reference"] = ["variable_avg_value" if value else None for value in applicable]
    out["expected_event_rate"] = variable_averages.where(applicable)
    out["expected_event_count"] = expected_counts
    out["event_count_delta"] = count_deltas
    out["binomial_p_value"] = two_tail_p_values
    out["binomial_adjusted_p_value"] = two_tail_adjusted_p_values
    out["binomial_pass"] = two_tail_pass_values
    out["binomial_significant"] = [None if value is None else not value for value in two_tail_pass_values]
    out["binomial_result"] = [
        "Pass" if value is True else "Reject" if value is False else "N/A"
        for value in two_tail_pass_values
    ]
    out["binomial_two_tail_pass"] = two_tail_pass_values
    out["binomial_two_tail_result"] = out["binomial_result"]
    out["binomial_one_tail_p_value"] = one_tail_p_values
    out["binomial_one_tail_adjusted_p_value"] = one_tail_adjusted_p_values
    out["binomial_one_tail_pass"] = one_tail_pass_values
    out["binomial_one_tail_result"] = [
        "Pass" if value is True else "Reject" if value is False else "N/A"
        for value in one_tail_pass_values
    ]
    out["binomial_ci_lower"] = ci_lowers
    out["binomial_ci_upper"] = ci_uppers
    out["binomial_alpha"] = float(alpha)
    out["binomial_effective_alpha"] = effective_alpha
    bucket_weights = pd.to_numeric(out.get("bucket_weight"), errors="coerce").fillna(0.0)
    out["hhi_contribution"] = bucket_weights**2
    out["bucket_hhi"] = out["hhi_contribution"]
    return out


def hhi_concentration_label(normalized_hhi: float | None) -> str | None:
    if normalized_hhi is None or not math.isfinite(float(normalized_hhi)):
        return None
    if float(normalized_hhi) < WOE_HHI_LOW_THRESHOLD:
        return "low"
    if float(normalized_hhi) < WOE_HHI_MODERATE_THRESHOLD:
        return "moderate"
    return "high"


def bin_quality_metrics(table: pd.DataFrame) -> dict[str, Any]:
    if table.empty:
        return {
            "hhi_total": 0.0,
            "normalized_hhi": None,
            "hhi_concentration": None,
            "hhi_well_distributed": None,
            "max_bin_concentration": None,
            "max_bucket_weight": None,
            "average_bucket_hhi": None,
            "variable_avg_value": None,
            "binomial_family_size": 0,
            "binomial_effective_alpha": None,
            "binomial_pass_bins": 0,
            "binomial_reject_bins": 0,
            "binomial_pass_weight": None,
            "binomial_one_tail_pass_bins": 0,
            "binomial_one_tail_reject_bins": 0,
            "binomial_one_tail_pass_weight": None,
            "binomial_not_applicable_bins": 0,
        }

    weight_column = "bucket_weight" if "bucket_weight" in table.columns else "all_concentration"
    shares = pd.to_numeric(table.get(weight_column), errors="coerce").fillna(0.0)
    positive_bins = int((pd.to_numeric(table.get("count"), errors="coerce").fillna(0) > 0).sum())
    hhi_total = float(np.square(shares).sum())
    if positive_bins > 1:
        min_hhi = 1.0 / positive_bins
        normalized_hhi = float((hhi_total - min_hhi) / (1.0 - min_hhi))
        normalized_hhi = float(max(0.0, min(1.0, normalized_hhi)))
    else:
        normalized_hhi = 0.0 if positive_bins == 1 else None
    concentration = hhi_concentration_label(hhi_total)

    applicable = table.get("binomial_applicable", pd.Series(False, index=table.index)).fillna(False).astype(bool)
    two_tail_pass = table.get("binomial_pass", pd.Series(False, index=table.index)).fillna(False).astype(bool)
    one_tail_pass = (
        table.get("binomial_one_tail_pass", pd.Series(False, index=table.index)).fillna(False).astype(bool)
    )
    populated = pd.to_numeric(table.get("count"), errors="coerce").fillna(0) > 0
    two_tail_reject = applicable & ~two_tail_pass
    one_tail_reject = applicable & ~one_tail_pass
    variable_values = pd.to_numeric(table.get("variable_avg_value"), errors="coerce")
    variable_counts = pd.to_numeric(table.get("count"), errors="coerce").fillna(0.0)
    normal_rows = table.get("kind", pd.Series("", index=table.index)).astype(str).eq("normal")
    variable_value_rows = normal_rows & variable_values.notna() & (variable_counts > 0)
    variable_avg_value = (
        None
        if not bool(variable_value_rows.any())
        else float(
            np.average(
                variable_values.loc[variable_value_rows],
                weights=variable_counts.loc[variable_value_rows],
            )
        )
    )
    return {
        "hhi_total": hhi_total,
        "normalized_hhi": normalized_hhi,
        "hhi_concentration": concentration,
        "hhi_well_distributed": bool(hhi_total < WOE_HHI_MODERATE_THRESHOLD),
        "max_bin_concentration": None if shares.empty else float(shares.max()),
        "max_bucket_weight": None if shares.empty else float(shares.max()),
        "average_bucket_hhi": None if positive_bins == 0 else float(hhi_total / positive_bins),
        "variable_avg_value": variable_avg_value,
        "binomial_reference": "bucket_variable_avg_value",
        "binomial_family_size": int(applicable.sum()),
        "binomial_effective_alpha": (
            None
            if "binomial_effective_alpha" not in table.columns or table["binomial_effective_alpha"].dropna().empty
            else float(table["binomial_effective_alpha"].dropna().iloc[0])
        ),
        "binomial_pass_bins": int((applicable & two_tail_pass).sum()),
        "binomial_reject_bins": int(two_tail_reject.sum()),
        "binomial_pass_weight": float(shares.loc[applicable & two_tail_pass].sum()),
        "binomial_one_tail_pass_bins": int((applicable & one_tail_pass).sum()),
        "binomial_one_tail_reject_bins": int(one_tail_reject.sum()),
        "binomial_one_tail_pass_weight": float(shares.loc[applicable & one_tail_pass].sum()),
        "binomial_not_applicable_bins": int((populated & ~applicable).sum()),
    }


def build_bin_table_from_counts(
    spec: dict[str, Any],
    counts_by_bin: dict[str, dict[str, Any]],
    total_rows: int,
    total_events: int,
    total_non_events: int,
    order: list[str],
    dataset_name: str = "Train",
    variable_is_probability: bool = False,
) -> pd.DataFrame:
    feature = str(spec["feature"])
    spec_by_id = {str(bin_spec.get("bin_id")): bin_spec for bin_spec in spec.get("bins", [])}
    rows: list[dict[str, Any]] = []

    for position, bin_id in enumerate(order, start=1):
        bin_spec = spec_by_id.get(bin_id, {})
        counts = counts_by_bin.get(bin_id, {})
        event_count = int(counts.get("event_count", 0))
        non_event_count = int(counts.get("non_event_count", 0))
        count = event_count + non_event_count
        raw_variable_avg = counts.get("variable_avg_value")
        try:
            variable_avg_value = float(raw_variable_avg)
        except (TypeError, ValueError):
            variable_avg_value = None
        if variable_avg_value is not None and not math.isfinite(variable_avg_value):
            variable_avg_value = None
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
                "label": bin_display_label(spec, bin_spec),
                "lower": bin_spec.get("lower"),
                "upper": bin_spec.get("upper"),
                "values": ", ".join(str(value) for value in bin_spec.get("values", [])),
                "count": count,
                "event_count": event_count,
                "non_event_count": non_event_count,
                "event_rate": None if count == 0 else event_count / count,
                "variable_avg_value": variable_avg_value,
                "bucket_weight": None if total_rows == 0 else count / total_rows,
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
    return append_binomial_hhi_tests(
        pd.DataFrame(rows),
        variable_is_probability=variable_is_probability,
        confidence_level=spec_binomial_confidence_level(spec),
    )


def precomputed_categorical_bin_table(
    df: pd.DataFrame,
    spec: dict[str, Any],
    dataset_name: str,
) -> pd.DataFrame | None:
    if dataset_name != "Train" or spec.get("feature_kind") != "categorical":
        return None
    profile = spec.get("evaluation_profile")
    if not isinstance(profile, dict) or int(profile.get("dataframe_id", -1)) != id(df):
        return None
    records = profile.get("categorical_values")
    if not isinstance(records, list):
        return None

    value_counts: dict[str, dict[str, int]] = {
        str(row.get("value")): {
            "count": int(row.get("count") or 0),
            "event_count": int(row.get("event_count") or 0),
            "non_event_count": int(row.get("non_event_count") or 0),
        }
        for row in records
        if isinstance(row, dict)
    }
    counts_by_bin: dict[str, dict[str, int]] = {}
    covered = {"count": 0, "event_count": 0, "non_event_count": 0}

    def add_counts(bin_id: str, counts: dict[str, int]) -> None:
        target_counts = counts_by_bin.setdefault(bin_id, {"count": 0, "event_count": 0, "non_event_count": 0})
        for key in ("count", "event_count", "non_event_count"):
            value = int(counts.get(key) or 0)
            target_counts[key] += value
            covered[key] += value

    for bin_spec in spec.get("bins", []):
        bin_id = str(bin_spec.get("bin_id"))
        kind = str(bin_spec.get("kind"))
        if kind == "missing":
            add_counts(bin_id, profile.get("missing", {}) if isinstance(profile.get("missing"), dict) else {})
        elif kind == "special":
            add_counts(bin_id, profile.get("special", {}) if isinstance(profile.get("special"), dict) else {})
        elif kind == "normal":
            for value in bin_spec.get("values", []):
                add_counts(bin_id, value_counts.get(str(value), {}))

    total_rows = int(profile.get("target_valid_rows") or 0)
    total_events = int(profile.get("total_events") or 0)
    total_non_events = int(profile.get("total_non_events") or 0)
    unmapped = {
        "count": max(0, total_rows - covered["count"]),
        "event_count": max(0, total_events - covered["event_count"]),
        "non_event_count": max(0, total_non_events - covered["non_event_count"]),
    }
    order = [str(bin_spec.get("bin_id")) for bin_spec in spec.get("bins", [])]
    if unmapped["count"] > 0:
        counts_by_bin["__unmapped__"] = unmapped
        order.append("__unmapped__")
    return build_bin_table_from_counts(
        spec,
        counts_by_bin,
        total_rows,
        total_events,
        total_non_events,
        order,
        dataset_name=dataset_name,
        variable_is_probability=False,
    )


def build_bin_table(
    df: pd.DataFrame,
    target: str,
    spec: dict[str, Any],
    positive_class: Any,
    dataset_name: str = "Train",
) -> pd.DataFrame:
    fast_table = precomputed_categorical_bin_table(df, spec, dataset_name)
    if fast_table is not None:
        return fast_table

    feature = str(spec["feature"])
    assigned = assign_bins(df[feature], spec)
    y_valid = df[target].notna()
    event = ((df[target] == positive_class) & y_valid).astype(int)
    total_events = int(event.sum())
    total_rows = int(y_valid.sum())
    total_non_events = total_rows - total_events
    variable_is_probability = False
    if total_rows:
        grouped_input = pd.DataFrame(
            {
                "bin_id": assigned.loc[y_valid].astype("object"),
                "event": event.loc[y_valid].astype(int),
            }
        )
        named_aggregations: dict[str, tuple[str, str]] = {
            "count": ("event", "size"),
            "event_count": ("event", "sum"),
        }
        if spec.get("feature_kind") == "numeric":
            grouped_input["feature_value"] = pd.to_numeric(df.loc[y_valid, feature], errors="coerce")
            named_aggregations["variable_avg_value"] = ("feature_value", "mean")
            normal_bin_ids = {
                str(bin_spec.get("bin_id"))
                for bin_spec in spec.get("bins", [])
                if bin_spec.get("kind") == "normal"
            }
            normal_values = grouped_input.loc[
                grouped_input["bin_id"].astype(str).isin(normal_bin_ids),
                "feature_value",
            ]
            variable_is_probability = bool(
                normal_values.notna().any()
                and normal_values.dropna().between(0.0, 1.0, inclusive="both").all()
            )
        grouped = grouped_input.groupby("bin_id", dropna=False).agg(**named_aggregations)
        counts_by_bin = {
            str(bin_id): {
                "count": int(row["count"]),
                "event_count": int(row["event_count"]),
                "non_event_count": int(row["count"] - row["event_count"]),
                "variable_avg_value": row.get("variable_avg_value"),
            }
            for bin_id, row in grouped.iterrows()
        }
    else:
        counts_by_bin = {}
    return build_bin_table_from_counts(
        spec,
        counts_by_bin,
        total_rows,
        total_events,
        total_non_events,
        bin_order(spec, assigned),
        dataset_name=dataset_name,
        variable_is_probability=variable_is_probability,
    )


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


def binary_auc_from_bin_table(table: pd.DataFrame, woe_column: str) -> float | None:
    required = {woe_column, "event_count", "non_event_count"}
    if table.empty or not required.issubset(table.columns):
        return None
    score_frame = table[[woe_column, "event_count", "non_event_count"]].copy()
    score_frame[woe_column] = pd.to_numeric(score_frame[woe_column], errors="coerce")
    score_frame["event_count"] = pd.to_numeric(score_frame["event_count"], errors="coerce").fillna(0)
    score_frame["non_event_count"] = pd.to_numeric(score_frame["non_event_count"], errors="coerce").fillna(0)
    score_frame = score_frame.dropna(subset=[woe_column])
    if score_frame.empty:
        return None
    score_frame["score"] = -score_frame[woe_column].astype(float)
    grouped = score_frame.groupby("score", sort=True)[["event_count", "non_event_count"]].sum()
    positives = int(grouped["event_count"].sum())
    negatives = int(grouped["non_event_count"].sum())
    if positives == 0 or negatives == 0:
        return None
    rank = 1.0
    rank_sum = 0.0
    for row in grouped.itertuples(index=False):
        events = int(row.event_count)
        non_events = int(row.non_event_count)
        count = events + non_events
        if count <= 0:
            continue
        rank_sum += events * (rank + (count - 1) / 2.0)
        rank += count
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
    calc_auc = binary_auc_from_bin_table(table, "calculated_woe")
    export_auc = binary_auc_from_bin_table(table, "export_woe")
    monotonicity = monotonicity_from_table(table)
    quality = bin_quality_metrics(table)
    confidence_level = spec_binomial_confidence_level(spec)
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
        "engine_used": str(spec.get("config", {}).get("engine_used", "unknown")),
        "binomial_confidence_level": confidence_level,
        "binomial_family_alpha": 1.0 - confidence_level,
        "binomial_alternative": WOE_BINOMIAL_ALTERNATIVE,
        "binomial_test_method": WOE_BINOMIAL_METHOD,
        "binomial_one_tail_alternative": WOE_BINOMIAL_ONE_TAIL_ALTERNATIVE,
        "binomial_one_tail_test_method": WOE_BINOMIAL_ONE_TAIL_METHOD,
        "binomial_multiple_testing": WOE_BINOMIAL_ADJUSTMENT,
        **quality,
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
    original_semantic = {key: value for key, value in original_spec.items() if key != "evaluation_profile"}
    current_semantic = {key: value for key, value in current_spec.items() if key != "evaluation_profile"}
    same_mapping = original_semantic == current_semantic
    original_train = evaluate_spec(train_df, target, original_spec, positive_class, "Train")
    current_train = copy.deepcopy(original_train) if same_mapping else evaluate_spec(
        train_df,
        target,
        current_spec,
        positive_class,
        "Train",
    )
    result = {
        "original_train": original_train,
        "current_train": current_train,
        "original_test": None,
        "current_test": None,
    }
    if test_df is not None:
        result["original_test"] = evaluate_spec(test_df, target, original_spec, positive_class, "Test")
        result["current_test"] = copy.deepcopy(result["original_test"]) if same_mapping else evaluate_spec(
            test_df,
            target,
            current_spec,
            positive_class,
            "Test",
        )
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


def numeric_values_close(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= 1e-9


def edited_numeric_boundary(
    old_left: dict[str, Any],
    old_right: dict[str, Any],
    edited_left: dict[str, Any],
    edited_right: dict[str, Any],
) -> float | None:
    old_boundary = safe_float_or_none(old_left.get("upper"))
    old_right_lower = safe_float_or_none(old_right.get("lower"))
    if old_boundary is None:
        old_boundary = old_right_lower
    edited_upper = safe_float_or_none(edited_left.get("upper"))
    edited_lower = safe_float_or_none(edited_right.get("lower"))
    upper_changed = not numeric_values_close(edited_upper, old_boundary)
    lower_changed = not numeric_values_close(edited_lower, old_boundary)

    if upper_changed and lower_changed and not numeric_values_close(edited_upper, edited_lower):
        raise ValueError(
            "Adjacent lower/upper boundary mismatch. Edit only one side of the boundary or set both sides equal."
        )
    if lower_changed:
        return edited_lower
    if upper_changed:
        return edited_upper
    return old_boundary


def apply_numeric_range_edits_from_table(spec: dict[str, Any], edited_table: pd.DataFrame) -> dict[str, Any]:
    if spec.get("feature_kind") != "numeric" or "bin_id" not in edited_table.columns:
        return copy_spec(spec)

    old_normal = normal_bins(spec)
    if len(old_normal) <= 1:
        return copy_spec(spec)
    edited_by_id = edited_table.set_index("bin_id").to_dict("index")
    edited_normal = [edited_by_id.get(str(bin_spec.get("bin_id")), {}) for bin_spec in old_normal]

    first_lower = safe_float_or_none(edited_normal[0].get("lower"))
    last_upper = safe_float_or_none(edited_normal[-1].get("upper"))
    if first_lower is not None or last_upper is not None:
        raise ValueError("The first lower and last upper boundary must stay blank to cover the full value range.")

    cutpoints: list[float] = []
    for left_index in range(len(old_normal) - 1):
        boundary = edited_numeric_boundary(
            old_normal[left_index],
            old_normal[left_index + 1],
            edited_normal[left_index],
            edited_normal[left_index + 1],
        )
        if boundary is None:
            raise ValueError("Interior numeric boundaries cannot be blank.")
        cutpoints.append(boundary)
    if any(right <= left for left, right in zip(cutpoints, cutpoints[1:])):
        raise ValueError("Numeric boundaries must be strictly increasing.")

    return apply_numeric_cutpoints(spec, cutpoints)


def apply_bin_table_edits(spec: dict[str, Any], edited_table: pd.DataFrame) -> dict[str, Any]:
    updated = apply_numeric_range_edits_from_table(spec, edited_table)
    return set_assigned_woe_from_table(updated, edited_table)


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


def merge_selected_bins(spec: dict[str, Any], bin_ids: list[str]) -> dict[str, Any]:
    updated = copy_spec(spec)
    prefix = special_prefix_bins(updated)
    normal = normal_bins(updated)
    selected = {str(bin_id) for bin_id in bin_ids}
    selected_indices = [
        idx for idx, bin_spec in enumerate(normal) if str(bin_spec.get("bin_id")) in selected
    ]
    if len(selected_indices) < 2:
        return updated

    if updated.get("feature_kind") == "numeric":
        if max(selected_indices) - min(selected_indices) + 1 != len(selected_indices):
            raise ValueError("Numeric bin merge requires adjacent bins.")
        index = min(selected_indices)
        right_index = max(selected_indices)
        left = copy.deepcopy(normal[index])
        right = normal[right_index]
        left["upper"] = right.get("upper")
        left["label"] = numeric_bin_label(safe_float_or_none(left.get("lower")), safe_float_or_none(left.get("upper")))
        left["assigned_woe"] = None
        merged = normal[:index] + [left] + normal[right_index + 1 :]
    else:
        insert_at = min(selected_indices)
        merged_values: list[Any] = []
        for idx in selected_indices:
            merged_values.extend(normal[idx].get("values", []))
        merged_bin = copy.deepcopy(normal[insert_at])
        merged_bin["values"] = list(dict.fromkeys(str(value) for value in merged_values))
        merged_bin["label"] = ", ".join(str(value) for value in merged_bin["values"][:4]) + (
            " ..." if len(merged_bin["values"]) > 4 else ""
        )
        merged_bin["assigned_woe"] = None
        merged = []
        for idx, bin_spec in enumerate(normal):
            if idx == insert_at:
                merged.append(merged_bin)
            if idx in selected_indices:
                continue
            merged.append(bin_spec)
    updated["bins"] = renumber_normal_bins(prefix, merged)
    return updated


def merge_with_next(spec: dict[str, Any], bin_id: str) -> dict[str, Any]:
    normal = normal_bins(spec)
    index = next((idx for idx, bin_spec in enumerate(normal) if str(bin_spec.get("bin_id")) == str(bin_id)), None)
    if index is None or index >= len(normal) - 1:
        return copy_spec(spec)
    return merge_selected_bins(spec, [str(normal[index].get("bin_id")), str(normal[index + 1].get("bin_id"))])


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
