from __future__ import annotations

from typing import Any

import pandas as pd


def _row_to_mapping(row: dict[str, Any] | pd.Series | pd.DataFrame) -> dict[str, Any]:
    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            raise ValueError("DataFrame scoring input must contain exactly one row.")
        return row.iloc[0].to_dict()
    if isinstance(row, pd.Series):
        return row.to_dict()
    if isinstance(row, dict):
        return row
    raise TypeError("row must be a dict, pandas Series, or single-row DataFrame.")


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _category_value(value: Any) -> Any:
    return "__MISSING__" if _is_missing(value) else value


def _same_value(left: Any, right: Any) -> bool:
    left = _category_value(left)
    right = _category_value(right)
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def condition_matches(condition: dict[str, Any], row: dict[str, Any]) -> bool:
    operator = condition.get("operator")
    feature = condition.get("feature")
    value = row.get(feature)

    if operator in ("<=", ">"):
        if _is_missing(value):
            return bool(condition.get("includes_missing"))
        numeric_value = float(value)
        threshold = float(condition["threshold"])
        return numeric_value <= threshold if operator == "<=" else numeric_value > threshold

    if operator == "range":
        if _is_missing(value):
            return bool(condition.get("includes_missing"))
        numeric_value = float(value)
        lower = float(condition["lower"])
        upper = float(condition["upper"])
        lower_ok = numeric_value >= lower if condition.get("lower_inclusive") else numeric_value > lower
        upper_ok = numeric_value <= upper if condition.get("upper_inclusive") else numeric_value < upper
        return lower_ok and upper_ok

    if operator == "==":
        return _same_value(value, condition.get("value"))

    if operator == "!=":
        return not _same_value(value, condition.get("value"))

    if operator == "in":
        return any(_same_value(value, item) for item in condition.get("values", []))

    if operator == "not_in":
        return not any(_same_value(value, item) for item in condition.get("values", []))

    return False


def _class_probabilities(target_summary: dict[str, Any]) -> dict[str, float]:
    distribution = target_summary.get("class_distribution")
    if not isinstance(distribution, list):
        return {}

    counts: dict[str, float] = {}
    total = 0.0
    for item in distribution:
        if not isinstance(item, dict):
            continue
        label = str(item.get("value"))
        count = float(item.get("count", 0) or 0)
        counts[label] = counts.get(label, 0.0) + count
        total += count

    if total <= 0:
        return {}
    return {label: count / total for label, count in counts.items()}


def _prediction_probability(
    prediction: Any,
    target_summary: dict[str, Any],
    probabilities: dict[str, float],
) -> float | None:
    prediction_key = str(prediction)
    if prediction_key in probabilities:
        return probabilities[prediction_key]

    positive_class = target_summary.get("positive_class")
    default_rate = target_summary.get("default_rate")
    if positive_class is not None and default_rate is not None:
        positive_probability = float(default_rate)
        if _same_value(prediction, positive_class):
            return positive_probability
        return 1.0 - positive_probability
    return None


def score_tree_payload(
    payload: dict[str, Any],
    row: dict[str, Any] | pd.Series | pd.DataFrame,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary.")
    if "tree" not in payload:
        raise ValueError("payload must contain a nested `tree` export.")

    row_values = _row_to_mapping(row)
    node = payload["tree"]
    trace: list[dict[str, Any]] = []

    while True:
        if node.get("is_leaf") or not node.get("branches"):
            leaf = node.get("leaf") or {}
            target_summary = node.get("target_summary", {})
            prediction = leaf.get("prediction", target_summary.get("prediction"))
            probabilities = _class_probabilities(target_summary)
            positive_class = target_summary.get("positive_class", payload.get("positive_class"))
            positive_probability = (
                float(target_summary["default_rate"])
                if target_summary.get("default_rate") is not None
                else probabilities.get(str(positive_class))
            )
            return {
                "prediction": prediction,
                "prediction_probability": _prediction_probability(
                    prediction,
                    target_summary,
                    probabilities,
                ),
                "class_probabilities": probabilities,
                "positive_class": positive_class,
                "positive_class_probability": positive_probability,
                "leaf_node_id": node.get("node_id"),
                "leaf_path": node.get("path"),
                "target_summary": target_summary,
                "trace": trace,
            }

        for branch in node.get("branches", []):
            condition = branch.get("condition", {})
            if condition_matches(condition, row_values):
                trace.append(
                    {
                        "node_id": node.get("node_id"),
                        "split": node.get("split", {}).get("label"),
                        "branch_label": branch.get("label"),
                        "condition": condition,
                    }
                )
                node = branch["child"]
                break
        else:
            raise ValueError(
                f"No matching branch at node {node.get('node_id')} for row values: {row_values}"
            )
