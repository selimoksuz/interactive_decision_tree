from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

RowInput = dict[str, Any] | pd.Series | pd.DataFrame
TreeNode = dict[str, Any]
TraceStep = dict[str, Any]


def _row_to_mapping(row: RowInput) -> dict[str, Any]:
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


def _threshold_matches(condition: dict[str, Any], value: Any) -> bool:
    if _is_missing(value):
        return bool(condition.get("includes_missing"))
    numeric_value = float(value)
    threshold = float(condition["threshold"])
    return numeric_value <= threshold if condition.get("operator") == "<=" else numeric_value > threshold


def _range_matches(condition: dict[str, Any], value: Any) -> bool:
    if _is_missing(value):
        return bool(condition.get("includes_missing"))
    numeric_value = float(value)
    lower = float(condition["lower"])
    upper = float(condition["upper"])
    return _lower_bound_matches(condition, numeric_value, lower) and _upper_bound_matches(
        condition,
        numeric_value,
        upper,
    )


def _lower_bound_matches(condition: dict[str, Any], numeric_value: float, lower: float) -> bool:
    return numeric_value >= lower if condition.get("lower_inclusive") else numeric_value > lower


def _upper_bound_matches(condition: dict[str, Any], numeric_value: float, upper: float) -> bool:
    return numeric_value <= upper if condition.get("upper_inclusive") else numeric_value < upper


def _equals_matches(condition: dict[str, Any], value: Any) -> bool:
    return _same_value(value, condition.get("value"))


def _not_equals_matches(condition: dict[str, Any], value: Any) -> bool:
    return not _equals_matches(condition, value)


def _in_matches(condition: dict[str, Any], value: Any) -> bool:
    return any(_same_value(value, item) for item in condition.get("values", []))


def _not_in_matches(condition: dict[str, Any], value: Any) -> bool:
    return not _in_matches(condition, value)


def _is_missing_matches(condition: dict[str, Any], value: Any) -> bool:
    return _is_missing(value)


_CONDITION_HANDLERS: dict[str, Callable[[dict[str, Any], Any], bool]] = {
    "<=": _threshold_matches,
    ">": _threshold_matches,
    "range": _range_matches,
    "==": _equals_matches,
    "!=": _not_equals_matches,
    "in": _in_matches,
    "not_in": _not_in_matches,
    "is_missing": _is_missing_matches,
}


def condition_matches(condition: dict[str, Any], row: dict[str, Any]) -> bool:
    operator = condition.get("operator")
    handler = _CONDITION_HANDLERS.get(str(operator))
    if handler is None:
        return False
    return handler(condition, row.get(condition.get("feature")))


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


def _trace_step_path(trace_step: dict[str, Any]) -> str:
    condition = trace_step.get("condition") or {}
    feature = condition.get("feature")
    branch_label = trace_step.get("branch_label")
    if feature is not None and branch_label is not None:
        return f"{feature} {branch_label}"
    if branch_label is not None:
        return str(branch_label)
    split_label = trace_step.get("split")
    return str(split_label) if split_label is not None else ""


def _leaf_path_from_trace(trace: list[dict[str, Any]], fallback_path: Any) -> str | None:
    if not trace:
        return str(fallback_path) if fallback_path is not None else None
    parts = ["root"]
    for trace_step in trace:
        step_path = _trace_step_path(trace_step)
        if step_path:
            parts.append(step_path)
    return " -> ".join(parts)


def _tree_root(payload: dict[str, Any]) -> TreeNode:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary.")
    tree = payload.get("tree")
    if not isinstance(tree, dict):
        raise ValueError("payload must contain a nested `tree` export.")
    return tree


def _node_branches(node: TreeNode) -> list[dict[str, Any]]:
    branches = node.get("branches")
    return branches if isinstance(branches, list) else []


def _is_leaf_node(node: TreeNode) -> bool:
    return bool(node.get("is_leaf")) or not _node_branches(node)


def _matching_branch(node: TreeNode, row_values: dict[str, Any]) -> dict[str, Any] | None:
    for branch in _node_branches(node):
        condition = branch.get("condition", {})
        if condition_matches(condition, row_values):
            return branch
    return None


def _trace_step(node: TreeNode, branch: dict[str, Any]) -> TraceStep:
    return {
        "node_id": node.get("node_id"),
        "split": (node.get("split") or {}).get("label"),
        "branch_label": branch.get("label"),
        "condition": branch.get("condition", {}),
    }


def _branch_child(branch: dict[str, Any]) -> TreeNode:
    child = branch.get("child")
    if not isinstance(child, dict):
        raise ValueError("Matching branch does not contain a valid child node.")
    return child


def _walk_tree(root: TreeNode, row_values: dict[str, Any]) -> tuple[TreeNode, list[TraceStep]]:
    node = root
    trace: list[TraceStep] = []
    while not _is_leaf_node(node):
        branch = _matching_branch(node, row_values)
        if branch is None:
            raise ValueError(f"No matching branch at node {node.get('node_id')} for row values: {row_values}")
        trace.append(_trace_step(node, branch))
        node = _branch_child(branch)
    return node, trace


def _target_summary(node: TreeNode) -> dict[str, Any]:
    target_summary = node.get("target_summary", {})
    return target_summary if isinstance(target_summary, dict) else {}


def _leaf_prediction(node: TreeNode, target_summary: dict[str, Any]) -> Any:
    leaf = node.get("leaf") or {}
    return leaf.get("prediction", target_summary.get("prediction"))


def _positive_class(payload: dict[str, Any], target_summary: dict[str, Any]) -> Any:
    return target_summary.get("positive_class", payload.get("positive_class"))


def _positive_probability(
    positive_class: Any,
    target_summary: dict[str, Any],
    probabilities: dict[str, float],
) -> float | None:
    if target_summary.get("default_rate") is not None:
        return float(target_summary["default_rate"])
    return probabilities.get(str(positive_class))


def _leaf_score_result(
    payload: dict[str, Any],
    node: TreeNode,
    trace: list[TraceStep],
) -> dict[str, Any]:
    target_summary = _target_summary(node)
    prediction = _leaf_prediction(node, target_summary)
    probabilities = _class_probabilities(target_summary)
    positive_class = _positive_class(payload, target_summary)
    return {
        "prediction": prediction,
        "prediction_probability": _prediction_probability(prediction, target_summary, probabilities),
        "class_probabilities": probabilities,
        "positive_class": positive_class,
        "positive_class_probability": _positive_probability(positive_class, target_summary, probabilities),
        "leaf_node_id": node.get("node_id"),
        "leaf_path": _leaf_path_from_trace(trace, node.get("path")),
        "exported_leaf_path": node.get("path"),
        "target_summary": target_summary,
        "trace": trace,
    }


def _score_with_root(
    payload: dict[str, Any],
    root: TreeNode,
    row: RowInput,
) -> dict[str, Any]:
    leaf, trace = _walk_tree(root, _row_to_mapping(row))
    return _leaf_score_result(payload, leaf, trace)


def compile_tree_scorer(payload: dict[str, Any]) -> Callable[[RowInput], dict[str, Any]]:
    root = _tree_root(payload)

    def score(row: RowInput) -> dict[str, Any]:
        return _score_with_root(payload, root, row)

    return score


def score_tree_payload(payload: dict[str, Any], row: RowInput) -> dict[str, Any]:
    return _score_with_root(payload, _tree_root(payload), row)
