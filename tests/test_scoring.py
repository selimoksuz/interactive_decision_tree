from __future__ import annotations

import pandas as pd
import pytest

from interactive_decision_tree import compile_tree_scorer, condition_matches, score_tree_payload


def sample_payload():
    return {
        "format": "interactive_entropy_decision_tree",
        "target": "risk_flag",
        "tree": {
            "node_id": 0,
            "is_leaf": False,
            "path": "root",
            "target_summary": {
                "prediction": "low_risk",
                "positive_class": "high_risk",
                "default_rate": 0.5,
                "class_distribution": [
                    {"value": "high_risk", "count": 2},
                    {"value": "low_risk", "count": 2},
                ],
            },
            "split": {"label": "income <= 42000"},
            "branches": [
                {
                    "label": "<= 42000",
                    "condition": {"feature": "income", "operator": "<=", "threshold": 42000},
                    "child": {
                        "node_id": 1,
                        "is_leaf": True,
                        "path": "root -> income <= 42000",
                        "leaf": {"prediction": "high_risk"},
                        "target_summary": {
                            "prediction": "high_risk",
                            "positive_class": "high_risk",
                            "default_rate": 0.8,
                            "class_distribution": [
                                {"value": "high_risk", "count": 8},
                                {"value": "low_risk", "count": 2},
                            ],
                        },
                        "branches": [],
                    },
                },
                {
                    "label": "> 42000",
                    "condition": {
                        "feature": "income",
                        "operator": ">",
                        "threshold": 42000,
                        "includes_missing": True,
                    },
                    "child": {
                        "node_id": 2,
                        "is_leaf": True,
                        "path": "root -> income > 42000",
                        "leaf": {"prediction": "low_risk"},
                        "target_summary": {
                            "prediction": "low_risk",
                            "positive_class": "high_risk",
                            "default_rate": 0.2,
                            "class_distribution": [
                                {"value": "high_risk", "count": 2},
                                {"value": "low_risk", "count": 8},
                            ],
                        },
                        "branches": [],
                    },
                },
            ],
        },
    }


def test_score_tree_payload_dict_row():
    result = score_tree_payload(sample_payload(), {"income": 30_000})

    assert result["prediction"] == "high_risk"
    assert result["prediction_probability"] == 0.8
    assert result["positive_class"] == "high_risk"
    assert result["positive_class_probability"] == 0.8
    assert result["class_probabilities"] == {"high_risk": 0.8, "low_risk": 0.2}
    assert result["leaf_node_id"] == 1
    assert result["trace"][0]["branch_label"] == "<= 42000"


def test_score_tree_payload_dataframe_row():
    row = pd.DataFrame([{"income": 50_000}])

    result = score_tree_payload(sample_payload(), row)

    assert result["prediction"] == "low_risk"
    assert result["prediction_probability"] == 0.8
    assert result["positive_class_probability"] == 0.2
    assert result["leaf_node_id"] == 2


def test_compile_tree_scorer_matches_public_scoring():
    scorer = compile_tree_scorer(sample_payload())

    compiled_result = scorer({"income": 30_000})
    direct_result = score_tree_payload(sample_payload(), {"income": 30_000})

    assert compiled_result == direct_result


@pytest.mark.parametrize(
    ("condition", "row", "expected"),
    [
        ({"feature": "x", "operator": "<=", "threshold": 10}, {"x": 9}, True),
        ({"feature": "x", "operator": ">", "threshold": 10}, {"x": 9}, False),
        (
            {
                "feature": "x",
                "operator": "range",
                "lower": 5,
                "upper": 10,
                "lower_inclusive": True,
                "upper_inclusive": False,
            },
            {"x": 5},
            True,
        ),
        ({"feature": "x", "operator": "==", "value": "A"}, {"x": "A"}, True),
        ({"feature": "x", "operator": "!=", "value": "A"}, {"x": "B"}, True),
        ({"feature": "x", "operator": "in", "values": ["A", "B"]}, {"x": "B"}, True),
        ({"feature": "x", "operator": "not_in", "values": ["A", "B"]}, {"x": "C"}, True),
        ({"feature": "x", "operator": ">", "threshold": 10, "includes_missing": True}, {"x": None}, True),
        ({"feature": "x", "operator": "is_missing"}, {"x": None}, True),
        ({"feature": "x", "operator": "is_missing"}, {"x": 7}, False),
    ],
)
def test_condition_matches_operator_helpers(condition, row, expected):
    assert condition_matches(condition, row) is expected


def test_score_tree_payload_rebuilds_leaf_path_from_trace_when_exported_path_is_stale():
    payload = {
        "format": "interactive_entropy_decision_tree",
        "target": "risk_flag",
        "tree": {
            "node_id": 0,
            "is_leaf": False,
            "path": "root",
            "target_summary": {"prediction": 0, "class_distribution": []},
            "split": {"label": "segment target-profile groups (2)"},
            "branches": [
                {
                    "label": "{C}",
                    "condition": {"feature": "segment", "operator": "in", "values": ["C"]},
                    "child": {
                        "node_id": 4,
                        "is_leaf": False,
                        "path": "root -> segment {C}",
                        "target_summary": {"prediction": 0, "class_distribution": []},
                        "split": {"label": "channel == mobile"},
                        "branches": [
                            {
                                "label": "== mobile",
                                "condition": {
                                    "feature": "channel",
                                    "operator": "==",
                                    "value": "mobile",
                                },
                                "child": {
                                    "node_id": 9,
                                    "is_leaf": False,
                                    "path": "root -> segment {C} -> channel == mobile",
                                    "target_summary": {"prediction": 0, "class_distribution": []},
                                    "split": {"label": "income <= 46677.3"},
                                    "branches": [
                                        {
                                            "label": "<= 46677.3",
                                            "condition": {
                                                "feature": "income",
                                                "operator": "<=",
                                                "threshold": 46677.3,
                                            },
                                            "child": {
                                                "node_id": 13,
                                                "is_leaf": True,
                                                "path": "root -> segment {C} -> income <= 46677.3",
                                                "leaf": {"prediction": 1},
                                                "target_summary": {
                                                    "prediction": 1,
                                                    "positive_class": 1,
                                                    "default_rate": 0.7,
                                                    "class_distribution": [
                                                        {"value": 1, "count": 7},
                                                        {"value": 0, "count": 3},
                                                    ],
                                                },
                                                "branches": [],
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                }
            ],
        },
    }

    result = score_tree_payload(
        payload,
        {"segment": "C", "channel": "mobile", "income": 42_000},
    )

    assert result["leaf_node_id"] == 13
    assert (
        result["leaf_path"]
        == "root -> segment {C} -> channel == mobile -> income <= 46677.3"
    )
    assert result["exported_leaf_path"] == "root -> segment {C} -> income <= 46677.3"


def test_score_tree_payload_uses_node_summary_when_category_has_no_branch():
    payload = {
        "format": "interactive_entropy_decision_tree",
        "target": "risk_flag",
        "positive_class": "high_risk",
        "tree": {
            "node_id": 0,
            "is_leaf": False,
            "path": "root",
            "target_summary": {
                "prediction": "low_risk",
                "positive_class": "high_risk",
                "default_rate": 0.4,
                "class_distribution": [
                    {"value": "high_risk", "count": 40},
                    {"value": "low_risk", "count": 60},
                ],
            },
            "split": {"label": "collection_status groups"},
            "branches": [
                {
                    "label": "{monitor}",
                    "condition": {"feature": "collection_status", "operator": "in", "values": ["monitor"]},
                    "child": {
                        "node_id": 2,
                        "is_leaf": False,
                        "path": "root -> collection_status {monitor}",
                        "target_summary": {
                            "prediction": "high_risk",
                            "positive_class": "high_risk",
                            "default_rate": 0.65,
                            "class_distribution": [
                                {"value": "high_risk", "count": 65},
                                {"value": "low_risk", "count": 35},
                            ],
                        },
                        "split": {"label": "segment target-profile groups"},
                        "branches": [
                            {
                                "label": "{B}",
                                "condition": {"feature": "segment", "operator": "in", "values": ["B"]},
                                "child": {
                                    "node_id": 5,
                                    "is_leaf": True,
                                    "path": "root -> collection_status {monitor} -> segment {B}",
                                    "leaf": {"prediction": "low_risk"},
                                    "target_summary": {
                                        "prediction": "low_risk",
                                        "positive_class": "high_risk",
                                        "default_rate": 0.3,
                                        "class_distribution": [
                                            {"value": "high_risk", "count": 30},
                                            {"value": "low_risk", "count": 70},
                                        ],
                                    },
                                    "branches": [],
                                },
                            }
                        ],
                    },
                }
            ],
        },
    }

    result = score_tree_payload(payload, {"collection_status": "monitor", "segment": "D"})

    assert result["fallback_reason"] == "no_matching_branch"
    assert result["leaf_node_id"] == 2
    assert result["prediction"] == "high_risk"
    assert result["positive_class_probability"] == 0.65
    assert result["leaf_path"] == "root -> collection_status {monitor}"
