from __future__ import annotations

import pandas as pd

from interactive_decision_tree import score_tree_payload


def sample_payload():
    return {
        "format": "interactive_entropy_decision_tree",
        "target": "risk_flag",
        "tree": {
            "node_id": 0,
            "is_leaf": False,
            "path": "root",
            "target_summary": {"prediction": "low_risk"},
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
                        "target_summary": {"prediction": "high_risk", "default_rate": 0.8},
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
                        "target_summary": {"prediction": "low_risk", "default_rate": 0.2},
                        "branches": [],
                    },
                },
            ],
        },
    }


def test_score_tree_payload_dict_row():
    result = score_tree_payload(sample_payload(), {"income": 30_000})

    assert result["prediction"] == "high_risk"
    assert result["leaf_node_id"] == 1
    assert result["trace"][0]["branch_label"] == "<= 42000"


def test_score_tree_payload_dataframe_row():
    row = pd.DataFrame([{"income": 50_000}])

    result = score_tree_payload(sample_payload(), row)

    assert result["prediction"] == "low_risk"
    assert result["leaf_node_id"] == 2
