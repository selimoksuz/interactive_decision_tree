from __future__ import annotations

import pandas as pd
import pytest

from interactive_decision_tree_app import TreeImportError, rebuild_editable_tree_from_export


def import_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"segment": "C", "channel": "mobile", "income": 42_000, "risk_flag": "high_risk"},
            {"segment": "C", "channel": "mobile", "income": 50_000, "risk_flag": "low_risk"},
            {"segment": "C", "channel": "web", "income": 40_000, "risk_flag": "low_risk"},
            {"segment": "A", "channel": "mobile", "income": 41_000, "risk_flag": "low_risk"},
        ]
    )


def import_payload() -> dict:
    return {
        "format": "interactive_entropy_decision_tree",
        "target": "risk_flag",
        "features": ["segment", "channel", "income"],
        "tree": {
            "node_id": 0,
            "depth": 0,
            "path": "root",
            "n": 4,
            "is_leaf": False,
            "split": {"feature": "segment", "type": "category_group", "label": "segment {C}"},
            "branches": [
                {
                    "label": "{C}",
                    "condition": {"feature": "segment", "operator": "in", "values": ["C"]},
                    "child": {
                        "node_id": 4,
                        "depth": 1,
                        "path": "root -> segment {C}",
                        "n": 3,
                        "is_leaf": False,
                        "split": {"feature": "channel", "type": "category_eq", "label": "channel == mobile"},
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
                                    "depth": 2,
                                    "path": "root -> segment {C} -> channel == mobile",
                                    "n": 2,
                                    "is_leaf": False,
                                    "split": {
                                        "feature": "income",
                                        "type": "numeric_le",
                                        "label": "income <= 46677.3",
                                    },
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
                                                "depth": 3,
                                                "path": "root -> segment {C} -> income <= 46677.3",
                                                "n": 1,
                                                "is_leaf": True,
                                                "branches": [],
                                            },
                                        },
                                        {
                                            "label": "> 46677.3",
                                            "condition": {
                                                "feature": "income",
                                                "operator": ">",
                                                "threshold": 46677.3,
                                                "includes_missing": True,
                                            },
                                            "child": {
                                                "node_id": 14,
                                                "depth": 3,
                                                "path": "root -> segment {C} -> income > 46677.3",
                                                "n": 1,
                                                "is_leaf": True,
                                                "branches": [],
                                            },
                                        },
                                    ],
                                },
                            },
                            {
                                "label": "!= mobile",
                                "condition": {
                                    "feature": "channel",
                                    "operator": "!=",
                                    "value": "mobile",
                                },
                                "child": {
                                    "node_id": 15,
                                    "depth": 2,
                                    "path": "root -> segment {C} -> channel != mobile",
                                    "n": 1,
                                    "is_leaf": True,
                                    "branches": [],
                                },
                            },
                        ],
                    },
                },
                {
                    "label": "other",
                    "condition": {"feature": "segment", "operator": "not_in", "values": ["C"]},
                    "child": {
                        "node_id": 2,
                        "depth": 1,
                        "path": "root -> segment other",
                        "n": 1,
                        "is_leaf": True,
                        "branches": [],
                    },
                },
            ],
        },
    }


def test_rebuild_editable_tree_from_export_recomputes_paths_and_rows():
    tree, next_node_id, split_history, features = rebuild_editable_tree_from_export(
        import_df(),
        "risk_flag",
        import_payload(),
    )

    assert next_node_id == 16
    assert split_history == [0, 4, 9]
    assert features == ["segment", "channel", "income"]
    assert tree[13]["row_idx"] == [0]
    assert tree[13]["path"] == "root -> segment {C} -> channel == mobile -> income <= 46677.3"
    assert tree[0]["split"]["value"] == ("C",)
    assert tree[4]["split"]["value"] == "mobile"
    assert tree[9]["split"]["value"] == 46677.3


def test_rebuild_editable_tree_from_export_rejects_data_mismatch():
    payload = import_payload()
    payload["tree"]["branches"][0]["child"]["n"] = 99

    with pytest.raises(TreeImportError, match="row mismatch"):
        rebuild_editable_tree_from_export(import_df(), "risk_flag", payload)
