from __future__ import annotations

import json

import pytest

from interactive_decision_tree import load_tree_json, load_tree_pickle, save_tree_pickle


def test_tree_pickle_roundtrip(tmp_path):
    payload = {
        "format": "interactive_entropy_decision_tree",
        "target": "risk_flag",
        "tree": {"node_type": "leaf", "prediction": "low_risk"},
    }
    path = save_tree_pickle(payload, tmp_path / "tree.pkl")

    assert load_tree_pickle(path) == payload


def test_tree_pickle_requires_dict(tmp_path):
    path = tmp_path / "bad.pkl"
    save_tree_pickle({"ok": True}, path)
    path.write_bytes(__import__("pickle").dumps(["not", "dict"]))

    with pytest.raises(ValueError):
        load_tree_pickle(path)


def test_tree_json_loader(tmp_path):
    path = tmp_path / "tree.json"
    payload = {"format": "interactive_entropy_decision_tree", "tree": {"id": 0}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_tree_json(path) == payload
