from __future__ import annotations

import streamlit as st

from interactive_decision_tree_app import graph_label_detail_rows


def test_graph_label_detail_rows_collects_only_shortened_branch_labels():
    st.session_state.clear()
    st.session_state.tree = {
        0: {
            "id": 0,
            "depth": 0,
            "path": "root",
            "row_idx": [0, 1],
            "split": None,
            "children": [
                {"id": 1, "label": "this is a very long branch condition label"},
                {"id": 2, "label": "short"},
            ],
            "left": None,
            "right": None,
        },
        1: {"id": 1, "depth": 1, "path": "root -> long", "row_idx": [0], "split": None, "children": []},
        2: {"id": 2, "depth": 1, "path": "root -> short", "row_idx": [1], "split": None, "children": []},
    }

    rows = graph_label_detail_rows(edge_label_width=12)

    assert len(rows) == 1
    assert rows[0]["item"] == "Node 0 -> Node 1"
    assert rows[0]["visible"].endswith("...")
    assert rows[0]["full"] == "this is a very long branch condition label"
