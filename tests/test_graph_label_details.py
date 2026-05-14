from __future__ import annotations

import streamlit as st

from interactive_decision_tree_app import selected_node_branch_detail_rows


def test_selected_node_branch_detail_rows_follow_selected_node_children():
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

    rows = selected_node_branch_detail_rows(node_id=0, edge_label_width=12)

    assert len(rows) == 2
    assert rows[0]["branch"] == "Node 0 -> Node 1"
    assert rows[0]["visible"].endswith("...")
    assert rows[0]["full"] == "this is a very long branch condition label"
    assert rows[0]["child_path"] == "root -> long"
    assert rows[1]["branch"] == "Node 0 -> Node 2"
    assert rows[1]["full"] == "short"
