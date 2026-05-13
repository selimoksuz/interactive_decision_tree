from __future__ import annotations

from uuid import uuid4

import pandas as pd
from streamlit.testing.v1 import AppTest

from interactive_decision_tree.session_store import save_dataframe_session


def test_streamlit_loads_session_data_and_restores_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERACTIVE_TREE_SESSION_DIR", str(tmp_path))
    df = pd.DataFrame(
        {
            "age": list(range(20, 80)),
            "income": [30_000 + i * 1_000 for i in range(60)],
            "target": ["bad" if i < 30 else "good" for i in range(60)],
        }
    )
    data_id, _ = save_dataframe_session(df, target="target", features=["age", "income"])
    work_id = f"test_{uuid4().hex}"

    at = AppTest.from_file("interactive_decision_tree_app.py")
    at.query_params["data_id"] = data_id
    at.query_params["work_id"] = work_id
    at.run(timeout=25)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]
    assert at.session_state.filtered_state["state_key"][0].startswith(f"session:{data_id}:")

    next(button for button in at.button if button.label == "Build optimal tree").click()
    at.run(timeout=35)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]
    split_count = sum(
        1 for node in at.session_state.filtered_state["tree"].values() if node["split"] is not None
    )
    assert split_count > 0

    at2 = AppTest.from_file("interactive_decision_tree_app.py")
    at2.query_params["data_id"] = data_id
    at2.query_params["work_id"] = work_id
    at2.run(timeout=25)
    assert len(at2.exception) == 0, [exc.value for exc in at2.exception]
    restored_split_count = sum(
        1 for node in at2.session_state.filtered_state["tree"].values() if node["split"] is not None
    )
    assert restored_split_count == split_count
