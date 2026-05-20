from __future__ import annotations

from uuid import uuid4

import pandas as pd
from streamlit.testing.v1 import AppTest

from interactive_decision_tree.session_store import save_dataframe_session


def test_streamlit_woe_workspace_runs_initial_binning(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERACTIVE_TREE_SESSION_DIR", str(tmp_path))
    df = pd.DataFrame(
        {
            "age": [20, 23, 25, 29, 31, 35, 41, 48, 55, 62] * 6,
            "income": [25_000, 29_000, 31_000, 36_000, 42_000, 47_000, 52_000, 61_000, 72_000, 90_000] * 6,
            "target": [1, 1, 1, 0, 1, 0, 0, 0, 0, 0] * 6,
        }
    )
    data_id, _ = save_dataframe_session(df, target="target", features=["age", "income"])

    at = AppTest.from_file("interactive_decision_tree_app.py")
    at.query_params["data_id"] = data_id
    at.query_params["work_id"] = f"woe_{uuid4().hex}"
    at.run(timeout=25)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]
    assert at.title[0].value == "Data Setup"

    next(button for button in at.button if button.label == "Apply data setup").click()
    at.run(timeout=25)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]

    next(radio for radio in at.radio if radio.label == "Workspace").set_value("WOE Binning")
    at.run(timeout=25)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]
    assert at.title[0].value == "WOE Binning"
    assert all(title.value != "Interactive entropy decision tree" for title in at.title)
    assert any(button.label == "Run initial WOE binning" for button in at.button)

    next(button for button in at.button if button.label == "Run initial WOE binning").click()
    at.run(timeout=30)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]
    projects = at.session_state.filtered_state["_interactive_tree_woe_projects"]
    assert len(projects) == 1
    project = next(iter(projects.values()))
    assert set(project["variables"]) == {"age", "income"}
    status_values = [
        value
        for key, value in at.session_state.filtered_state.items()
        if str(key).startswith("woe_initial_run_status::")
    ]
    assert status_values
    assert status_values[0]["state"] == "done"
    assert all(selectbox.label != "Variable status" for selectbox in at.selectbox)
    assert any(button.label == "Reset to auto mapping" for button in at.button)
    assert any(button.label == "Exclude from export" for button in at.button)
    assert any(button.label == "Merge selected bins" for button in at.button)
    assert any(button.label == "Apply cutpoints" for button in at.button)
    assert not any(button.label == "Approve mapping" for button in at.button)
    assert not any(checkbox.label == "Overwrite existing mappings on rerun" for checkbox in at.checkbox)
