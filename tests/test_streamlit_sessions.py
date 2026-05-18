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
    assert not any("Missing Submit Button" in str(warning.value) for warning in at.warning)
    applied_context_key = "_interactive_tree_applied_data_context"
    assert at.session_state.filtered_state[applied_context_key]["features"] == ["age", "income"]
    assert at.session_state.filtered_state.get("state_key") is None
    assert not any(str(key).startswith("candidate_cache::") for key in at.session_state.filtered_state)

    assert any(str(dataframe.key).startswith("feature_manager::") for dataframe in at.dataframe)
    assert any(number_input.label == "Split ranking variable limit" for number_input in at.number_input)

    next(checkbox for checkbox in at.checkbox if checkbox.label == "Use sampled working data").set_value(True)
    next(number_input for number_input in at.number_input if number_input.label == "Sample rows").set_value(55)
    at.run(timeout=25)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]
    assert at.session_state.filtered_state.get("state_key") is None

    next(radio for radio in at.radio if radio.label == "Test / validation source").set_value("Separate data source")
    at.run(timeout=25)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]
    assert any(radio.label == "Test data source" for radio in at.radio)
    assert not any(slider.label == "Test share" for slider in at.slider)
    assert any(button.label == "Apply data setup" for button in at.button)
    assert at.session_state.filtered_state.get("state_key") is None

    next(radio for radio in at.radio if radio.label == "Test / validation source").set_value("Split train data")
    at.run(timeout=25)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]
    assert any(slider.label == "Test share" for slider in at.slider)
    assert not any(radio.label == "Test data source" for radio in at.radio)
    assert at.session_state.filtered_state.get("state_key") is None

    next(button for button in at.button if button.label == "Apply data setup").click()
    at.run(timeout=25)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]
    assert ":sample:" in at.session_state.filtered_state[applied_context_key]["data_key"]
    assert ":train_split:" in at.session_state.filtered_state[applied_context_key]["data_key"]
    assert at.session_state.filtered_state[applied_context_key]["features"] == ["age", "income"]
    assert at.session_state.filtered_state[applied_context_key]["split_variable_limit"] == 2
    assert at.session_state.filtered_state.get("state_key") is None

    next(radio for radio in at.radio if radio.label == "Workspace").set_value("Tree Builder")
    at.run(timeout=25)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]
    assert at.session_state.filtered_state["state_key"][0].startswith(f"session:{data_id}:")
    assert ":sample:" in at.session_state.filtered_state["state_key"][0]
    assert ":train_split:" in at.session_state.filtered_state["state_key"][0]
    assert len(at.session_state.filtered_state["tree"][0]["row_idx"]) < len(df)

    next(button for button in at.button if button.label == "Build from root").click()
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
