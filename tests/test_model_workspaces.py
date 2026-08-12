from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
from streamlit.testing.v1 import AppTest

from interactive_decision_tree.session_store import save_dataframe_session


APP_PATH = Path(__file__).resolve().parents[1] / "interactive_decision_tree_app.py"


def test_model_shap_and_what_if_workspaces_render_without_tree_side_effects(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERACTIVE_TREE_SESSION_DIR", str(tmp_path))
    df = pd.DataFrame(
        {
            "age": [21, 35, 47, 62],
            "income": [30_000, 45_000, 62_000, 81_000],
            "target": ["bad", "bad", "good", "good"],
        }
    )
    data_id, _ = save_dataframe_session(df, target="target", features=["age", "income"])
    work_id = f"test_{uuid4().hex}"

    at = AppTest.from_file(APP_PATH)
    at.query_params["data_id"] = data_id
    at.query_params["work_id"] = work_id
    at.run(timeout=25)
    assert len(at.exception) == 0, [exc.value for exc in at.exception]

    workspace = next(radio for radio in at.radio if radio.label == "Workspace")
    assert {"Model Setup", "SHAP Analysis", "What-if Simulator"}.issubset(set(workspace.options))

    for workspace_name in ["Model Setup", "SHAP Analysis", "What-if Simulator"]:
        next(radio for radio in at.radio if radio.label == "Workspace").set_value(workspace_name)
        at.run(timeout=25)
        assert len(at.exception) == 0, [exc.value for exc in at.exception]
        assert at.session_state.filtered_state.get("state_key") is None
