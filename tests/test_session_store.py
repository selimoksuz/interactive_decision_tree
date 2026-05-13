from __future__ import annotations

import pandas as pd
import pytest

from interactive_decision_tree.session_store import (
    load_dataframe_session,
    normalize_data_id,
    save_dataframe_session,
    session_path,
)


def test_dataframe_session_roundtrip_preserves_common_dtypes(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERACTIVE_TREE_SESSION_DIR", str(tmp_path))
    df = pd.DataFrame(
        {
            "num": [1.5, None, 3.2],
            "txt": ["a", None, "c"],
            "cat": pd.Series(["x", "y", "x"], dtype="category"),
            "dt": pd.to_datetime(["2026-01-01", None, "2026-01-03"]),
            "target": [0, 1, 0],
        }
    )

    data_id, metadata = save_dataframe_session(
        df,
        target="target",
        features=["num", "txt", "cat", "dt"],
        name="unit-test",
    )
    restored, restored_metadata = load_dataframe_session(data_id)

    pd.testing.assert_frame_equal(restored, df)
    assert metadata["target"] == "target"
    assert restored_metadata["features"] == ["num", "txt", "cat", "dt"]
    assert restored_metadata["rows"] == 3


def test_data_id_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERACTIVE_TREE_SESSION_DIR", str(tmp_path))

    for bad_id in ["../x", "..\\x", "C:/temp/x", "/tmp/x", "abc/def", "", "x y"]:
        assert normalize_data_id(bad_id) is None
        with pytest.raises(ValueError):
            session_path(bad_id)
