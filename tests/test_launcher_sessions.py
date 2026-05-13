from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from interactive_decision_tree import launch_tree


def test_launch_tree_writes_to_explicit_session_dir(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})

    url = launch_tree(
        df,
        target="target",
        start_server=False,
        open_browser=False,
        session_dir=tmp_path,
    )

    data_id = parse_qs(urlparse(url).query)["data_id"][0]
    session_path = Path(tmp_path) / data_id
    assert (session_path / "data.pkl").exists()
    assert (session_path / "metadata.json").exists()
