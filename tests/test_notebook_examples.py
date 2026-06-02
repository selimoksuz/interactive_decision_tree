from __future__ import annotations

import json
from pathlib import Path


def test_notebook_synthetic_sample_keeps_customer_id_out_of_model_features():
    notebook = json.loads(Path("examples/notebook_dataframe_sql_demo.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "customer_id = np.arange" in source
    assert '"customer_id": customer_id' in source
    assert 'column not in {"risk_flag", "customer_id"}' in source
