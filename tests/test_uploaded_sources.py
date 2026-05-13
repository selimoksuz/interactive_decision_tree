from __future__ import annotations

import io

import pandas as pd

from interactive_decision_tree_app import read_uploaded_table


def named_buffer(name: str, data: bytes) -> io.BytesIO:
    buffer = io.BytesIO(data)
    buffer.name = name
    return buffer


def test_read_uploaded_csv():
    uploaded = named_buffer("sample.csv", b"a,target\n1,bad\n2,good\n")

    df = read_uploaded_table(uploaded)

    assert df.to_dict("records") == [
        {"a": 1, "target": "bad"},
        {"a": 2, "target": "good"},
    ]


def test_read_uploaded_xlsx():
    source = pd.DataFrame({"a": [1, 2], "target": ["bad", "good"]})
    output = io.BytesIO()
    source.to_excel(output, index=False)
    output.seek(0)
    output.name = "sample.xlsx"

    df = read_uploaded_table(output)

    pd.testing.assert_frame_equal(df, source)
