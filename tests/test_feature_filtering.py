from __future__ import annotations

from interactive_decision_tree_app import (
    MIN_INFORMATION_GAIN_EPSILON,
    SplitCandidate,
    feature_summary_rows,
    ordered_features_by_gain,
)


def candidate(feature: str, gain: float) -> SplitCandidate:
    return SplitCandidate(
        feature=feature,
        split_type="numeric_le",
        value=1.0,
        parent_entropy=1.0,
        weighted_entropy=1.0 - gain,
        information_gain=gain,
        branch_count=2,
        branch_labels=("<=", ">"),
        branch_ns=(10, 10),
        branch_entropies=(0.5, 0.5),
        label=f"{feature} <= 1",
    )


def test_zero_gain_features_are_excluded_from_selection_options():
    rows = feature_summary_rows(
        [candidate("zero_gain", 0.0), candidate("positive_gain", 0.1)],
        ["zero_gain", "positive_gain", "missing"],
    )
    stats = {row["variable"]: row for row in rows}

    assert "zero_gain" not in stats
    assert "missing" not in stats
    assert ordered_features_by_gain(["zero_gain", "positive_gain", "missing"], stats) == [
        "positive_gain"
    ]


def test_zero_gain_rows_can_still_be_shown_in_diagnostic_table():
    rows = feature_summary_rows(
        [candidate("zero_gain", 0.0)],
        ["zero_gain", "missing"],
        include_zero_gain=True,
    )

    assert [row["variable"] for row in rows] == ["zero_gain", "missing"]
    assert rows[0]["best_information_gain"] <= MIN_INFORMATION_GAIN_EPSILON
