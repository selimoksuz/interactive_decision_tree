from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from interactive_decision_tree.model_tools import (
    feature_alignment,
    kernel_shap_contributions,
    load_model_from_bytes,
    model_capabilities,
    model_class_labels,
    model_feature_names,
    predict_model_scores,
    prepare_model_frame,
    shap_global_importance,
    shap_local_contributions,
)


def raw_pipeline() -> tuple[Pipeline, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "x1": np.linspace(0, 1, 40),
            "x2": np.r_[np.zeros(20), np.ones(20)],
        }
    )
    y = np.where(df["x1"] + df["x2"] > 0.9, "bad", "good")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(random_state=1)),
        ]
    )
    model.fit(df, y)
    return model, df


def tree_payload() -> dict:
    return {
        "format": "interactive_entropy_decision_tree",
        "target": "risk_flag",
        "positive_class": "high_risk",
        "features": ["income"],
        "tree": {
            "node_id": 0,
            "is_leaf": False,
            "target_summary": {
                "prediction": "low_risk",
                "positive_class": "high_risk",
                "default_rate": 0.5,
                "class_distribution": [
                    {"value": "high_risk", "count": 2},
                    {"value": "low_risk", "count": 2},
                ],
            },
            "split": {"label": "income <= 42000"},
            "branches": [
                {
                    "label": "<= 42000",
                    "condition": {"feature": "income", "operator": "<=", "threshold": 42000},
                    "child": {
                        "node_id": 1,
                        "is_leaf": True,
                        "leaf": {"prediction": "high_risk"},
                        "target_summary": {
                            "prediction": "high_risk",
                            "positive_class": "high_risk",
                            "default_rate": 0.8,
                            "class_distribution": [
                                {"value": "high_risk", "count": 8},
                                {"value": "low_risk", "count": 2},
                            ],
                        },
                        "branches": [],
                    },
                },
                {
                    "label": "> 42000",
                    "condition": {"feature": "income", "operator": ">", "threshold": 42000, "includes_missing": True},
                    "child": {
                        "node_id": 2,
                        "is_leaf": True,
                        "leaf": {"prediction": "low_risk"},
                        "target_summary": {
                            "prediction": "low_risk",
                            "positive_class": "high_risk",
                            "default_rate": 0.2,
                            "class_distribution": [
                                {"value": "high_risk", "count": 2},
                                {"value": "low_risk", "count": 8},
                            ],
                        },
                        "branches": [],
                    },
                },
            ],
        },
    }


def test_loaded_raw_pipeline_predicts_with_feature_alignment():
    model, df = raw_pipeline()
    loaded = load_model_from_bytes(pickle.dumps(model))
    feature_names = model_feature_names(loaded)

    assert feature_names == ["x1", "x2"]
    assert model_class_labels(loaded) == ["bad", "good"]
    assert model_capabilities(loaded)["predict_proba"] is True

    prepared = prepare_model_frame(df.head(3), feature_names)
    prediction = predict_model_scores(loaded, prepared, positive_class="bad")
    alignment = feature_alignment(df.assign(extra=1), feature_names)

    assert prediction.output_name == "predict_proba"
    assert prediction.scores.shape == (3,)
    assert set(alignment["status"]) == {"available", "unused_extra"}


def test_interactive_tree_pickle_loads_as_raw_feature_model_adapter():
    model = load_model_from_bytes(pickle.dumps(tree_payload()))
    df = pd.DataFrame({"income": [30_000, 50_000]})

    assert model_feature_names(model) == ["income"]
    assert model_class_labels(model) == ["low_risk", "high_risk"]
    assert model_capabilities(model)["predict_proba"] is True

    prediction = predict_model_scores(model, df, positive_class="high_risk")

    assert prediction.output_name == "predict_proba"
    assert prediction.positive_index == 1
    assert prediction.scores.tolist() == [0.8, 0.2]
    assert model.predict(df).tolist() == ["high_risk", "low_risk"]


def test_kernel_shap_contributions_are_real_shap_values():
    pytest.importorskip("shap")
    model, df = raw_pipeline()
    feature_names = model_feature_names(model)

    result = kernel_shap_contributions(
        model,
        df.head(8),
        df.tail(2),
        feature_names,
        positive_class="bad",
        nsamples=50,
    )

    assert result["values"].shape == (2, 2)
    assert shap_global_importance(result)["feature"].tolist()
    assert shap_local_contributions(result, 0)["feature"].tolist()
