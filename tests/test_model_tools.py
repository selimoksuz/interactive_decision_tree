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
