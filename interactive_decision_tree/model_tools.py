from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelPrediction:
    scores: np.ndarray
    output_name: str
    positive_index: int | None


def load_model_from_bytes(payload: bytes) -> Any:
    try:
        import joblib

        return joblib.load(io.BytesIO(payload))
    except Exception:
        return pickle.loads(payload)


def model_feature_names(model: Any) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "named_steps"):
        for step in getattr(model, "named_steps", {}).values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                break
    if names is None:
        return []
    return [str(name) for name in list(names)]


def model_class_labels(model: Any) -> list[Any]:
    labels = getattr(model, "classes_", None)
    if labels is None and hasattr(model, "named_steps"):
        steps = list(getattr(model, "named_steps", {}).values())
        if steps:
            labels = getattr(steps[-1], "classes_", None)
    if labels is None:
        return []
    return list(labels)


def model_capabilities(model: Any) -> dict[str, bool]:
    return {
        "predict_proba": callable(getattr(model, "predict_proba", None)),
        "predict": callable(getattr(model, "predict", None)),
        "decision_function": callable(getattr(model, "decision_function", None)),
    }


def choose_positive_index(labels: list[Any], positive_class: Any | None = None) -> int | None:
    if not labels:
        return None
    if positive_class is not None:
        for index, label in enumerate(labels):
            if str(label) == str(positive_class):
                return index
    return 1 if len(labels) > 1 else 0


def feature_alignment(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame_columns = set(map(str, df.columns))
    for feature in feature_names:
        rows.append(
            {
                "feature": feature,
                "status": "available" if feature in frame_columns else "missing",
                "dtype": str(df[feature].dtype) if feature in df.columns else "",
            }
        )
    extras = [str(column) for column in df.columns if str(column) not in set(feature_names)]
    for feature in extras:
        rows.append({"feature": feature, "status": "unused_extra", "dtype": str(df[feature].dtype)})
    return pd.DataFrame(rows)


def prepare_model_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [feature for feature in feature_names if feature not in df.columns]
    if missing:
        raise ValueError(f"Model input is missing column(s): {', '.join(missing)}")
    return df.loc[:, feature_names].copy()


def _as_1d_scores(values: Any, positive_index: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array.astype(float)
    if array.ndim == 2:
        column = positive_index if positive_index is not None else array.shape[1] - 1
        return array[:, int(column)].astype(float)
    raise ValueError(f"Unsupported prediction output shape: {array.shape}")


def predict_model_scores(
    model: Any,
    frame: pd.DataFrame,
    *,
    positive_class: Any | None = None,
    positive_index: int | None = None,
) -> ModelPrediction:
    labels = model_class_labels(model)
    resolved_positive_index = positive_index if positive_index is not None else choose_positive_index(labels, positive_class)
    if callable(getattr(model, "predict_proba", None)):
        return ModelPrediction(
            scores=_as_1d_scores(model.predict_proba(frame), resolved_positive_index),
            output_name="predict_proba",
            positive_index=resolved_positive_index,
        )
    if callable(getattr(model, "decision_function", None)):
        return ModelPrediction(
            scores=_as_1d_scores(model.decision_function(frame)),
            output_name="decision_function",
            positive_index=None,
        )
    if callable(getattr(model, "predict", None)):
        return ModelPrediction(
            scores=_as_1d_scores(model.predict(frame)),
            output_name="predict",
            positive_index=None,
        )
    raise ValueError("Uploaded model must expose predict_proba, decision_function, or predict.")


def dataframe_from_shap_array(data: Any, feature_names: list[str], template: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.loc[:, feature_names].copy()
    else:
        array = np.asarray(data, dtype=object)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        frame = pd.DataFrame(array, columns=feature_names)
    for feature in feature_names:
        if feature in template.columns and pd.api.types.is_numeric_dtype(template[feature]):
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return frame


def require_shap() -> Any:
    try:
        import shap
    except Exception as exc:
        raise RuntimeError("The shap package is required for SHAP Analysis. Install shap>=0.52.") from exc
    return shap


def kernel_shap_contributions(
    model: Any,
    background: pd.DataFrame,
    explain: pd.DataFrame,
    feature_names: list[str],
    *,
    positive_class: Any | None = None,
    positive_index: int | None = None,
    nsamples: int | str = 100,
) -> dict[str, Any]:
    shap = require_shap()
    background_frame = prepare_model_frame(background, feature_names)
    explain_frame = prepare_model_frame(explain, feature_names)

    def predict_fn(data: Any) -> np.ndarray:
        frame = dataframe_from_shap_array(data, feature_names, background_frame)
        return predict_model_scores(
            model,
            frame,
            positive_class=positive_class,
            positive_index=positive_index,
        ).scores

    explainer = shap.KernelExplainer(predict_fn, background_frame)
    try:
        shap_values = explainer.shap_values(explain_frame, nsamples=nsamples, silent=True)
    except TypeError:
        shap_values = explainer.shap_values(explain_frame, nsamples=nsamples)
    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected = float(np.asarray(expected_value, dtype=float).reshape(-1)[-1])
    else:
        expected = float(expected_value)
    return {
        "values": values,
        "expected_value": expected,
        "feature_names": list(feature_names),
        "row_index": explain_frame.index.tolist(),
    }


def shap_global_importance(shap_result: dict[str, Any]) -> pd.DataFrame:
    values = np.asarray(shap_result["values"], dtype=float)
    feature_names = list(shap_result["feature_names"])
    mean_abs = np.abs(values).mean(axis=0)
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap",
        ascending=False,
    )


def shap_local_contributions(shap_result: dict[str, Any], row_position: int = 0) -> pd.DataFrame:
    values = np.asarray(shap_result["values"], dtype=float)
    feature_names = list(shap_result["feature_names"])
    row_values = values[int(row_position)]
    return pd.DataFrame({"feature": feature_names, "shap_value": row_values}).assign(
        abs_shap=lambda frame: frame["shap_value"].abs()
    ).sort_values("abs_shap", ascending=False)


def changed_value_rows(base: pd.Series, edited: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in base.index:
        old = base[column]
        new = edited[column]
        old_missing = pd.isna(old)
        new_missing = pd.isna(new)
        changed = bool(old_missing != new_missing or (not old_missing and not new_missing and old != new))
        if changed:
            rows.append({"feature": str(column), "old_value": old, "new_value": new})
    return pd.DataFrame(rows)
