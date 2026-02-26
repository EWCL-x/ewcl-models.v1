"""Predict disorder probabilities from aligned feature matrices.

Supports optional post-hoc calibration (temperature scaling, Platt scaling)
stored in each model's ``calibration/calibration.json``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ewcl_models.loaders import LoadedModel
from ewcl_models.schema import SchemaRules, align_features


def _apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-7
    p = np.clip(p.astype(np.float64), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(logit / float(T))))


def _apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    eps = 1e-7
    p = np.clip(p.astype(np.float64), eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(a * logit + b)))


def calibrate(p_raw: np.ndarray, calib: Dict[str, Any]) -> np.ndarray:
    """Apply calibration transform to raw model probabilities."""
    method = calib.get("method", "none")
    if method == "none":
        return p_raw
    if method == "temperature":
        return _apply_temperature(p_raw, float(calib["temperature"]))
    if method == "platt":
        a = float(calib["parameters"]["a"])
        b = float(calib["parameters"]["b"])
        return _apply_platt(p_raw, a, b)
    raise ValueError(f"Unknown calibration method: {method}")


def predict_from_features(
    df_features: pd.DataFrame,
    loaded_model: LoadedModel,
) -> pd.DataFrame:
    """Run inference with strict schema alignment.

    Parameters
    ----------
    df_features : pd.DataFrame
        Must contain all columns listed in ``loaded_model.feature_list``.
        May also contain metadata columns (protein_id, residue_index, aa).
    loaded_model : LoadedModel
        Output of :func:`ewcl_models.loaders.load_from_zip`.

    Returns
    -------
    pd.DataFrame
        Per-residue predictions with columns ``p_raw`` and ``p`` (calibrated).
    """
    rules = SchemaRules(
        allow_missing=bool(
            loaded_model.schema_rules.get("allow_missing", False)
        ),
        fill_value=loaded_model.schema_rules.get("fill_value", None),
        require_numeric=True,
        allowed_meta_cols=list(
            loaded_model.schema_rules.get(
                "allowed_meta_cols", ["protein_id", "residue_index", "aa"]
            )
        ),
    )

    X_df, missing = align_features(
        df_features.copy(), loaded_model.feature_list, rules
    )
    X = X_df.to_numpy(np.float64, copy=False)

    # Dispatch: sklearn uses predict_proba, LightGBM Booster uses predict
    model = loaded_model.model
    model_type = type(model).__name__

    if hasattr(model, "predict_proba"):
        # sklearn LGBMClassifier
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        disorder_idx = classes.index(1)
        p_raw = proba[:, disorder_idx].astype(np.float64)
    else:
        # LightGBM Booster (predict returns probabilities for binary)
        p_raw = np.asarray(model.predict(X), dtype=np.float64)

    p_raw = np.clip(p_raw, 0.0, 1.0)
    p_cal = calibrate(p_raw, loaded_model.calibration)

    meta_cols = [
        c
        for c in ["protein_id", "residue_index", "aa"]
        if c in df_features.columns
    ]
    out = df_features[meta_cols].copy()
    out["p_raw"] = p_raw
    out["p"] = p_cal
    out.attrs["missing_features"] = missing
    out.attrs["model_name"] = loaded_model.name
    return out
