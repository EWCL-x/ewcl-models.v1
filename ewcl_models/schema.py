"""Feature alignment and schema validation for EWCL models.

Enforces strict feature ordering from feature_list.json and dtype rules
from schema_rules.json.  The canonical feature_list is the single source
of truth: order matters, column names are case-sensitive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SchemaRules:
    """Per-model missing-feature policy loaded from schema_rules.json."""

    allow_missing: bool = False
    fill_value: Optional[float] = None  # None  ➜  NaN
    require_numeric: bool = True
    allowed_meta_cols: List[str] = field(
        default_factory=lambda: ["protein_id", "residue_index", "aa"]
    )


def load_feature_list(path: str | Path) -> List[str]:
    """Load the canonical ordered feature list from JSON."""
    feats = json.loads(Path(path).read_text())
    if not isinstance(feats, list) or not feats:
        raise ValueError("feature_list.json must be a non-empty JSON list")
    return [str(x) for x in feats]


def load_schema_rules(path: str | Path) -> SchemaRules:
    """Load schema rules from JSON."""
    obj = json.loads(Path(path).read_text())
    return SchemaRules(
        allow_missing=bool(obj.get("allow_missing", False)),
        fill_value=obj.get("fill_value", None),
        require_numeric=bool(obj.get("require_numeric", True)),
        allowed_meta_cols=list(
            obj.get("allowed_meta_cols", ["protein_id", "residue_index", "aa"])
        ),
    )


def align_features(
    df: pd.DataFrame,
    feature_list: List[str],
    rules: SchemaRules,
) -> Tuple[pd.DataFrame, List[str]]:
    """Reorder *df* to match *feature_list*; fill or fail on missing columns.

    Returns
    -------
    aligned : pd.DataFrame
        Columns in the exact canonical order required by the model.
    missing : list[str]
        Any feature names that had to be filled (empty if none were missing).
    """
    missing = [c for c in feature_list if c not in df.columns]
    if missing and not rules.allow_missing:
        raise ValueError(
            f"Missing {len(missing)} required features "
            f"(first 20): {missing[:20]}"
        )
    fv = np.nan if rules.fill_value is None else float(rules.fill_value)
    for c in missing:
        df[c] = fv

    if rules.require_numeric:
        for c in feature_list:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[feature_list], missing
