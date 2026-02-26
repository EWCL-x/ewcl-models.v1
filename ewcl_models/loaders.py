"""Load EWCL model weights and contracts directly from zip archives.

Each zip contains:
    model/model.txt          or  model/model.pkl
    contract/feature_list.json
    contract/inference_contract.json
    contract/schema_rules.json
    calibration/calibration.json
    provenance/versions.json
    provenance/data_manifest.json
    provenance/training_meta.json
    docs/README_MODEL.md
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class LoadedModel:
    """In-memory representation of a model unpacked from its zip archive."""

    name: str
    feature_list: List[str]
    schema_rules: Dict[str, Any]
    inference_contract: Dict[str, Any]
    calibration: Dict[str, Any]
    versions: Dict[str, Any]
    model: Any  # lgb.Booster  or  sklearn estimator


def load_from_zip(zip_path: str | Path) -> LoadedModel:
    """Open *zip_path* and return a ready-to-predict :class:`LoadedModel`.

    Supports both LightGBM text files (``model.txt``) and scikit-learn
    pickles (``model.pkl``).
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        feature_list: List[str] = json.loads(
            z.read("contract/feature_list.json")
        )
        schema_rules: Dict[str, Any] = json.loads(
            z.read("contract/schema_rules.json")
        )
        inference_contract: Dict[str, Any] = json.loads(
            z.read("contract/inference_contract.json")
        )
        calibration: Dict[str, Any] = json.loads(
            z.read("calibration/calibration.json")
        )
        versions: Dict[str, Any] = json.loads(
            z.read("provenance/versions.json")
        )

        # Detect model format from archive contents
        names_in_zip = z.namelist()
        if "model/model.txt" in names_in_zip:
            import lightgbm as lgb

            model_txt = z.read("model/model.txt").decode("utf-8")
            model = lgb.Booster(model_str=model_txt)
        elif "model/model.pkl" in names_in_zip:
            import joblib

            model_bytes = z.read("model/model.pkl")
            model = joblib.load(io.BytesIO(model_bytes))
        else:
            raise FileNotFoundError(
                "zip must contain model/model.txt or model/model.pkl"
            )

    return LoadedModel(
        name=zip_path.stem,
        feature_list=feature_list,
        schema_rules=schema_rules,
        inference_contract=inference_contract,
        calibration=calibration,
        versions=versions,
        model=model,
    )


def load_model(model_dir: str | Path) -> LoadedModel:
    """Load an EWCL model from an **extracted** model directory.

    Parameters
    ----------
    model_dir : str or Path
        Path to the directory produced by unzipping a model archive, e.g.::

            unzip dist/EWCL-Sequence_v1.0.0.zip -d ~/ewcl_models/EWCL-Sequence_v1.0.0
            model = load_model("~/ewcl_models/EWCL-Sequence_v1.0.0")

    Raises
    ------
    ValueError
        If a ``.zip`` file path is passed instead of an extracted directory.
    FileNotFoundError
        If the path does not exist or is not a directory.
    """
    p = Path(model_dir).expanduser().resolve()

    # Guard: user accidentally passed the zip path
    if p.suffix.lower() == ".zip":
        raise ValueError(
            f"EWCL model bundles must be extracted before loading.\n"
            f"You passed a zip file: {p}\n\n"
            f"Unzip it first, then pass the extracted directory:\n"
            f"  unzip {p} -d ~/ewcl_models/{p.stem}\n"
            f"  load_model('~/ewcl_models/{p.stem}')"
        )

    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(
            f"Model path must be an existing extracted directory, got: {p}"
        )

    # Detect model weight format
    txt_path = p / "model" / "model.txt"
    pkl_path = p / "model" / "model.pkl"

    def _read_json(rel: str) -> dict:
        fp = p / rel
        if fp.exists():
            return json.loads(fp.read_text(encoding="utf-8"))
        return {}

    feature_list: List[str] = _read_json("contract/feature_list.json")
    schema_rules = _read_json("contract/schema_rules.json")
    inference_contract = _read_json("contract/inference_contract.json")
    calibration = _read_json("calibration/calibration.json")
    versions = _read_json("provenance/versions.json")

    if txt_path.exists():
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(txt_path))
    elif pkl_path.exists():
        import joblib
        model = joblib.load(pkl_path)
    else:
        raise FileNotFoundError(
            f"No model weights found in {p / 'model'}.\n"
            f"Expected model/model.txt or model/model.pkl."
        )

    return LoadedModel(
        name=p.name,
        feature_list=feature_list,
        schema_rules=schema_rules,
        inference_contract=inference_contract,
        calibration=calibration,
        versions=versions,
        model=model,
    )
