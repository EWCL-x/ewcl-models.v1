#!/usr/bin/env python3
"""Build model zip archives for distribution.

Usage
-----
    python tools/build_model_zip.py

Zip layout::

    model/model.txt   (LightGBM — EWCL-Disorder, EWCL-Structure)
    model/model.pkl   (sklearn/joblib — EWCL-Sequence)
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

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DIST_DIR = ROOT / "dist"
VERSION = "1.0.0"

MODELS = [
    ("EWCL-Sequence", "ewcl-sequence"),
    ("EWCL-Disorder", "ewcl-disorder"),
    ("EWCL-Structure", "ewcl-structure"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_zip(model_name: str, subfolder: str) -> Path:
    model_dir = MODELS_DIR / subfolder

    # Auto-detect weight file: prefer .pkl, fallback to .txt
    pkl_path = model_dir / "model" / "model.pkl"
    txt_path = model_dir / "model" / "model.txt"

    if pkl_path.exists():
        weight_path = pkl_path
        weight_arc = "model/model.pkl"
    elif txt_path.exists():
        weight_path = txt_path
        weight_arc = "model/model.txt"
    else:
        raise FileNotFoundError(
            f"No model weights found for {model_name}.\n"
            f"Expected: {model_dir}/model/model.pkl  or  model.txt"
        )

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    zip_name = f"{model_name}_v{VERSION}.zip"
    zip_path = DIST_DIR / zip_name
    feat_path = MODELS_DIR / f"{model_name}_feature_list.json"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(weight_path, weight_arc)

        if feat_path.exists():
            z.write(feat_path, "contract/feature_list.json")

        for sub in ["contract", "calibration", "provenance"]:
            sub_dir = model_dir / sub
            if sub_dir.exists():
                for f in sorted(sub_dir.iterdir()):
                    if f.is_file() and f.suffix == ".json":
                        z.write(f, f"{sub}/{f.name}")

        readme_path = model_dir / "docs" / "README_MODEL.md"
        if readme_path.exists():
            z.write(readme_path, "docs/README_MODEL.md")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ {zip_name}  ({size_mb:.1f} MB)  [{weight_arc}]")
    return zip_path


def main() -> None:
    print(f"Building model zips (v{VERSION})…\n")
    sha_lines: list[str] = []
    for model_name, subfolder in MODELS:
        try:
            zp = build_zip(model_name, subfolder)
            sha = sha256_file(zp)
            sha_lines.append(f"{sha}  {zp.name}")
        except FileNotFoundError as e:
            print(f"  ✗ {model_name}: {e}")

    if sha_lines:
        sha_path = DIST_DIR / "SHA256SUMS.txt"
        sha_path.write_text("\n".join(sha_lines) + "\n")
        print(f"\n  ✓ SHA256SUMS.txt → {sha_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
