#!/usr/bin/env python3
"""Build model zip archives for distribution.

Usage
-----
    python tools/build_model_zip.py

This script packages each model's weights and contract files into a
self-contained zip archive under ``dist/``.

Zip layout::

    model/model.txt            (or model/model.pkl)
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
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DIST_DIR = ROOT / "dist"
VERSION = "1.0.0"

# Each entry: (model_name, model_weight_filename, feature_list_json)
MODELS = [
    (
        "EWCL-Sequence",
        "EWCL-Sequence.pkl",
        "EWCL-Sequence_feature_list.json",
    ),
    (
        "EWCL-Disorder",
        "EWCL-Disorder.txt",
        "EWCL-Disorder_feature_list.json",
    ),
    (
        "EWCL-Structure",
        "EWCL-Structure.txt",
        "EWCL-Structure_feature_list.json",
    ),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_zip(name: str, weight_file: str, feat_list_file: str) -> Path:
    model_dir = MODELS_DIR / name
    weight_path = MODELS_DIR / weight_file

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weight_path}\n"
            f"Copy the trained model file to {MODELS_DIR}/"
        )

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    zip_name = f"{name}_v{VERSION}.zip"
    zip_path = DIST_DIR / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        # Model weights
        ext = weight_path.suffix
        z.write(weight_path, f"model/model{ext}")

        # Feature list
        feat_path = MODELS_DIR / feat_list_file
        z.write(feat_path, "contract/feature_list.json")

        # Contract files
        for sub in ["contract", "calibration", "provenance"]:
            sub_dir = model_dir / sub
            if sub_dir.exists():
                for f in sorted(sub_dir.iterdir()):
                    if f.is_file() and f.suffix == ".json":
                        z.write(f, f"{sub}/{f.name}")

        # README
        readme_path = model_dir / "docs" / "README_MODEL.md"
        if readme_path.exists():
            z.write(readme_path, "docs/README_MODEL.md")

    print(f"  ✓ {zip_name}  ({zip_path.stat().st_size:,} bytes)")
    return zip_path


def main() -> None:
    print(f"Building model zips (v{VERSION})…\n")

    sha_lines: list[str] = []
    for name, weight, feats in MODELS:
        try:
            zp = build_zip(name, weight, feats)
            sha = sha256_file(zp)
            sha_lines.append(f"{sha}  {zp.name}")
        except FileNotFoundError as e:
            print(f"  ✗ {name}: {e}")

    # Write SHA256SUMS
    if sha_lines:
        sha_path = DIST_DIR / "SHA256SUMS.txt"
        sha_path.write_text("\n".join(sha_lines) + "\n")
        print(f"\n  ✓ SHA256SUMS.txt written to {sha_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
