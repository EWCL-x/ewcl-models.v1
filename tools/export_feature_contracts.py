#!/usr/bin/env python3
"""Export immutable feature-order contracts for EWCL publication models.

Goal
----
Generate per-model feature lists (ordered) + hashes that adopters can use as a
"feature contract" to guarantee they are feeding the model exactly what it was
trained on.

This is intentionally model-centric:
- The model defines the required feature names and their order.
- Any feature extractor implementation must satisfy this contract.

Outputs
-------
contracts/features/
  EWCL-Sequence.feature_list.csv
  EWCL-Disorder.feature_list.csv
  EWCL-Structure.feature_list.csv
  contracts.manifest.json

The manifest includes SHA256 hashes and feature counts.

"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import caid_dual_benchmark as cdb  # noqa: E402


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature"])
        for r in rows:
            w.writerow([r])


def main() -> None:
    out_dir = Path("contracts/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = cdb.load_local_models()

    # ordered feature lists as defined by the model itself
    feature_lists = {
        "EWCL-Sequence": list(models.seq_feats),
        "EWCL-Disorder": list(models.dis_feats),
        "EWCL-Structure": list(models.str_feats),
    }

    manifest = {
        "schema": "ewcl.feature_contract.v1",
        "source": "local_models_in_repo",
        "files": {},
    }

    for model_name, feats in feature_lists.items():
        fname = model_name.replace(" ", "-") + ".feature_list.csv"
        out_path = out_dir / fname
        write_csv(out_path, feats)

        joined = "\n".join(feats) + "\n"
        manifest["files"][model_name] = {
            "path": str(out_path),
            "n_features": len(feats),
            "sha256": sha256_text(joined),
        }

    (Path("contracts") / "contracts.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("Wrote contracts/features and contracts/contracts.manifest.json")


if __name__ == "__main__":
    main()
