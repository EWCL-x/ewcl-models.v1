# EWCL feature contracts (v1)

This folder contains **adopter-facing feature contracts** for EWCL v1 models.

A *feature contract* is an immutable, ordered list of feature names that a model
expects at inference time.

Why this exists
--------------
These models are sensitive to feature ordering. If an adopter computes the right
features but feeds them in a different order, predictions will be wrong.

To prevent this, the model distribution includes a frozen feature list and a
validator script.

What is in here
---------------
- `features/`
  - `EWCL-Sequence.feature_list.csv`
  - `EWCL-Disorder.feature_list.csv`
  - `EWCL-Structure.feature_list.csv`
- `contracts.manifest.json`
  - Hashes (SHA256) + counts for each list.

How to regenerate (repo maintainers)
-----------------------------------
From this repository root:

```bash
python tools/export_feature_contracts.py
```

How to validate your integration (adopters)
-------------------------------------------
From this repository root:

```bash
# Validates the frozen extractor vs the contracts and runs predictions.
python tools/validate_feature_contracts.py --pdb-dir .

# If you don't have AlphaFold PDBs available and only care about sequence models:
python tools/validate_feature_contracts.py --allow-structure-fallback
```

Canonical feature extractor
---------------------------
The source of truth for EWCL v1 feature generation in this repo is the backend
implementation:

- `backend/models/feature_extractors/ewclv1_features.py::build_ewclv1_features`

All benchmark/validation tooling routes feature generation through that
implementation (via `caid_dual_benchmark.py::_build_ewclv1_features`).

Cross-check local vs published zips vs deployed backend
------------------------------------------------------

```bash
python tools/compare_local_zip_backend.py \
  --backend https://ewcl-api-production.up.railway.app \
  --uniprot P00441 \
  --uniprot P04637
```

Expected results
----------------
- **local vs zip** should be *exactly identical* for Sequence + Disorder (abs_max=0).
- **local vs backend** should also be identical **if** the deployed backend is
  running the same version with the same feature implementation.

If local vs backend differs, the backend deployment is out of sync with the
published contracts/models.
