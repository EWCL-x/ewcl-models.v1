# EWCL-Models

**Frozen publication models for protein disorder and collapse propensity prediction.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![Weights: Non-Commercial](https://img.shields.io/badge/weights-non--commercial-orange.svg)](WEIGHTS_LICENSE.md)

> **Note:** This repository distributes *frozen publication model zips* (weights + contracts).
> The source code is MIT-licensed; the model artifacts (weights and zip archives) are
> research-only / non-commercial — see [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md).

---

## Overview

Three EWCL publication models, each distributed as a self-contained zip archive with
frozen feature extractors, contract files, and a Python inference package.

| Model | Description | Features | Format |
|---|---|---|---|
| **EWCL-Sequence** | Sequence-only disorder / collapse propensity | 249 | LightGBM `.txt` |
| **EWCL-Disorder** | Positional-context disorder / collapse propensity | 239 | LightGBM `.txt` |
| **EWCL-Structure** | Structure-aware disorder / collapse propensity | 230 | LightGBM `.txt` |

Each model zip contains:
```
model/model.txt
contract/feature_list.json
contract/inference_contract.json
contract/schema_rules.json
calibration/calibration.json
provenance/versions.json
provenance/data_manifest.json
provenance/training_meta.json
docs/README_MODEL.md
```

---

## Quickstart (local use)

### 1) Install the package

```bash
pip install -e .
```

Or with optional structure parsing:

```bash
pip install -e ".[structure]"
```

### 2) Download a model zip

Pre-built model archives are in `dist/` (or GitHub Releases):

```
dist/EWCL-Sequence_v1.0.0.zip
dist/EWCL-Disorder_v1.0.0.zip
dist/EWCL-Structure_v1.0.0.zip
```

### 3) Unzip to a local folder

**Model bundles must be extracted before loading.** Extract to any directory you choose:

```bash
unzip dist/EWCL-Sequence_v1.0.0.zip -d ~/ewcl_models/EWCL-Sequence_v1.0.0
```

You will get a directory like:

```
~/ewcl_models/EWCL-Sequence_v1.0.0/
  model/model.txt
  contract/
  calibration/
  provenance/
  docs/
```

### 4) Load and predict (Python API)

```python
from ewcl_models.loaders import load_model
from ewcl_models.feature_extractors import build_sequence_features
from ewcl_models.predictors import predict_from_features

# Load from extracted directory (NOT the .zip path)
model = load_model("~/ewcl_models/EWCL-Sequence_v1.0.0")

# Build features
fb = build_sequence_features("MKFLILLFNILCLFPVLAADNHGVS...")

# Predict
result = predict_from_features(fb.all_df, model)
print(result[["residue_index", "aa", "p_raw", "p"]])
```

> **Important:** pass the extracted directory path to `load_model()`.
> Do **not** pass the `.zip` file path — the loader will raise an error with instructions.

### 5) Command-line

```bash
# Sequence-only prediction
ewcl-predict --model ~/ewcl_models/EWCL-Sequence_v1.0.0 \
             --fasta examples/example.fasta \
             --out results.csv

# Structure-aware prediction (pLDDT from AlphaFold PDB)
ewcl-predict --model ~/ewcl_models/EWCL-Structure_v1.0.0 \
             --fasta examples/example.fasta \
             --pdb examples/example.pdb \
             --out results.csv

# Parquet output
ewcl-predict --model ~/ewcl_models/EWCL-Disorder_v1.0.0 \
             --fasta examples/example.fasta \
             --out results.parquet --format parquet
```

---

## Structure-Aware Prediction

```python
from ewcl_models.loaders import load_model
from ewcl_models.feature_extractors import build_sequence_features, compute_structure_features
import pandas as pd

model = load_model("~/ewcl_models/EWCL-Structure_v1.0.0")

seq = "MKFLILLFNILCLFPVLAADNHGVS..."
fb = build_sequence_features(seq)
struct_df = compute_structure_features(seq, plddt_vals=[95.2, 93.1, ...])

combined = pd.concat(
    [fb.base_df.reset_index(drop=True), struct_df.reset_index(drop=True)],
    axis=1,
)
result = predict_from_features(combined, model)
```

---

## EWCL–pLDDT Disagreement Index (EDI)

The EDI is a derived diagnostic that quantifies local discordance between
EWCL-Structure predictions and AlphaFold per-residue confidence (pLDDT).

```python
from ewcl_models.diagnostics import compute_edi, compute_cds, edi_segments

edi = compute_edi(result["p"].values, plddt_vals)
cds = compute_cds(result["p"].values, plddt_vals)
segments = edi_segments(edi, threshold=0.2, min_length=5)
```

---

## Repository Layout

```
EWCL-Models/
├── ewcl_models/                    # Python package (MIT-licensed code)
│   ├── loaders.py                  # load_model(dir) → LoadedModel
│   ├── predictors.py               # Inference + calibration
│   ├── schema.py                   # Feature alignment & validation
│   ├── diagnostics.py              # EDI / CDS derived diagnostics
│   ├── io.py                       # FASTA/PDB parsing, output writers
│   ├── cli.py                      # ewcl-predict CLI
│   └── feature_extractors/
│       ├── sequence_features.py    # Frozen sequence feature extractor
│       └── structure_features.py   # Frozen structure feature extractor
├── models/                         # Model contracts, calibration, provenance
│   ├── ewcl-sequence/              # + models/ewcl-sequence/model.txt (weights)
│   ├── ewcl-disorder/
│   └── ewcl-structure/
├── dist/                           # Pre-built zip archives (weights + contracts)
│   ├── EWCL-Sequence_v1.0.0.zip   ← non-commercial, see WEIGHTS_LICENSE.md
│   ├── EWCL-Disorder_v1.0.0.zip
│   ├── EWCL-Structure_v1.0.0.zip
│   └── SHA256SUMS.txt
├── examples/
├── tools/
│   └── build_model_zip.py          # Maintainers only — see below
├── requirements/
├── pyproject.toml
├── LICENSE                         # MIT (code)
├── WEIGHTS_LICENSE.md              # Non-commercial (model weights/zips)
├── COMMERCIAL.md                   # How to request commercial permission
├── CITATION.cff
└── CHANGELOG.md
```

Model bundles can be extracted to **any location on your system**;
provide the extracted directory path to `load_model()`.

---

## Requirements

- Python ≥ 3.9
- numpy, pandas, lightgbm, scikit-learn, joblib

Optional (structure parsing):
- gemmi (preferred) or biopython
- pyarrow (for Parquet output)

---

## Citation

> Cristino, L., & Uversky, V. N. Entropy-Weighted Collapse Likelihood (EWCL):
> sequence- and structure-conditioned predictors of intrinsic disorder and
> collapse propensity. *Manuscript in preparation*, 2026.

**BibTeX:**

```bibtex
@unpublished{CristinoUversky_EWCL_2026,
  author = {Cristino, Lucas and Uversky, Vladimir N.},
  title  = {Entropy-Weighted Collapse Likelihood (EWCL): sequence- and
            structure-conditioned predictors of intrinsic disorder and
            collapse propensity},
  note   = {Manuscript in preparation (preprint/journal submission forthcoming)},
  year   = {2026}
}
```

See also [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## License

| What | License |
|---|---|
| **Code** (`ewcl_models/`, `tools/`, `examples/`, etc.) | MIT — see [`LICENSE`](LICENSE) |
| **Model weights / zip archives** (`dist/*.zip`, `models/**/model.*`) | Research non-commercial only — see [`WEIGHTS_LICENSE.md`](WEIGHTS_LICENSE.md) |
| **Commercial use of weights** | Written permission required — see [`COMMERCIAL.md`](COMMERCIAL.md) |

---

## Maintainers: Building Model Zips (developers only)

> End users do not need to build zips. Use the pre-built archives in `dist/` or GitHub Releases.

Model weights (`model.txt`) must already be present in each model subdirectory
(e.g., `models/ewcl-sequence/model.txt`). Then run:

```bash
python tools/build_model_zip.py
```

This creates `dist/EWCL-{Sequence,Disorder,Structure}_v1.0.0.zip` and `dist/SHA256SUMS.txt`.
