# EWCL-Models

**Frozen publication models for protein disorder and collapse propensity prediction.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository distributes the three EWCL publication models as
self-contained zip archives with frozen feature extractors, contract files,
and a Python inference package.

| Model | Description | Features | Format |
|---|---|---|---|
| **EWCL-Sequence** | Sequence-only disorder / collapse propensity | 249 | sklearn `.pkl` |
| **EWCL-Disorder** | Positional-context disorder / collapse propensity | 239 | LightGBM `.txt` |
| **EWCL-Structure** | Structure-aware disorder / collapse propensity | 230 | LightGBM `.txt` |

Each model zip contains:
```
model/model.txt  (or model.pkl)
contract/feature_list.json
contract/inference_contract.json
contract/schema_rules.json
calibration/calibration.json
provenance/versions.json
provenance/data_manifest.json
provenance/training_meta.json
docs/README_MODEL.md
```

## Quick Start

### Install

```bash
pip install -e .
```

Or with optional structure parsing:

```bash
pip install -e ".[structure]"
```

### Command-line

```bash
# Sequence-only prediction
ewcl-predict --model dist/EWCL-Sequence_v1.0.0.zip \
             --fasta examples/example.fasta \
             --out results.csv

# With structure (pLDDT from AlphaFold PDB)
ewcl-predict --model dist/EWCL-Structure_v1.0.0.zip \
             --fasta examples/example.fasta \
             --pdb examples/example.pdb \
             --out results.csv

# Parquet output
ewcl-predict --model dist/EWCL-Disorder_v1.0.0.zip \
             --fasta examples/example.fasta \
             --out results.parquet --format parquet
```

### Python API

```python
from ewcl_models.loaders import load_from_zip
from ewcl_models.feature_extractors import build_sequence_features
from ewcl_models.predictors import predict_from_features

# Load model
model = load_from_zip("dist/EWCL-Sequence_v1.0.0.zip")

# Build features
fb = build_sequence_features("MKFLILLFNILCLFPVLAADNHGVS...")

# Predict
result = predict_from_features(fb.all_df, model)
print(result[["protein_id", "residue_index", "aa", "p"]])
```

### Structure-aware prediction

```python
from ewcl_models.feature_extractors import (
    build_sequence_features,
    compute_structure_features,
)
import pandas as pd

model = load_from_zip("dist/EWCL-Structure_v1.0.0.zip")

seq = "MKFLILLFNILCLFPVLAADNHGVS..."
fb = build_sequence_features(seq)
struct_df = compute_structure_features(seq, plddt_vals=[95.2, 93.1, ...])

combined = pd.concat(
    [fb.base_df.reset_index(drop=True), struct_df.reset_index(drop=True)],
    axis=1,
)
result = predict_from_features(combined, model)
```

### EWCL–pLDDT Disagreement Index (EDI)

The EDI is a **derived diagnostic** — not part of any model — that
quantifies local discordance between EWCL-Structure predictions and
AlphaFold per-residue confidence (pLDDT).

```python
from ewcl_models.diagnostics import compute_edi, compute_cds, edi_segments

edi = compute_edi(result["p"].values, plddt_vals)
cds = compute_cds(result["p"].values, plddt_vals)
segments = edi_segments(edi, threshold=0.2, min_length=5)
```

## Repository Layout

```
EWCL-Models/
├── ewcl_models/                    # Python package
│   ├── __init__.py
│   ├── version.py
│   ├── schema.py                   # Feature alignment & validation
│   ├── loaders.py                  # Zip → LoadedModel
│   ├── predictors.py               # Inference with calibration
│   ├── diagnostics.py              # EDI / CDS derived diagnostics
│   ├── io.py                       # FASTA/PDB parsing, output writers
│   ├── cli.py                      # ewcl-predict CLI
│   └── feature_extractors/
│       ├── sequence_features.py    # Frozen sequence feature extractor
│       └── structure_features.py   # Frozen structure feature extractor
├── models/                         # Model contracts & provenance
│   ├── EWCL-Sequence/
│   ├── EWCL-Disorder/
│   └── EWCL-Structure/
├── dist/                           # Built zip archives (after build)
├── examples/
├── tools/
│   └── build_model_zip.py          # Zip packaging script
├── requirements/
├── pyproject.toml
├── LICENSE
├── CITATION.cff
└── CHANGELOG.md
```

## Building Model Zips

Before building zips, copy trained model weight files into `models/`:

```bash
cp /path/to/EWCL-Sequence.pkl  models/
cp /path/to/EWCL-Disorder.txt  models/
cp /path/to/EWCL-Structure.txt models/
```

Then:

```bash
python tools/build_model_zip.py
```

This creates `dist/EWCL-{Sequence,Disorder,Structure}_v1.0.0.zip` and
`dist/SHA256SUMS.txt`.

## Requirements

- Python ≥ 3.9
- numpy
- pandas
- lightgbm
- scikit-learn
- joblib

Optional (for structure parsing):
- gemmi (preferred) or biopython
- pyarrow (for Parquet output)

## Citation

> Cristino, L., & Uversky, V. N. Entropy-Weighted Collapse Likelihood (EWCL):
> sequence- and structure-conditioned predictors of intrinsic disorder and
> collapse propensity. *Manuscript in preparation*, 2026.

Journal details and DOI are pending — a preprint/journal submission is forthcoming.

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

## License

MIT — see [LICENSE](LICENSE).
