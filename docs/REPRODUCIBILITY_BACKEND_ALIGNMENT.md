# EWCL Reproducibility & Backend Alignment Guide

> **For reviewers, adopters, and contributors.**  
> This is the single source of truth for verifying that local predictions,
> published model zips, and the deployed API all produce identical results.

---

## TL;DR (30-second check)

```bash
# Requires Python 3.10+, numpy, lightgbm, scikit-learn (see requirements.txt)
python tools/compare_local_zip_backend.py \
  --backend https://ewcl-api-production.up.railway.app \
  --uniprot P04637 \
  --uniprot P37840
```

Expected output (verified 2026-03-03):

```
== P04637 ==
EWCL-Sequence: local vs zip     abs_max=0.000e+00  abs_mean=0.000e+00
EWCL-Sequence: local vs backend abs_max=0.000e+00  abs_mean=0.000e+00
EWCL-Disorder: local vs zip     abs_max=0.000e+00  abs_mean=0.000e+00
EWCL-Disorder: local vs backend abs_max=5.551e-17  abs_mean=1.412e-19

== P37840 ==
EWCL-Sequence: local vs zip     abs_max=0.000e+00  abs_mean=0.000e+00
EWCL-Sequence: local vs backend abs_max=0.000e+00  abs_mean=0.000e+00
EWCL-Disorder: local vs zip     abs_max=0.000e+00  abs_mean=0.000e+00
EWCL-Disorder: local vs backend abs_max=0.000e+00  abs_mean=0.000e+00

OK
```

If you see this → **local, zip, and backend are identical.**

---

## Background: what went wrong and what was fixed

### The root cause

Earlier benchmark scripts shipped a **vendored / inline copy** of the feature
extractor that had drifted from the production backend extractor.  
The backend was computing the *correct* features (including the v2 feature
blocks: `lcd_*`, `pat_*`, `pos_*`, `prot_*`, `seq_*`), while the benchmark
copy was missing or computing those differently.

This produced a large, systematic per-residue gap between local/zip predictions
and the deployed API — visible as ~0.2–0.6 differences across broad sequence
regions in the P04637 (p53) and P37840 (α-synuclein) overlay plots.

### The fix (2026-03-03)

`caid_dual_benchmark.py::_build_ewclv1_features()` was rewritten to **delegate
directly** to the backend canonical extractor:

```python
from backend.models.feature_extractors.ewclv1_features import build_ewclv1_features
fb = build_ewclv1_features(seq, pssm=pssm)
```

The old inlined implementation was removed. There is now exactly **one** feature
extractor in this repository.

---

## Source of truth: feature extraction

The canonical EWCL v1 feature extractor lives at:

```
backend/models/feature_extractors/ewclv1_features.py
    └── build_ewclv1_features(seq, pssm=None)
```

All benchmark, contract validation, and plotting tools **must** call this
function (directly or via `caid_dual_benchmark._build_ewclv1_features`).

### Feature blocks produced

| Block | Prefixes | Description |
|---|---|---|
| Biophysical windowed stats | `hydro_`, `polar_`, `vdw_`, `flex_`, `bulk_`, `helix_prop_`, `sheet_prop_`, `charge_` | Rolling mean/std/min/max at windows 5/11/25/50/100 |
| Entropy & complexity | `entropy_`, `low_complex_`, `comp_bias_`, `uversky_dist_` | Per-window entropy, Uversky boundary distance |
| Composition | `comp_` | Per-AA fraction + aromatic/positive/negative/polar/aliphatic/P/G totals |
| Position | `pos_` | Relative position, terminus distance, exponential decay terms |
| Sequence-level | `seq_` | Charged fraction, NCPR |
| Protein-level | `prot_` | log(length), inv_length, mean hydropathy, DP/OP/aro fractions, kappa |
| LCD (low-complexity) | `lcd_` | WF complexity, LZ proxy, longest run, aromatic density, repeat score |
| Pattern | `pat_` | Hydropathy variance/MAD, charge transitions, disorder block fraction, kappa |
| Poly-run flags | `in_poly_*_run_ge3` | Per-AA homopolymer run flags (P,E,K,Q,S,G,D,N) |
| Misc | `scd_local`, `is_unknown_aa` | Local charge decoration, unknown-AA flag |
| PSSM (zero-filled if absent) | AA letters + `pssm_entropy/max_score/variance/has_pssm_data` | PSSM-derived features |

**Total per model:**
- EWCL-Sequence: **249 features** (base + PSSM)
- EWCL-Disorder: **224 features** (model-declared subset of base)
- EWCL-Structure: **230 features** (base + 6 structure features)

Feature order is always **determined by the model itself** via
`model.feature_name()` (LightGBM) or `model.feature_names_in_` (sklearn).

---

## Verification tools

### 1. Numeric identity: local == zip == backend

```bash
python tools/compare_local_zip_backend.py \
  --backend https://ewcl-api-production.up.railway.app \
  --uniprot P04637 --uniprot P37840
```

Expected: all `abs_max ≤ 1e-16`.

### 2. Overlay plots for visual review

```bash
python tools/plot_backend_vs_local.py \
  --backend https://ewcl-api-production.up.railway.app \
  --uniprot P04637 --uniprot P37840
```

Plots saved to `plots_backend_vs_local/`.  
- Score plots are clipped to [0, 1]; threshold line at 0.5.
- Local and zip curves should sit exactly on top of the backend curve.  
- **Δ plots** (`*_diff_sequence.png`, `*_diff_disorder.png`) show
  `Δ = backend − local`. Negative values mean backend < local at that residue —
  these are *differences*, not scores.

### 3. Feature contract export

```bash
python tools/export_feature_contracts.py --local-models models/
```

Writes ordered feature CSVs to `contracts/features/` and SHA256 hashes to
`contracts/contracts.manifest.json`.

### 4. Feature contract validation

```bash
python tools/validate_feature_contracts.py \
  --local-models models/ \
  --allow-structure-fallback
```

Confirms:
- model-declared feature list matches contract CSV exactly
- canonical extractor produces all required features
- predictions run and lengths match sequences

---

## Interpreting prediction scores

All EWCL models output **calibrated probabilities in [0, 1]**:

| Score range | Meaning |
|---|---|
| > 0.5 | Disordered (EWCL-Sequence / EWCL-Disorder) |
| < 0.5 | Ordered / structured |

The API always returns probabilities (not raw logits or margins).

---

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `local vs backend` large diff | Extractor in local code diverged | Ensure `_build_ewclv1_features` delegates to `backend/...ewclv1_features.py` |
| `local vs zip` large diff | Wrong weights or feature list in zip | Re-download zip; re-export contracts |
| Feature mismatch in validator | Old contract CSV | Re-run `export_feature_contracts.py` against current models |
| Negative values in score plot | You're looking at a Δ plot | Check the plot title |
| Scores degenerate (all 0 or 1) | Wrong / old inverted model loaded | Confirm model paths in Railway env vars |

---

## Reproducibility statement (for paper/submission)

> All per-residue predictions reported in this work were generated using the
> EWCL v1 canonical feature extractor
> (`backend/models/feature_extractors/ewclv1_features.py`) with feature order
> determined by the model artifact's own declared feature list.  Local
> predictions are numerically identical (within IEEE 754 floating-point
> tolerance) to the deployed backend API and to the published model zips in
> `EWCL-x/ewcl-models.v1`.  Reproducibility was verified on 2026-03-03 using
> `tools/compare_local_zip_backend.py` against UniProt entries P04637 (p53)
> and P37840 (α-synuclein).

---

## File map

```
backend/
  models/
    feature_extractors/
      ewclv1_features.py          ← canonical extractor (source of truth)

caid_dual_benchmark.py            ← CAID benchmark runner; delegates to backend extractor
tools/
  compare_local_zip_backend.py    ← numeric identity check (local == zip == backend)
  plot_backend_vs_local.py        ← overlay + diff plots
  export_feature_contracts.py     ← write per-model ordered feature CSVs + SHA256 manifest
  validate_feature_contracts.py   ← verify extractor satisfies each model's contract

contracts/
  features/
    EWCL-Sequence.feature_list.csv
    EWCL-Disorder.feature_list.csv
    EWCL-Structure.feature_list.csv
  contracts.manifest.json         ← SHA256 hashes + feature counts

plots_backend_vs_local/           ← output plots from plot_backend_vs_local.py
```
