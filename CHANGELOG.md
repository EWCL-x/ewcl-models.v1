# Changelog

All notable changes to EWCL-Models will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] — 2026-03-03

### Fixed

- **Critical**: `ewcl_models/feature_extractors/sequence_features.py` was
  missing the v2 feature blocks (`lcd_*`, `pat_*`, `pos_*`, `prot_*`, `seq_*`).
  This caused locally-computed predictions to differ from the production backend
  by ~0.2–0.6 per residue.  The file is now synced with the canonical backend
  extractor (`backend/models/feature_extractors/ewclv1_features.py`).
  Verified 2026-03-03: local == backend (abs_max = 0.000e+00, P04637 + P37840).
- `build_sequence_features` retained as a backwards-compatible alias for
  `build_ewclv1_features` so existing user code continues to work.

## [1.0.0] — 2026

> Cristino, L., & Uversky, V. N. *Manuscript in preparation*, 2026.

### Added

- **EWCL-Sequence** v1.0.0 — Sequence-only disorder / collapse propensity model.
  scikit-learn LGBMClassifier, 249 features, 5000 trees.
- **EWCL-Disorder** v1.0.0 — Positional-context disorder / collapse propensity model.
  LightGBM Booster, 239 features, 1000 trees.
- **EWCL-Structure** v1.0.0 — Structure-aware disorder / collapse propensity model.
  LightGBM Booster, 230 features (224 base + 6 structure), 1000 trees.
- Frozen feature extractors (`sequence_features.py`, `structure_features.py`).
- Model zip packaging with contract files (feature_list, inference_contract,
  schema_rules, calibration, provenance).
- Python package with CLI (`ewcl-predict`).
- EWCL–pLDDT Disagreement Index (EDI) and Confidence Disagreement Score (CDS)
  as derived diagnostics in `ewcl_models.diagnostics`.
- CSV, Parquet, and JSONL output formats.
