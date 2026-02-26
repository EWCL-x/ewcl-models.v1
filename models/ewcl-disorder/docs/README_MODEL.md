# EWCL-Disorder

**Positional-context Disorder / Collapse Propensity Model**

*One-line descriptor:* Sequence-based disorder propensity enriched with positional and protein-level context.

## Model Details

| Property | Value |
|---|---|
| Framework | LightGBM Booster |
| Format | `.txt` (model string) |
| Features | 239 (224 base + 15 positional) |
| Prediction | `model.predict(X)` → disorder probability |
| Trees | 1000, objective = binary sigmoid |

## Feature Set

The 239 features come from `build_sequence_features()` returning `base_df`:

- **224 base sequence features** (shared with EWCL-Sequence and EWCL-Structure)
- **15 positional features** unique to this model:
  - `pos_norm`, `pos_dist_n`, `pos_dist_c`
  - `pos_is_n10`, `pos_is_n20`, `pos_is_n30`
  - `pos_is_c10`, `pos_is_c20`, `pos_is_c30`
  - `pos_nexp10`, `pos_nexp30`, `pos_cexp10`, `pos_cexp30`
  - `prot_log_len`, `prot_inv_len`

## Usage

```python
from ewcl_models.loaders import load_from_zip
from ewcl_models.feature_extractors import build_sequence_features
from ewcl_models.predictors import predict_from_features

model = load_from_zip("EWCL-Disorder_v1.0.0.zip")
fb = build_sequence_features("MKFLILLFNILCLFPVLAADNHGVS...")
result = predict_from_features(fb.base_df, model)
```
