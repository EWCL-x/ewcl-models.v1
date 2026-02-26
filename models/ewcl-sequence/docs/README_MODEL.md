# EWCL-Sequence

**Sequence-only Disorder / Collapse Propensity Model**

*One-line descriptor:* Pure-sequence disorder propensity from physicochemical and compositional features.

## Model Details

| Property | Value |
|---|---|
| Framework | scikit-learn LGBMClassifier |
| Format | `.pkl` (joblib) |
| Features | 249 (224 base + 20 AA one-hot + 4 PSSM + 1 is_unknown_aa) |
| Prediction | `predict_proba` → disorder probability |
| Boosting | GBDT, 5000 trees, max_depth=12 |

## Feature Set

The 249 features come from `build_sequence_features()` returning `all_df`:

- **8 physicochemical tracks** (hydropathy, polarity, vdw_volume, flexibility, bulkiness, helix_prop, sheet_prop, charge_pH7) × 5 windows × 4 stats = 160 rolling features + 8 base values
- **Windowed entropy, low complexity, composition bias, Uversky distance** (5 windows each)
- **Global composition** (20 AA fractions + 7 aggregate fractions)
- **Positional features** (relative position, terminal distances, decay)
- **Sequence/protein-level features** (charge fractions, Uversky axis, kappa, etc.)
- **LCD features** (top1 dominance, WF complexity, LZ proxy, run stats)
- **Pattern features** (hydropathy variance/MAD, charge transitions, disorder blocks)
- **Poly-AA run flags**, **SCD local**, **is_unknown_aa**
- **20 PSSM AA scores** + pssm_entropy + pssm_max_score + pssm_variance + has_pssm_data

## Usage

```python
from ewcl_models.loaders import load_from_zip
from ewcl_models.feature_extractors import build_sequence_features
from ewcl_models.predictors import predict_from_features

model = load_from_zip("EWCL-Sequence_v1.0.0.zip")
fb = build_sequence_features("MKFLILLFNILCLFPVLAADNHGVS...")
result = predict_from_features(fb.all_df, model)
```
