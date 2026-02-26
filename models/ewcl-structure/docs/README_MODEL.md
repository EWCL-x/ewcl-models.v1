# EWCL-Structure

**Structure-aware Disorder / Collapse Propensity Model**

*One-line descriptor:* Structure-conditioned disorder propensity / collapse likelihood from 3D context.

*Primary tag:* structure-aware propensity
*Secondary tags:* PDB/AlphaFold compatible, geometry-conditioned, contextual disorder

## Model Details

| Property | Value |
|---|---|
| Framework | LightGBM Booster |
| Format | `.txt` (model string) |
| Features | 230 (224 base + 6 structure) |
| Prediction | `model.predict(X)` → disorder probability |
| Trees | 1000, objective = binary sigmoid |

## Feature Set

The 230 features consist of:

- **224 base sequence features** from `build_sequence_features().base_df`
- **6 structure features** from `compute_structure_features()`:
  - `plddt` — AlphaFold per-residue confidence (from b-factor; default 50 if no structure)
  - `struct_curvature` — |2nd derivative of hydropathy|
  - `struct_hydropathy` — rolling mean hydropathy (window=5)
  - `struct_charge` — rolling mean charge (window=5)
  - `struct_hydro_entropy` — rolling Shannon entropy of hydropathy (5-bin, window=5)
  - `struct_charge_entropy` — rolling Shannon entropy of charge (5-bin, window=5)

## Derived Diagnostic: EWCL–pLDDT Disagreement Index (EDI)

The EDI is a **derived analysis metric**, not part of the model itself.
It quantifies local discordance between EWCL-Structure predictions and
AlphaFold pLDDT confidence:

```
EDI(i) = EWCL_Structure(i) - (1 - pLDDT(i) / 100)
```

- **EDI > 0**: EWCL predicts more disorder than pLDDT expects → potential dynamic region that AlphaFold modelled as structured
- **EDI < 0**: AlphaFold is less confident than EWCL's disorder prediction → potential structured region with poor AF2 model quality
- **EDI ≈ 0**: Agreement between EWCL and AlphaFold confidence

See `ewcl_models.diagnostics` for the implementation.

## Usage

```python
from ewcl_models.loaders import load_from_zip
from ewcl_models.feature_extractors import build_sequence_features, compute_structure_features
from ewcl_models.predictors import predict_from_features
import pandas as pd

model = load_from_zip("EWCL-Structure_v1.0.0.zip")
seq = "MKFLILLFNILCLFPVLAADNHGVS..."
fb = build_sequence_features(seq)
struct_df = compute_structure_features(seq, plddt_vals=[95.2, 93.1, ...])
combined = pd.concat([fb.base_df.reset_index(drop=True),
                      struct_df.reset_index(drop=True)], axis=1)
result = predict_from_features(combined, model)

# Compute EDI diagnostic
from ewcl_models.diagnostics import compute_edi
edi = compute_edi(result["p"].values, plddt_vals)
```
