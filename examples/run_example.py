#!/usr/bin/env python3
"""Example: Python API usage for all three EWCL models.

Run from the repository root after installing:
    pip install -e .
    python tools/build_model_zip.py
    python examples/run_example.py
"""

from pathlib import Path

import pandas as pd

from ewcl_models.diagnostics import compute_cds, compute_edi, edi_segments
from ewcl_models.feature_extractors import (
    build_sequence_features,
    compute_structure_features,
)
from ewcl_models.io import read_fasta
from ewcl_models.loaders import load_from_zip
from ewcl_models.predictors import predict_from_features

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"


def main() -> None:
    # ── Read example FASTA ────────────────────────────────────────────────
    records = read_fasta(ROOT / "examples" / "example.fasta")
    print(f"Read {len(records)} sequence(s)\n")

    # ── EWCL-Sequence ─────────────────────────────────────────────────────
    model_seq = load_from_zip(DIST / "EWCL-Sequence_v1.0.0.zip")
    print(f"Loaded {model_seq.name} ({len(model_seq.feature_list)} features)")

    for seq_id, sequence in records:
        fb = build_sequence_features(sequence)
        df = fb.all_df.copy()
        df["protein_id"] = seq_id
        df["residue_index"] = range(1, len(sequence) + 1)
        df["aa"] = list(sequence)
        result = predict_from_features(df, model_seq)
        print(f"  {seq_id}: mean_p={result['p'].mean():.4f}")

    # ── EWCL-Disorder ─────────────────────────────────────────────────────
    print()
    model_dis = load_from_zip(DIST / "EWCL-Disorder_v1.0.0.zip")
    print(f"Loaded {model_dis.name} ({len(model_dis.feature_list)} features)")

    for seq_id, sequence in records:
        fb = build_sequence_features(sequence)
        df = fb.base_df.copy()
        df["protein_id"] = seq_id
        df["residue_index"] = range(1, len(sequence) + 1)
        df["aa"] = list(sequence)
        result = predict_from_features(df, model_dis)
        print(f"  {seq_id}: mean_p={result['p'].mean():.4f}")

    # ── EWCL-Structure (FASTA only → pLDDT=50) ───────────────────────────
    print()
    model_str = load_from_zip(DIST / "EWCL-Structure_v1.0.0.zip")
    print(f"Loaded {model_str.name} ({len(model_str.feature_list)} features)")

    for seq_id, sequence in records:
        fb = build_sequence_features(sequence)
        struct_df = compute_structure_features(sequence, plddt_vals=None)
        combined = pd.concat(
            [fb.base_df.reset_index(drop=True), struct_df.reset_index(drop=True)],
            axis=1,
        )
        combined["protein_id"] = seq_id
        combined["residue_index"] = range(1, len(sequence) + 1)
        combined["aa"] = list(sequence)
        result = predict_from_features(combined, model_str)
        print(f"  {seq_id}: mean_p={result['p'].mean():.4f}")

        # ── EDI diagnostic (with synthetic pLDDT for demo) ────────────────
        import numpy as np

        fake_plddt = np.full(len(sequence), 50.0)
        edi = compute_edi(result["p"].values, fake_plddt)
        cds = compute_cds(result["p"].values, fake_plddt)
        segs = edi_segments(edi, threshold=0.15, min_length=3)
        print(f"    EDI: CDS={cds:.4f}, {len(segs)} discordant segments")

    print("\nDone.")


if __name__ == "__main__":
    main()
