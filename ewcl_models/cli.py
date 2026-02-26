"""Command-line interface for EWCL-Models.

Usage
-----
    ewcl-predict --model MODEL.zip --fasta INPUT.fasta --out results.csv
    ewcl-predict --model MODEL.zip --fasta INPUT.fasta --pdb structure.pdb --out results.csv
    ewcl-predict --model MODEL.zip --fasta INPUT.fasta --out results.parquet --format parquet

The CLI loads a model zip archive, extracts features from the input
sequence(s), runs inference, and writes per-residue predictions.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ewcl_models.feature_extractors.sequence_features import (
    build_sequence_features,
)
from ewcl_models.feature_extractors.structure_features import (
    compute_structure_features,
)
from ewcl_models.io import (
    WRITERS,
    extract_plddt,
    read_fasta,
    read_structure,
)
from ewcl_models.loaders import LoadedModel, load_from_zip
from ewcl_models.predictors import predict_from_features
from ewcl_models.version import __version__

log = logging.getLogger("ewcl_models")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ewcl-predict",
        description="Run EWCL disorder prediction from a model zip archive.",
    )
    p.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to model zip (e.g. EWCL-Sequence_v1.0.0.zip).",
    )
    p.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="Input FASTA file (one or more sequences).",
    )
    p.add_argument(
        "--pdb",
        type=Path,
        default=None,
        help="Optional PDB/mmCIF file for structure features (EWCL-Structure only).",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output file path.",
    )
    p.add_argument(
        "--format",
        choices=["csv", "parquet", "jsonl"],
        default="csv",
        help="Output format (default: csv).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Disorder classification threshold (default: 0.5).",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return p


def _needs_structure(loaded: LoadedModel) -> bool:
    """Check if the model requires structure features."""
    struct_feats = {
        "plddt",
        "struct_curvature",
        "struct_hydropathy",
        "struct_charge",
        "struct_hydro_entropy",
        "struct_charge_entropy",
    }
    return bool(struct_feats & set(loaded.feature_list))


def _run_single(
    seq_id: str,
    sequence: str,
    loaded: LoadedModel,
    plddt_vals: Optional[List[float]],
    threshold: float,
) -> pd.DataFrame:
    """Run prediction on a single sequence."""
    fb = build_sequence_features(sequence, pssm=None)

    # Choose the correct base DataFrame based on model feature set
    has_pssm_feats = any(
        c in loaded.feature_list
        for c in ["has_pssm_data", "pssm_entropy", "pssm_max_score"]
    )
    has_struct_feats = _needs_structure(loaded)

    if has_pssm_feats:
        # EWCL-Sequence: uses all_df (249 features)
        base = fb.all_df.copy()
    else:
        # EWCL-Disorder / EWCL-Structure: uses base_df (224 features)
        base = fb.base_df.copy()

    if has_struct_feats:
        struct_df = compute_structure_features(sequence, plddt_vals)
        base = pd.concat(
            [base.reset_index(drop=True), struct_df.reset_index(drop=True)],
            axis=1,
        )

    # Add metadata columns
    base["protein_id"] = seq_id
    base["residue_index"] = range(1, len(sequence) + 1)
    base["aa"] = list(sequence)

    result = predict_from_features(base, loaded)
    result["disorder_binary"] = (result["p"] >= threshold).astype(int)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    # Validate inputs
    if not args.model.exists():
        log.error("Model zip not found: %s", args.model)
        return 1
    if not args.fasta.exists():
        log.error("FASTA file not found: %s", args.fasta)
        return 1

    # Load model
    t0 = time.perf_counter()
    loaded = load_from_zip(args.model)
    log.info(
        "Loaded %s  (%d features)  in %.2fs",
        loaded.name,
        len(loaded.feature_list),
        time.perf_counter() - t0,
    )

    # Parse structure if provided
    plddt_map: dict[str, List[float]] = {}
    if args.pdb is not None:
        if not args.pdb.exists():
            log.error("PDB/CIF file not found: %s", args.pdb)
            return 1
        struct_df, struct_seq, parser_used = read_structure(args.pdb)
        plddt_vals = extract_plddt(struct_df)
        log.info(
            "Parsed structure: %d residues (%s), seq=%s…",
            len(plddt_vals),
            parser_used,
            struct_seq[:20],
        )
        # Map to first FASTA sequence (or by matching length)
        plddt_map["__structure__"] = plddt_vals
        plddt_map["__struct_seq__"] = struct_seq  # type: ignore[assignment]

    # Read FASTA
    records = read_fasta(args.fasta)
    if not records:
        log.error("No sequences found in %s", args.fasta)
        return 1
    log.info("Read %d sequence(s) from %s", len(records), args.fasta)

    # Run predictions
    all_results: List[pd.DataFrame] = []
    needs_struct = _needs_structure(loaded)

    for seq_id, sequence in records:
        plddt_vals = None
        if needs_struct and plddt_map:
            plddt_vals = plddt_map.get("__structure__")
            # Only use pLDDT if sequence lengths match
            if plddt_vals is not None and len(plddt_vals) != len(sequence):
                log.warning(
                    "pLDDT length mismatch for %s: struct=%d, seq=%d. "
                    "Using default pLDDT=50.",
                    seq_id,
                    len(plddt_vals),
                    len(sequence),
                )
                plddt_vals = None

        t1 = time.perf_counter()
        result = _run_single(seq_id, sequence, loaded, plddt_vals, args.threshold)
        dt = time.perf_counter() - t1
        mean_p = result["p"].mean()
        log.info(
            "  %s  len=%d  mean_p=%.4f  (%.3fs)",
            seq_id,
            len(sequence),
            mean_p,
            dt,
        )
        all_results.append(result)

    # Combine and write
    combined = pd.concat(all_results, ignore_index=True)
    writer = WRITERS[args.format]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer(combined, args.out)
    log.info("Wrote %d rows to %s (%s)", len(combined), args.out, args.format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
