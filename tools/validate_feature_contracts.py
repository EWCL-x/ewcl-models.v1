#!/usr/bin/env python3
"""Validate that an extractor implementation satisfies the EWCL feature contract.

This is the key adopter-facing sanity check:
- Loads the publication models.
- Loads the immutable feature-order contracts (CSV).
- Builds features from the *frozen publication extractor* (the one embedded in
  caid_dual_benchmark.py via _build_ewclv1_features).
- Asserts:
    1) every required feature exists
    2) ordering matches exactly
    3) predictions run without shape issues

If any of these fail, the user's integration is wrong.

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import caid_dual_benchmark as cdb  # noqa: E402


TEST_SEQS = {
    "P00441_SOD1": "ATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGLHGFHVHEFGDNTAGCTSAGPHFNPLSRKHGGPKDEERHVGDLGNVTADKDGVADVSIEDSVISLSGDHCIIGRTLVVHEKADDLGKGGNEESTKTGNAGSRLACGVIGIAQ",
    "P04637_p53": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
}


def _extract_bfactor_per_residue(pdb_path: Path, *, expected_len: int) -> list[float] | None:
    """Extract per-residue B-factor signal from a PDB file.

    - For AlphaFold PDBs, B-factor encodes pLDDT (0-100).
    - For crystallographic/X-ray PDBs, B-factor is the (often 10-80+) thermal
      factor.

    For EWCL-Structure we only need a *numeric per-residue vector*. If the file
    is AlphaFold, this corresponds to true pLDDT. If it's X-ray, it's still a
    meaningful confidence-like signal (different semantics), and the rest of the
    structural features (hydropathy/charge/entropy/curvature) are computed from
    sequence anyway.

    Returns list[float] of length expected_len when possible. If the file is
    missing/unparseable or too mismatched, returns None.
    """
    if not pdb_path.exists():
        return None

    # residue index (1-based in PDB) -> list of bfactors
    per_res: dict[int, list[float]] = {}
    try:
        with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                # PDB fixed-width fields
                if len(line) < 66:
                    continue
                try:
                    resseq = int(line[22:26].strip())
                    bfac = float(line[60:66].strip())
                except Exception:
                    continue

                per_res.setdefault(resseq, []).append(bfac)
    except Exception:
        return None

    if not per_res:
        return None

    # Build a vector length expected_len. We don't attempt a complex alignment
    # here; we assume residue numbering roughly follows 1..N.
    # If residues are missing, fill with the global mean.
    all_vals = [v for vs in per_res.values() for v in vs]
    if not all_vals:
        return None
    global_mean = float(np.mean(all_vals))

    out: list[float] = []
    for i in range(1, expected_len + 1):
        vals = per_res.get(i)
        v = float(np.mean(vals)) if vals else global_mean
        # Keep inside [0,100] to match pLDDT scale used by training.
        # (X-ray B-factors are usually within this range anyway.)
        if v < 0.0:
            v = 0.0
        elif v > 100.0:
            v = 100.0
        out.append(v)

    # Heuristic: accept only AlphaFold-style pLDDT-in-Bfactor.
    # For X-ray structures, B-factors have inverted semantics vs pLDDT and a
    # different distribution. For robustness we treat those as "no pLDDT" so the
    # caller can fall back to sequence-only behavior.
    mn = float(np.min(out))
    mx = float(np.max(out))
    mean = float(np.mean(out))
    looks_like_plddt = mn >= 0.0 and mx <= 100.0 and mean >= 50.0
    if not looks_like_plddt:
        return None

    return out


def read_feature_csv(path: Path) -> list[str]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        feats = [row["feature"] for row in r]
    return feats


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", type=Path, default=Path("contracts/features"))
    ap.add_argument(
        "--pdb-dir",
        type=Path,
        default=Path("."),
        help="Directory to look for <uniprot>.pdb files (B-factor assumed pLDDT).",
    )
    ap.add_argument(
        "--allow-structure-fallback",
        action="store_true",
        help="Don't fail if structure features can't be computed; validate Sequence/Disorder only.",
    )
    args = ap.parse_args()

    models = cdb.load_local_models()

    contract_files = {
        "EWCL-Sequence": args.contracts / "EWCL-Sequence.feature_list.csv",
        "EWCL-Disorder": args.contracts / "EWCL-Disorder.feature_list.csv",
        "EWCL-Structure": args.contracts / "EWCL-Structure.feature_list.csv",
    }

    required = {
        "EWCL-Sequence": read_feature_csv(contract_files["EWCL-Sequence"]),
        "EWCL-Disorder": read_feature_csv(contract_files["EWCL-Disorder"]),
        "EWCL-Structure": read_feature_csv(contract_files["EWCL-Structure"]),
    }

    # 1) Verify model-declared feature list matches contract exactly
    model_lists = {
        "EWCL-Sequence": list(models.seq_feats),
        "EWCL-Disorder": list(models.dis_feats),
        "EWCL-Structure": list(models.str_feats),
    }

    for k in required:
        if required[k] != model_lists[k]:
            raise SystemExit(
                f"Contract mismatch for {k}: contract({len(required[k])}) != model({len(model_lists[k])})"
            )

    # 2) Build frozen-publication features and ensure all required features exist
    for name, seq in TEST_SEQS.items():
        # Optional: try to load pLDDT for EWCL-Structure validation.
        # We accept either '<id>.pdb' or 'AF-<id>-F1-model_v*.pdb' patterns.
        uniprot = name.split("_")[0]
        candidates = [
            args.pdb_dir / f"{uniprot}.pdb",
        ]
        # A few common AF naming patterns found in this repo
        candidates += sorted(args.pdb_dir.glob(f"AF-{uniprot}-F1-model_v*.pdb"))

        plddt_vals = None
        for c in candidates:
            plddt_vals = _extract_bfactor_per_residue(c, expected_len=len(seq))
            if plddt_vals is not None:
                break

        fb = cdb._build_ewclv1_features(seq, pssm=None)
        base_df = fb.base_df
        all_df = fb.all_df

        for model_name, feats in required.items():
            if model_name == "EWCL-Sequence":
                df = all_df
            elif model_name == "EWCL-Disorder":
                df = base_df
            else:
                # EWCL-Structure needs the 6 structural features.
                if plddt_vals is None:
                    if args.allow_structure_fallback:
                        continue
                    raise SystemExit(
                        f"No structure/pLDDT available for {name}. "
                        f"Provide a matching PDB in --pdb-dir or pass --allow-structure-fallback."
                    )
                struct_df = cdb._compute_structure_features(seq, plddt_vals=plddt_vals)
                df = cdb.pd.concat(
                    [base_df.reset_index(drop=True), struct_df.reset_index(drop=True)],
                    axis=1,
                )
            missing = [f for f in feats if f not in df.columns]
            if missing:
                raise SystemExit(f"Missing {len(missing)} features for {model_name} on {name}: {missing[:10]}")

            X = df[feats].values
            if X.shape[1] != len(feats):
                raise SystemExit(f"Bad feature matrix shape for {model_name}: {X.shape}")

        # 3) Run predictions (feature-order enforced by contract)
        preds = models.predict_all(seq, plddt_vals=plddt_vals)
        for m in cdb.MODEL_NAMES:
            arr = np.asarray(preds[m], dtype=float)
            if arr.shape[0] != len(seq):
                raise SystemExit(f"Prediction length mismatch {m} on {name}: {arr.shape[0]} != {len(seq)}")

        # Print stable checksums so regressions are visible
        dis = np.asarray(preds["EWCL-Disorder"], dtype=float)
        seqp = np.asarray(preds["EWCL-Sequence"], dtype=float)
        print(
            f"{name}: len={len(seq)} "
            f"EWCL-Disorder.mean={dis.mean():.6f} sha={sha256_text(','.join(f'{x:.6f}' for x in dis[:50]))[:16]} "
            f"EWCL-Sequence.mean={seqp.mean():.6f} sha={sha256_text(','.join(f'{x:.6f}' for x in seqp[:50]))[:16]}"
        )

    print("OK: feature contracts + frozen extractor validated")


if __name__ == "__main__":
    main()
