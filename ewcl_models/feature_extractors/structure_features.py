"""Frozen structure feature extractor for EWCL-Structure.

Produces the 6 structure-derived features appended to the 224 base sequence
features for EWCL-Structure inference:

    plddt, struct_curvature, struct_hydropathy, struct_charge,
    struct_hydro_entropy, struct_charge_entropy

This file is a frozen copy of the feature extraction pipeline used to train
the EWCL-Structure model.  **Do not modify** — any change will invalidate
the trained weights.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Property tables (structure-specific) ──────────────────────────────────────

HYDROPATHY: Dict[str, float] = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3, "X": 0.0,
}

CHARGE_PH7: Dict[str, float] = {
    "A": 0.0, "C": 0.0, "D": -1.0, "E": -1.0, "F": 0.0, "G": 0.0,
    "H": 0.0, "I": 0.0, "K": 1.0, "L": 0.0, "M": 0.0, "N": 0.0,
    "P": 0.0, "Q": 0.0, "R": 1.0, "S": 0.0, "T": 0.0, "V": 0.0,
    "W": 0.0, "Y": 0.0, "X": 0.0,
}

AA3_TO_1: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K", "HYP": "P",
    "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _entropy_base2(probs: np.ndarray) -> float:
    """Shannon entropy in bits."""
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return -float(np.sum(p * np.log2(p)))


def compute_structure_features(
    sequence: str,
    plddt_vals: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Build the 6 structure features for EWCL-Structure.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (one-letter codes).
    plddt_vals : list[float] or None
        Per-residue pLDDT from AlphaFold (0–100).
        If ``None``, defaults to 50.0 for every residue.

    Returns
    -------
    pd.DataFrame
        Columns: ``plddt``, ``struct_curvature``, ``struct_hydropathy``,
        ``struct_charge``, ``struct_hydro_entropy``, ``struct_charge_entropy``.
    """
    n = len(sequence)
    hydro = np.array([HYDROPATHY.get(aa, 0.0) for aa in sequence])
    charge = np.array([CHARGE_PH7.get(aa, 0.0) for aa in sequence])
    w = 5
    hw = w // 2  # 2

    # Rolling mean via cumsum
    def _rolling_mean(arr: np.ndarray) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(arr)])
        out = np.empty(n)
        for i in range(n):
            s = max(0, i - hw)
            e = min(n, i + hw + 1)
            out[i] = (cs[e] - cs[s]) / (e - s)
        return out

    sh = _rolling_mean(hydro)
    sc = _rolling_mean(charge)

    # Curvature = |2nd derivative of hydropathy|
    scurv = np.zeros(n)
    if n > 2:
        scurv[1:-1] = np.abs(hydro[:-2] - 2.0 * hydro[1:-1] + hydro[2:])

    # Rolling entropy (5-bin histogram per window)
    def _rolling_entropy(arr: np.ndarray) -> np.ndarray:
        out = np.zeros(n)
        amin, amax = arr.min(), arr.max()
        if amax == amin:
            return out
        rng = amax - amin + 1e-10
        for i in range(n):
            s = max(0, i - hw)
            e = min(n, i + hw + 1)
            win = arr[s:e]
            if len(win) <= 1:
                continue
            counts = np.zeros(5)
            for v in win:
                idx = int((v - amin) / rng * 5)
                if idx >= 5:
                    idx = 4
                counts[idx] += 1
            total = counts.sum()
            if total > 0:
                p = counts / total
                out[i] = _entropy_base2(p)
        return out

    she = _rolling_entropy(hydro)
    sce = _rolling_entropy(charge)

    # pLDDT
    if plddt_vals is not None and len(plddt_vals) == n:
        plddt_arr = np.array(
            [v if v is not None else 50.0 for v in plddt_vals], dtype=float
        )
    else:
        plddt_arr = np.full(n, 50.0)

    return pd.DataFrame(
        {
            "plddt": plddt_arr,
            "struct_curvature": scurv,
            "struct_hydropathy": sh,
            "struct_charge": sc,
            "struct_hydro_entropy": she,
            "struct_charge_entropy": sce,
        }
    )
