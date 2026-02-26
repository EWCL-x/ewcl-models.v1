"""Derived diagnostic scores for EWCL model analysis.

These are **post-hoc analysis tools**, not part of any trained model.
They quantify agreement/disagreement between EWCL predictions and
external confidence measures (e.g., AlphaFold pLDDT).

Modules
-------
compute_edi
    EWCL–pLDDT Disagreement Index.  Quantifies local discordance between
    EWCL-Structure disorder predictions and AlphaFold per-residue confidence.

compute_cds
    EWCL–Confidence Disagreement Score.  Protein-level summary of the EDI,
    useful for ranking proteins by the degree of EWCL/AlphaFold disagreement.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


def compute_edi(
    ewcl_proba: np.ndarray,
    plddt: Union[np.ndarray, List[float]],
) -> np.ndarray:
    """EWCL–pLDDT Disagreement Index (EDI).

    Per-residue score quantifying local discordance between EWCL disorder
    predictions and AlphaFold pLDDT confidence.

    .. math::

        \\text{EDI}(i) = p_{\\text{EWCL}}(i) - \\bigl(1 - \\tfrac{\\text{pLDDT}(i)}{100}\\bigr)

    Interpretation
    ~~~~~~~~~~~~~~
    - **EDI > 0** — EWCL predicts *more* disorder than pLDDT expects.
      Potential dynamic region that AlphaFold modelled as structured.
    - **EDI < 0** — AlphaFold is *less* confident than EWCL's disorder
      prediction.  Potential structured region with poor AF2 model quality.
    - **EDI ≈ 0** — Agreement between EWCL and AlphaFold confidence.

    Parameters
    ----------
    ewcl_proba : array-like, shape (n,)
        Per-residue disorder probabilities from EWCL-Structure (0–1).
    plddt : array-like, shape (n,)
        Per-residue pLDDT values from AlphaFold (0–100).

    Returns
    -------
    edi : np.ndarray, shape (n,)
        Per-residue disagreement index.
    """
    p = np.asarray(ewcl_proba, dtype=np.float64)
    c = np.asarray(plddt, dtype=np.float64)
    if p.shape != c.shape:
        raise ValueError(
            f"Shape mismatch: ewcl_proba={p.shape}, plddt={c.shape}"
        )
    # pLDDT → expected disorder: (1 - pLDDT/100)
    expected_disorder = 1.0 - c / 100.0
    return p - expected_disorder


def compute_cds(
    ewcl_proba: np.ndarray,
    plddt: Union[np.ndarray, List[float]],
    method: str = "mean_abs",
) -> float:
    """EWCL–Confidence Disagreement Score (CDS).

    Protein-level summary of the EDI, useful for ranking proteins by
    the degree of EWCL / AlphaFold disagreement.

    Parameters
    ----------
    ewcl_proba : array-like, shape (n,)
        Per-residue disorder probabilities from EWCL-Structure.
    plddt : array-like, shape (n,)
        Per-residue pLDDT values from AlphaFold (0–100).
    method : str
        Aggregation method.

        - ``"mean_abs"`` (default): mean |EDI|
        - ``"mean_signed"``: mean EDI (preserves direction)
        - ``"max_abs"``: max |EDI|
        - ``"rms"``: root-mean-square EDI

    Returns
    -------
    cds : float
        Protein-level disagreement score.
    """
    edi = compute_edi(ewcl_proba, plddt)
    if method == "mean_abs":
        return float(np.mean(np.abs(edi)))
    if method == "mean_signed":
        return float(np.mean(edi))
    if method == "max_abs":
        return float(np.max(np.abs(edi)))
    if method == "rms":
        return float(np.sqrt(np.mean(edi**2)))
    raise ValueError(f"Unknown CDS method: {method!r}")


def edi_segments(
    edi: np.ndarray,
    threshold: float = 0.2,
    min_length: int = 3,
) -> list[dict]:
    """Identify contiguous segments of high EDI.

    Parameters
    ----------
    edi : np.ndarray, shape (n,)
        Per-residue EDI values (from :func:`compute_edi`).
    threshold : float
        Absolute EDI threshold for a residue to be considered "discordant".
    min_length : int
        Minimum segment length to report.

    Returns
    -------
    segments : list[dict]
        Each dict has keys ``start``, ``end`` (0-based, inclusive),
        ``mean_edi``, ``direction`` ("ewcl_higher" or "plddt_higher").
    """
    n = len(edi)
    above = np.abs(edi) >= threshold
    segments: list[dict] = []
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= min_length:
                seg_edi = edi[i:j]
                mean_edi = float(np.mean(seg_edi))
                segments.append(
                    {
                        "start": i,
                        "end": j - 1,
                        "length": j - i,
                        "mean_edi": mean_edi,
                        "direction": "ewcl_higher"
                        if mean_edi > 0
                        else "plddt_higher",
                    }
                )
            i = j
        else:
            i += 1
    return segments
