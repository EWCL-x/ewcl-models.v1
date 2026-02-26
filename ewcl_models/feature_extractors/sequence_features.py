"""Frozen sequence feature extractor for EWCL publication models.

Produces the feature matrices consumed by EWCL-Sequence (249 features),
EWCL-Disorder (239 features), and the sequence-derived portion of
EWCL-Structure (224 base features).

This file is a frozen copy of the feature extraction pipeline used to train
the publication models.  **Do not modify** — any change will invalidate the
trained weights.  The canonical feature order for each model is stored in its
``feature_list.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Canonical amino-acid alphabet ─────────────────────────────────────────────

AA: List[str] = list("ARNDCQEGHILKMFPSTWYV")
AA_SET = set(AA)
AA_TO_IDX = {a: i for i, a in enumerate(AA)}

# ── Physicochemical property tables ───────────────────────────────────────────
# Kyte-Doolittle hydropathy
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
POLARITY = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5,
    "Q": 10.5, "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2,
    "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0,
    "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}
VDW = {
    "A": 67.0, "R": 148.0, "N": 96.0, "D": 91.0, "C": 86.0,
    "Q": 114.0, "E": 109.0, "G": 48.0, "H": 118.0, "I": 124.0,
    "L": 124.0, "K": 135.0, "M": 124.0, "F": 135.0, "P": 90.0,
    "S": 73.0, "T": 93.0, "W": 163.0, "Y": 141.0, "V": 105.0,
}
FLEX = {
    "A": 0.360, "R": 0.530, "N": 0.460, "D": 0.510, "C": 0.350,
    "Q": 0.490, "E": 0.500, "G": 0.540, "H": 0.320, "I": 0.460,
    "L": 0.400, "K": 0.470, "M": 0.300, "F": 0.310, "P": 0.510,
    "S": 0.510, "T": 0.440, "W": 0.310, "Y": 0.420, "V": 0.390,
}
BULK = {
    "A": 11.50, "R": 14.28, "N": 12.82, "D": 11.68, "C": 13.46,
    "Q": 14.45, "E": 13.57, "G": 3.40, "H": 13.69, "I": 21.40,
    "L": 21.40, "K": 15.71, "M": 16.25, "F": 19.80, "P": 17.43,
    "S": 9.47, "T": 15.77, "W": 21.67, "Y": 18.03, "V": 21.57,
}
HELIX = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
SHEET = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}
CHARGE7 = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

WINDOWS = [5, 11, 25, 50, 100]
AROMATIC = set("FWY")
DISORDER_PROMOTING = set("AEQSGPDRK")
ORDER_PROMOTING = set("CFILMVWY")


# ── Vectorised helpers ────────────────────────────────────────────────────────

def _seq_to_vec(seq: str, table: Dict[str, float]) -> np.ndarray:
    return np.array([table.get(a, 0.0) for a in seq], dtype=float)


def _rolling_stats(x: np.ndarray, w: int) -> Dict[str, np.ndarray]:
    """Rolling mean/std/min/max using cumsum + stride tricks."""
    n = len(x)
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="edge")

    cs = np.cumsum(xpad)
    cs = np.insert(cs, 0, 0.0)
    out_mean = (cs[w : w + n] - cs[:n]) / w

    cs2 = np.cumsum(xpad**2)
    cs2 = np.insert(cs2, 0, 0.0)
    mean_sq = (cs2[w : w + n] - cs2[:n]) / w
    out_std = np.sqrt(np.maximum(mean_sq - out_mean**2, 0.0))

    shape = (n, w)
    strides = (xpad.strides[0], xpad.strides[0])
    windows = np.lib.stride_tricks.as_strided(xpad, shape=shape, strides=strides)
    out_min = windows.min(axis=1)
    out_max = windows.max(axis=1)

    return {"mean": out_mean, "std": out_std, "min": out_min, "max": out_max}


def _window_entropy(seq: str, w: int) -> np.ndarray:
    """Shannon entropy of AA distribution in window."""
    n = len(seq)
    pad = w // 2
    spad = seq[0] * pad + seq + seq[-1] * pad
    npad = len(spad)
    aa_idx = {a: i for i, a in enumerate(AA)}
    onehot = np.zeros((npad, 20), dtype=float)
    for j, c in enumerate(spad):
        k = aa_idx.get(c, -1)
        if k >= 0:
            onehot[j, k] = 1.0
    cs = np.cumsum(onehot, axis=0)
    cs = np.vstack([np.zeros((1, 20)), cs])
    counts = cs[w : w + n] - cs[:n]
    p = counts / w
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    return -(p * logp).sum(axis=1)


def _low_complex_from_entropy(H: np.ndarray, thresh: float = 1.5) -> np.ndarray:
    return (H < thresh).astype(float)


def _composition_bias(seq: str, w: int) -> np.ndarray:
    """(max_fraction - uniform_fraction) per window."""
    n = len(seq)
    pad = w // 2
    spad = seq[0] * pad + seq + seq[-1] * pad
    npad = len(spad)
    aa_idx = {a: i for i, a in enumerate(AA)}
    onehot = np.zeros((npad, 20), dtype=float)
    for j, c in enumerate(spad):
        k = aa_idx.get(c, -1)
        if k >= 0:
            onehot[j, k] = 1.0
    cs = np.cumsum(onehot, axis=0)
    cs = np.vstack([np.zeros((1, 20)), cs])
    counts = cs[w : w + n] - cs[:n]
    frac = counts / w
    return frac.max(axis=1) - (1.0 / 20.0)


def _uversky_distance(
    hydro_mean: np.ndarray, charge_mean: np.ndarray
) -> np.ndarray:
    return hydro_mean - (1.151 * np.abs(charge_mean) + 0.693)


def _poly_run_flags(seq: str, aa: str, run_len: int = 3) -> np.ndarray:
    out = np.zeros(len(seq))
    cur = 0
    for i, c in enumerate(seq):
        cur = cur + 1 if c == aa else 0
        if cur >= run_len:
            out[i] = 1.0
    return out


def _scd_local(charge_vec: np.ndarray, w: int = 25) -> np.ndarray:
    """Sequence Charge Decoration — vectorised per-window."""
    n = len(charge_vec)
    pad = w // 2
    xpad = np.pad(charge_vec, (pad, pad), mode="edge")
    dist_sqrt = np.sqrt(np.arange(1, w, dtype=float))
    out = np.zeros(n)
    denom = (w * (w - 1)) / 2.0 if w > 1 else 1.0
    for i in range(n):
        win = xpad[i : i + w]
        s = 0.0
        for d in range(1, w):
            s += float(np.dot(win[: w - d], win[d:])) * dist_sqrt[d - 1]
        out[i] = s / denom
    return out


def _max_run_global(seq: str) -> int:
    """Longest homopolymer run in entire sequence."""
    if len(seq) == 0:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def _aro_spacing_cv(seq: str) -> float:
    """CV of spacing between aromatic residues."""
    positions = [i for i, c in enumerate(seq) if c in AROMATIC]
    if len(positions) < 2:
        return 0.0
    spacings = np.diff(positions).astype(float)
    m = spacings.mean()
    if m < 1e-12:
        return 0.0
    return float(spacings.std() / m)


# ── FeatureBlock dataclass ────────────────────────────────────────────────────

@dataclass
class FeatureBlock:
    """Container for the three feature DataFrames.

    Attributes
    ----------
    base_df : pd.DataFrame
        224 base sequence features (shared by Disorder and Structure).
    pssm_df : pd.DataFrame
        24 PSSM-derived features + has_pssm_data.
    all_df : pd.DataFrame
        249 features = base_df + pssm_df + is_unknown_aa (used by Sequence model).
    has_pssm : bool
        Whether real PSSM data was supplied.
    """

    base_df: pd.DataFrame
    pssm_df: pd.DataFrame
    all_df: pd.DataFrame
    has_pssm: bool


# ── Main entry point ─────────────────────────────────────────────────────────

def build_sequence_features(
    seq: str,
    pssm: Optional[pd.DataFrame] = None,
) -> FeatureBlock:
    """Build the frozen feature matrix for EWCL publication models.

    Parameters
    ----------
    seq : str
        Amino acid sequence (one-letter codes).
    pssm : pd.DataFrame or None
        PSSM matrix with columns ``A, R, N, D, …, V``.  If ``None``,
        PSSM features are zero-filled and ``has_pssm_data = 0``.

    Returns
    -------
    FeatureBlock
        Contains ``base_df`` (224 cols), ``pssm_df`` (25 cols),
        ``all_df`` (249 cols), and ``has_pssm`` flag.
    """
    seq = seq.strip().upper()
    n = len(seq)
    idx = np.arange(1, n + 1)

    # ── Per-residue physicochemical vectors ───────────────────────────────
    hyd = _seq_to_vec(seq, KD)
    pol = _seq_to_vec(seq, POLARITY)
    vdw = _seq_to_vec(seq, VDW)
    flx = _seq_to_vec(seq, FLEX)
    blk = _seq_to_vec(seq, BULK)
    hlx = _seq_to_vec(seq, HELIX)
    bta = _seq_to_vec(seq, SHEET)
    chg = _seq_to_vec(seq, CHARGE7)

    cols: Dict[str, np.ndarray] = {}

    # ── Windowed rolling statistics ───────────────────────────────────────
    def emit_track(prefix: str, x: np.ndarray) -> None:
        for w in WINDOWS:
            stats = _rolling_stats(x, w)
            cols[f"{prefix}_w{w}_mean"] = stats["mean"]
            cols[f"{prefix}_w{w}_std"] = stats["std"]
            cols[f"{prefix}_w{w}_min"] = stats["min"]
            cols[f"{prefix}_w{w}_max"] = stats["max"]

    emit_track("hydro", hyd)
    emit_track("polar", pol)
    emit_track("vdw", vdw)
    emit_track("flex", flx)
    emit_track("bulk", blk)
    emit_track("helix_prop", hlx)
    emit_track("sheet_prop", bta)
    emit_track("charge", chg)

    # ── Base single-residue features ──────────────────────────────────────
    cols["hydropathy"] = hyd
    cols["polarity"] = pol
    cols["vdw_volume"] = vdw
    cols["flexibility"] = flx
    cols["bulkiness"] = blk
    cols["helix_prop"] = hlx
    cols["sheet_prop"] = bta
    cols["charge_pH7"] = chg

    # ── Windowed entropy / low complexity / composition bias / Uversky ────
    for w in WINDOWS:
        Hw = _window_entropy(seq, w)
        cols[f"entropy_w{w}"] = Hw
        cols[f"low_complex_w{w}"] = _low_complex_from_entropy(Hw)
        cols[f"comp_bias_w{w}"] = _composition_bias(seq, w)
        cols[f"uversky_dist_w{w}"] = _uversky_distance(
            cols[f"hydro_w{w}_mean"], cols[f"charge_w{w}_mean"]
        )

    # ── Global composition fractions ──────────────────────────────────────
    counts = np.array([seq.count(a) for a in AA], dtype=float)
    frac = counts / max(1, counts.sum())
    for i, a in enumerate(AA):
        cols[f"comp_{a}"] = np.full(n, frac[i])
    cols["comp_frac_aromatic"] = np.full(
        n, frac[AA.index("F")] + frac[AA.index("W")] + frac[AA.index("Y")]
    )
    cols["comp_frac_positive"] = np.full(
        n, frac[AA.index("K")] + frac[AA.index("R")] + frac[AA.index("H")]
    )
    cols["comp_frac_negative"] = np.full(
        n, frac[AA.index("D")] + frac[AA.index("E")]
    )
    cols["comp_frac_polar"] = np.full(
        n,
        sum(
            frac[AA.index(x)]
            for x in ["S", "T", "N", "Q", "C", "Y", "H", "K", "R", "D", "E"]
        ),
    )
    cols["comp_frac_aliphatic"] = np.full(
        n,
        frac[AA.index("A")]
        + frac[AA.index("V")]
        + frac[AA.index("L")]
        + frac[AA.index("I")],
    )
    cols["comp_frac_proline"] = np.full(n, frac[AA.index("P")])
    cols["comp_frac_glycine"] = np.full(n, frac[AA.index("G")])

    # ── Position features ─────────────────────────────────────────────────
    pos_rel = np.arange(n, dtype=float) / max(1, n - 1)
    pos_dist_nterm = np.arange(n, dtype=float)
    pos_dist_cterm = np.arange(n - 1, -1, -1, dtype=float)
    pos_dist_terminus = np.minimum(pos_dist_nterm, pos_dist_cterm)

    cols["pos_rel"] = pos_rel
    cols["pos_norm"] = pos_rel
    cols["pos_dist_terminus_norm"] = pos_dist_terminus / max(1, n - 1)
    cols["pos_log_dist_terminus"] = np.log1p(pos_dist_terminus)
    cols["pos_is_nterm"] = (pos_dist_nterm < max(1, n * 0.05)).astype(float)
    cols["pos_is_cterm"] = (pos_dist_cterm < max(1, n * 0.05)).astype(float)
    cols["pos_dist_nterm_norm"] = pos_dist_nterm / max(1, n - 1)
    cols["pos_dist_cterm_norm"] = pos_dist_cterm / max(1, n - 1)
    cols["pos_dist_n"] = cols["pos_dist_nterm_norm"]
    cols["pos_dist_c"] = cols["pos_dist_cterm_norm"]
    for pct in [10, 20, 30]:
        cols[f"pos_is_n{pct}"] = (
            pos_dist_nterm < max(1, n * pct / 100)
        ).astype(float)
        cols[f"pos_is_c{pct}"] = (
            pos_dist_cterm < max(1, n * pct / 100)
        ).astype(float)
    for tau in [10, 30]:
        cols[f"pos_nexp{tau}"] = np.exp(-pos_dist_nterm / max(1, tau))
        cols[f"pos_cexp{tau}"] = np.exp(-pos_dist_cterm / max(1, tau))

    # ── Sequence-level features ───────────────────────────────────────────
    frac_pos_charge = (
        frac[AA.index("K")] + frac[AA.index("R")] + frac[AA.index("H")]
    )
    frac_neg_charge = frac[AA.index("D")] + frac[AA.index("E")]
    cols["seq_frac_charged"] = np.full(n, frac_pos_charge + frac_neg_charge)
    cols["seq_ncpr"] = np.full(n, frac_pos_charge - frac_neg_charge)

    # ── Protein-level features ────────────────────────────────────────────
    cols["prot_length_log"] = np.full(n, math.log(n) if n > 0 else 0.0)
    cols["prot_log_len"] = cols["prot_length_log"]
    cols["prot_inv_len"] = np.full(n, 1.0 / max(1, n))
    cols["prot_mean_H"] = np.full(n, float(hyd.mean()))
    prot_uversky = float(hyd.mean()) - 1.151 * abs(float(chg.mean())) - 0.693
    cols["prot_uversky_axis"] = np.full(n, prot_uversky)
    dp_count = sum(1 for c in seq if c in DISORDER_PROMOTING)
    op_count = sum(1 for c in seq if c in ORDER_PROMOTING)
    cols["prot_dp_frac"] = np.full(n, dp_count / max(1, n))
    cols["prot_op_frac"] = np.full(n, op_count / max(1, n))
    aro_count = sum(1 for c in seq if c in AROMATIC)
    cols["prot_aro_frac"] = np.full(n, aro_count / max(1, n))

    # Kappa at protein level
    if n >= 10:
        overall_var = float(np.var(chg))
        if overall_var > 1e-12:
            bsize = 5
            nblocks = n // bsize
            if nblocks >= 2:
                block_means = [
                    float(chg[i * bsize : (i + 1) * bsize].mean())
                    for i in range(nblocks)
                ]
                prot_kappa = float(np.var(block_means) / overall_var)
            else:
                prot_kappa = 0.0
        else:
            prot_kappa = 0.0
    else:
        prot_kappa = 0.0
    cols["prot_kappa"] = np.full(n, prot_kappa)

    # ── LCD features (low complexity domain) ──────────────────────────────
    _aa_to_int = {a: i for i, a in enumerate(sorted(set(seq)))}
    seq_int = np.array([_aa_to_int[c] for c in seq], dtype=np.int32)
    n_unique = len(_aa_to_int)

    for w in [5, 11, 25]:
        tag = f"w{w}"
        pad_lcd = w // 2
        spad_int = np.pad(seq_int, (pad_lcd, pad_lcd), mode="edge")
        shape_lcd = (n, w)
        strides_lcd = (spad_int.strides[0], spad_int.strides[0])
        wins_int = np.lib.stride_tricks.as_strided(
            spad_int, shape=shape_lcd, strides=strides_lcd
        )

        # top1 dominance via cumsum one-hot
        onehot_lcd = np.zeros((n + 2 * pad_lcd, n_unique), dtype=np.float64)
        for k, v in _aa_to_int.items():
            onehot_lcd[spad_int == v, v] = 1.0
        cs_lcd = np.vstack(
            [np.zeros((1, n_unique)), np.cumsum(onehot_lcd, axis=0)]
        )
        win_counts = cs_lcd[w : w + n] - cs_lcd[:n]
        cols[f"lcd_top1_dom_{tag}"] = win_counts.max(axis=1) / w

        # Wootton-Federhen complexity
        freqs = win_counts / w
        log_terms = np.where(freqs > 0, freqs * np.log(freqs), 0.0)
        cols[f"lcd_wf_complex_{tag}"] = np.exp(log_terms.sum(axis=1))

        # Lempel-Ziv proxy
        spad_str = seq[0] * pad_lcd + seq + seq[-1] * pad_lcd
        max_possible = min(400, max(1, w - 1))
        lz_arr = np.zeros(n)
        for i in range(n):
            ws = spad_str[i : i + w]
            bigrams = set()
            for j in range(len(ws) - 1):
                bigrams.add(ws[j : j + 2])
            lz_arr[i] = len(bigrams) / max_possible
        cols[f"lcd_lz_proxy_{tag}"] = lz_arr

        # Longest homopolymer run in window
        same = wins_int[:, 1:] == wins_int[:, :-1]
        longest_arr = np.ones(n, dtype=np.float64)
        cur_run = np.ones(n, dtype=np.float64)
        for j in range(w - 1):
            cur_run = np.where(same[:, j], cur_run + 1, 1.0)
            longest_arr = np.maximum(longest_arr, cur_run)
        cols[f"lcd_longest_run_{tag}"] = longest_arr

    # Mean run and frac_runs_ge4 for w11, w25
    for w in [11, 25]:
        tag = f"w{w}"
        pad_lcd = w // 2
        spad_int2 = np.pad(seq_int, (pad_lcd, pad_lcd), mode="edge")
        shape_lcd2 = (n, w)
        strides_lcd2 = (spad_int2.strides[0], spad_int2.strides[0])
        wins_int2 = np.lib.stride_tricks.as_strided(
            spad_int2, shape=shape_lcd2, strides=strides_lcd2
        )
        same2 = wins_int2[:, 1:] == wins_int2[:, :-1]
        cur_run2 = np.ones(n, dtype=np.float64)
        run_count = np.ones(n, dtype=np.float64)
        total_len = np.zeros(n, dtype=np.float64)
        ge4_total = np.zeros(n, dtype=np.float64)
        for j in range(w - 1):
            continuing = same2[:, j]
            ended = ~continuing
            total_len += np.where(ended, cur_run2, 0.0)
            ge4_total += np.where(ended & (cur_run2 >= 4), cur_run2, 0.0)
            run_count += np.where(ended, 1.0, 0.0)
            cur_run2 = np.where(continuing, cur_run2 + 1, 1.0)
        total_len += cur_run2
        ge4_total += np.where(cur_run2 >= 4, cur_run2, 0.0)
        cols[f"lcd_mean_run_{tag}"] = total_len / run_count
        cols[f"lcd_frac_runs_ge4_{tag}"] = ge4_total / w

    # Global max run
    cols["lcd_max_run_global"] = np.full(n, float(_max_run_global(seq)))

    # Aromatic density for w11, w25
    aro_mask = np.array([1.0 if c in AROMATIC else 0.0 for c in seq])
    for w in [11, 25]:
        pad_a = w // 2
        apad = np.pad(aro_mask, (pad_a, pad_a), mode="edge")
        cs_a = np.insert(np.cumsum(apad), 0, 0.0)
        cols[f"lcd_aro_density_w{w}"] = (cs_a[w : w + n] - cs_a[:n]) / w

    # Aromatic spacing CV (global)
    cols["lcd_aro_spacing_cv"] = np.full(n, _aro_spacing_cv(seq))

    # Repeat score for w11, w25
    for w in [11, 25]:
        pad_r = w // 2
        spad_r = np.pad(seq_int, (pad_r, pad_r), mode="edge")
        shape_r = (n, w)
        strides_r = (spad_r.strides[0], spad_r.strides[0])
        wins_r = np.lib.stride_tricks.as_strided(
            spad_r, shape=shape_r, strides=strides_r
        )
        if w >= 3:
            matches_r = (wins_r[:, :-2] == wins_r[:, 2:]).sum(axis=1).astype(
                float
            )
            cols[f"lcd_repeat_score_w{w}"] = matches_r / (w - 2)
        else:
            cols[f"lcd_repeat_score_w{w}"] = np.zeros(n)

    # ── Pattern features ──────────────────────────────────────────────────
    # Hydropathy variance & MAD for w11, w25, w50
    for w in [11, 25, 50]:
        tag = f"w{w}"
        pad_h = w // 2
        hpad = np.pad(hyd, (pad_h, pad_h), mode="edge")
        cs1 = np.insert(np.cumsum(hpad), 0, 0.0)
        cs2 = np.insert(np.cumsum(hpad**2), 0, 0.0)
        rmean = (cs1[w : w + n] - cs1[:n]) / w
        rmean2 = (cs2[w : w + n] - cs2[:n]) / w
        cols[f"pat_hydro_var_{tag}"] = np.maximum(rmean2 - rmean**2, 0.0)
        shape = (n, w)
        strides = (hpad.strides[0], hpad.strides[0])
        wins = np.lib.stride_tricks.as_strided(
            hpad, shape=shape, strides=strides
        )
        medians = np.median(wins, axis=1)
        cols[f"pat_hydro_mad_{tag}"] = np.median(
            np.abs(wins - medians[:, None]), axis=1
        )

    # Charge transitions, pos/neg patch for w11, w25, w50
    for w in [11, 25, 50]:
        tag = f"w{w}"
        pad_c = w // 2
        cpad = np.pad(chg, (pad_c, pad_c), mode="edge")
        shape = (n, w)
        strides = (cpad.strides[0], cpad.strides[0])
        wins = np.lib.stride_tricks.as_strided(
            cpad, shape=shape, strides=strides
        )
        signs = np.sign(wins)
        cols[f"pat_charge_trans_{tag}"] = (
            np.sum(signs[:, 1:] != signs[:, :-1], axis=1) / max(1, w - 1)
        )
        cols[f"pat_pos_patch_{tag}"] = (
            np.sum(wins > 0, axis=1).astype(float) / w
        )
        cols[f"pat_neg_patch_{tag}"] = (
            np.sum(wins < 0, axis=1).astype(float) / w
        )

    # Disorder block fraction & transitions for w11, w25
    dis_mask = np.array(
        [1.0 if c in DISORDER_PROMOTING else 0.0 for c in seq]
    )
    for w in [11, 25]:
        tag = f"w{w}"
        pad_d = w // 2
        dpad = np.pad(dis_mask, (pad_d, pad_d), mode="edge")
        cs_d = np.insert(np.cumsum(dpad), 0, 0.0)
        cols[f"pat_dis_block_{tag}"] = (cs_d[w : w + n] - cs_d[:n]) / w
        spad_d = seq[0] * pad_d + seq + seq[-1] * pad_d
        dis_flags_pad = np.array(
            [1.0 if c in DISORDER_PROMOTING else 0.0 for c in spad_d]
        )
        shape_d = (n, w)
        strides_d = (dis_flags_pad.strides[0], dis_flags_pad.strides[0])
        wins_d = np.lib.stride_tricks.as_strided(
            dis_flags_pad, shape=shape_d, strides=strides_d
        )
        cols[f"pat_dis_trans_{tag}"] = (
            np.sum(wins_d[:, 1:] != wins_d[:, :-1], axis=1).astype(float)
            / max(1, w - 1)
        )

    # Kappa for w11, w25
    for w in [11, 25]:
        tag = f"w{w}"
        pad_k = w // 2
        kpad = np.pad(chg, (pad_k, pad_k), mode="edge")
        shape_k = (n, w)
        strides_k = (kpad.strides[0], kpad.strides[0])
        wins_k = np.lib.stride_tricks.as_strided(
            kpad, shape=shape_k, strides=strides_k
        )
        overall_var = np.var(wins_k, axis=1)
        bsize = 5
        nblocks = w // bsize
        if nblocks >= 2:
            block_part = wins_k[:, : nblocks * bsize].reshape(
                n, nblocks, bsize
            )
            block_means = block_part.mean(axis=2)
            block_var = np.var(block_means, axis=1)
            kappa_arr = np.where(
                overall_var > 1e-12, block_var / overall_var, 0.0
            )
        else:
            kappa_arr = np.zeros(n)
        cols[f"pat_kappa_{tag}"] = kappa_arr

    # Uversky axis for w11, w25, w50
    for w in [11, 25, 50]:
        tag = f"w{w}"
        pad_u = w // 2
        hpad_u = np.pad(hyd, (pad_u, pad_u), mode="edge")
        cpad_u = np.pad(chg, (pad_u, pad_u), mode="edge")
        cs_h = np.insert(np.cumsum(hpad_u), 0, 0.0)
        cs_c = np.insert(np.cumsum(cpad_u), 0, 0.0)
        hmean = (cs_h[w : w + n] - cs_h[:n]) / w
        cmean = (cs_c[w : w + n] - cs_c[:n]) / w
        cols[f"pat_uversky_axis_{tag}"] = (
            hmean - 1.151 * np.abs(cmean) - 0.693
        )

    # Poly-AA run flags
    for a in ["P", "E", "K", "Q", "S", "G", "D", "N"]:
        cols[f"in_poly_{a}_run_ge3"] = _poly_run_flags(seq, a, run_len=3)

    # Unknown AA flag
    cols["is_unknown_aa"] = np.array(
        [0.0 if c in AA_SET else 1.0 for c in seq], dtype=float
    )

    # SCD local (window=25)
    cols["scd_local"] = _scd_local(cols["charge_pH7"])

    base_df = pd.DataFrame(cols, index=idx)

    # ── PSSM block ────────────────────────────────────────────────────────
    has_pssm = pssm is not None and all(k in pssm.columns for k in list(AA))
    if not has_pssm:
        pssm_cols = {a: np.zeros(n) for a in AA}
        pssm_cols["pssm_entropy"] = np.zeros(n)
        pssm_cols["pssm_max_score"] = np.zeros(n)
        pssm_cols["pssm_variance"] = np.zeros(n)
        pssm_cols["has_pssm_data"] = np.zeros(n)
        pssm_df = pd.DataFrame(pssm_cols, index=idx)
    else:
        P = pssm[AA].to_numpy(dtype=float)
        row_sum = P.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        probs = P / row_sum
        H = -(np.where(probs > 0, probs * np.log(probs), 0.0)).sum(axis=1)
        pssm_df = pd.DataFrame(
            {a: P[:, i] for i, a in enumerate(AA)}, index=idx
        )
        pssm_df["pssm_entropy"] = H
        pssm_df["pssm_max_score"] = P.max(axis=1)
        pssm_df["pssm_variance"] = P.var(axis=1)
        pssm_df["has_pssm_data"] = 1.0

    all_df = pd.concat([base_df, pssm_df], axis=1)
    return FeatureBlock(
        base_df=base_df, pssm_df=pssm_df, all_df=all_df, has_pssm=has_pssm
    )
