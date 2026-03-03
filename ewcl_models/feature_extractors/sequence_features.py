"""Frozen sequence feature extractor for EWCL publication models.

Produces the feature matrices consumed by EWCL-Sequence (249 features),
EWCL-Disorder (239 features), and the sequence-derived portion of
EWCL-Structure (224 base features).

This file is the canonical feature extraction pipeline used to train the
publication models.  **Do not modify** — any change will invalidate the
trained weights.  The canonical feature order for each model is stored in its
``feature_list.json``.

Synced from backend canonical extractor: 2026-03-03.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd

# ---------- constants ----------
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

# Kyte-Doolittle hydropathy
KD = {
    'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,
    'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,
    'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5
}

# Polarity (Grantham; lower=nonpolar, higher=polar)
POLARITY = {
    'G':0,'A':0,'V':0,'L':0,'I':0,'F':0,'W':0,'Y':0,
    'S':1,'T':1,'C':1,'M':1,'N':1,'Q':1,'D':1,'E':1,'K':1,'R':1,'H':1,'P':0
}

# VdW volume (approx, A^3)
VDW = {
    'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,
    'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,
    'T':116.1,'W':227.8,'Y':193.6,'V':140.0
}

# Flexibility (Bhaskaran & Ponnuswamy)
FLEX = {
    'A':0.357,'R':0.529,'N':0.463,'D':0.511,'C':0.346,'Q':0.493,'E':0.497,'G':0.544,
    'H':0.323,'I':0.462,'L':0.365,'K':0.466,'M':0.295,'F':0.314,'P':0.509,'S':0.507,
    'T':0.444,'W':0.305,'Y':0.420,'V':0.386
}

# Bulkiness (Zimmerman)
BULK = {
    'A':11.5,'R':14.28,'N':12.82,'D':11.68,'C':13.46,'Q':14.45,'E':13.57,'G':3.4,
    'H':13.69,'I':21.4,'L':21.4,'K':15.71,'M':16.25,'F':19.8,'P':17.43,'S':9.47,
    'T':15.77,'W':21.67,'Y':18.03,'V':21.57
}

# Chou-Fasman helix & sheet propensity (normalized)
HELIX = {'A':1.45,'R':1.00,'N':0.67,'D':1.01,'C':0.77,'Q':1.11,'E':1.51,'G':0.57,
         'H':1.00,'I':1.08,'L':1.34,'K':1.07,'M':1.20,'F':1.12,'P':0.57,'S':0.77,
         'T':0.83,'W':1.14,'Y':0.61,'V':1.06}
SHEET = {'A':0.97,'R':0.90,'N':0.89,'D':0.54,'C':1.30,'Q':1.10,'E':0.37,'G':0.75,
         'H':0.87,'I':1.60,'L':1.22,'K':0.74,'M':1.67,'F':1.28,'P':0.55,'S':0.75,
         'T':1.19,'W':1.19,'Y':1.29,'V':1.70}

# Charge at pH 7 (very coarse: +1 K/R, -1 D/E, ~0 others; H as +0.1)
CHARGE7 = {**{a:0.0 for a in AA}, 'K':+1.0,'R':+1.0,'D':-1.0,'E':-1.0,'H':+0.1}

WINDOWS = [5, 11, 25, 50, 100]

# AA to index mapping for aa_encoded feature
AA_TO_IDX = {a: i for i, a in enumerate("ARNDCQEGHILKMFPSTWYV")}

# ---------- helpers ----------
def _seq_to_vec(seq: str, table: Dict[str, float]) -> np.ndarray:
    return np.array([table.get(a, 0.0) for a in seq], dtype=float)

def _rolling_stats(x: np.ndarray, w: int) -> Dict[str, np.ndarray]:
    """Vectorized rolling mean/std/min/max using cumsum + stride tricks."""
    n = len(x)
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode='edge')

    # ── rolling mean via cumsum (O(n)) ──
    cs = np.cumsum(xpad)
    cs = np.insert(cs, 0, 0.0)          # cs[0] = 0
    out_mean = (cs[w:w + n] - cs[:n]) / w

    # ── rolling variance via cumsum of squares ──
    cs2 = np.cumsum(xpad ** 2)
    cs2 = np.insert(cs2, 0, 0.0)
    mean_sq = (cs2[w:w + n] - cs2[:n]) / w
    out_std = np.sqrt(np.maximum(mean_sq - out_mean ** 2, 0.0))  # population std

    # ── rolling min/max via stride_tricks (vectorised) ──
    shape = (n, w)
    strides = (xpad.strides[0], xpad.strides[0])
    windows = np.lib.stride_tricks.as_strided(xpad, shape=shape, strides=strides)
    out_min = windows.min(axis=1)
    out_max = windows.max(axis=1)

    return {
        "mean": out_mean,
        "std":  out_std,
        "min":  out_min,
        "max":  out_max,
    }

def _window_entropy(seq: str, w: int) -> np.ndarray:
    """Shannon entropy of AA distribution in window — vectorised via rolling counts."""
    n = len(seq)
    pad = w // 2
    spad = seq[0] * pad + seq + seq[-1] * pad
    # Build one-hot (len_padded × 20) and use cumsum for rolling counts
    npad = len(spad)
    aa_idx = {a: i for i, a in enumerate(AA)}
    onehot = np.zeros((npad, 20), dtype=float)
    for j, c in enumerate(spad):
        k = aa_idx.get(c, -1)
        if k >= 0:
            onehot[j, k] = 1.0
    cs = np.cumsum(onehot, axis=0)
    cs = np.vstack([np.zeros((1, 20)), cs])  # prepend row of zeros
    # rolling counts: cs[i+w] - cs[i] for each position
    counts = cs[w:w + n] - cs[:n]  # shape (n, 20)
    p = counts / w
    # entropy: -sum(p * log(p)) where p > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.where(p > 0, np.log(p), 0.0)
    out = -(p * logp).sum(axis=1)
    return out

def _low_complex_from_entropy(H: np.ndarray, thresh: float = 1.5) -> np.ndarray:
    # Flag low complexity by entropy threshold
    return (H < thresh).astype(float)

def _composition_bias(seq: str, w: int) -> np.ndarray:
    """(max_fraction - uniform_fraction) — vectorised via rolling AA counts."""
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
    counts = cs[w:w + n] - cs[:n]  # (n, 20)
    frac = counts / w
    out = frac.max(axis=1) - (1.0 / 20.0)
    return out

def _uversky_distance(hydro_mean: np.ndarray, charge_mean: np.ndarray) -> np.ndarray:
    # Uversky boundary (approx): H = 1.151*|Q| + 0.693; distance = H_mean - (1.151*|Q_mean| + 0.693)
    return hydro_mean - (1.151*np.abs(charge_mean) + 0.693)

def _poly_run_flags(seq: str, aa: str, run_len: int = 3) -> np.ndarray:
    out = np.zeros(len(seq))
    cur = 0
    for i, c in enumerate(seq):
        cur = cur + 1 if c == aa else 0
        if cur >= run_len:
            out[i] = 1.0
    return out

def _scd_local(charge_vec: np.ndarray, w: int = 25) -> np.ndarray:
    """SCD (Sequence Charge Decoration) - vectorised per-window."""
    n = len(charge_vec)
    pad = w // 2
    xpad = np.pad(charge_vec, (pad, pad), mode='edge')

    # Pre-compute sqrt distance weights once for the window size
    dist_sqrt = np.sqrt(np.arange(1, w, dtype=float))  # len w-1

    out = np.zeros(n)
    denom = (w * (w - 1)) / 2.0 if w > 1 else 1.0
    for i in range(n):
        win = xpad[i:i + w]
        # Vectorised: for each pair (a, b), sum qa*qb*sqrt(b-a)
        # Equivalent to sum over d=1..w-1 of sum_a(win[a]*win[a+d]) * sqrt(d)
        s = 0.0
        for d in range(1, w):
            s += float(np.dot(win[:w - d], win[d:])) * dist_sqrt[d - 1]
        out[i] = s / denom
    return out

# ---------- NEW v2 feature helpers (lcd_, pat_, pos_, prot_, seq_) ----------

AROMATIC = set('FWY')
DISORDER_PROMOTING = set('AEQSGPDRK')
ORDER_PROMOTING = set('CFILMVWY')

def _windowed_apply(seq_or_vec, w: int, func, is_str: bool = True):
    """Apply func to each centered window. Works on string or numeric array."""
    n = len(seq_or_vec)
    pad = w // 2
    if is_str:
        spad = seq_or_vec[0] * pad + seq_or_vec + seq_or_vec[-1] * pad
    else:
        spad = np.pad(seq_or_vec, (pad, pad), mode='edge')
    out = np.zeros(n)
    for i in range(n):
        out[i] = func(spad[i:i + w])
    return out

def _top1_dominance(win):
    """Fraction of most common AA in window."""
    counts = {}
    for c in win:
        counts[c] = counts.get(c, 0) + 1
    return max(counts.values()) / len(win) if win else 0.0

def _wf_complexity(win):
    """Wootton-Federhen complexity: product of (count/n)^(count/n)."""
    n = len(win)
    if n == 0:
        return 0.0
    counts = {}
    for c in win:
        counts[c] = counts.get(c, 0) + 1
    prod = 1.0
    for cnt in counts.values():
        f = cnt / n
        prod *= f ** f
    return prod

def _lz_proxy(win):
    """Lempel-Ziv complexity proxy: number of distinct substrings of length 2."""
    bigrams = set()
    for i in range(len(win) - 1):
        bigrams.add(win[i:i + 2])
    max_possible = min(20 * 20, max(1, len(win) - 1))
    return len(bigrams) / max_possible

def _longest_run(win):
    """Length of longest homopolymer run in window."""
    if len(win) == 0:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(win)):
        if win[i] == win[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best

def _mean_run(win):
    """Mean homopolymer run length in window."""
    if len(win) == 0:
        return 0.0
    runs = []
    cur = 1
    for i in range(1, len(win)):
        if win[i] == win[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return np.mean(runs)

def _frac_runs_ge4(win):
    """Fraction of residues in runs of length >= 4."""
    if len(win) == 0:
        return 0.0
    runs = []
    cur = 1
    for i in range(1, len(win)):
        if win[i] == win[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    total_ge4 = sum(r for r in runs if r >= 4)
    return total_ge4 / len(win)

def _aro_density(win):
    """Fraction of aromatic residues in window."""
    return sum(1 for c in win if c in AROMATIC) / max(1, len(win))

def _repeat_score(win):
    """Simple repeat score: fraction of positions where win[i] == win[i+2] (period-2 repeats)."""
    if len(win) < 3:
        return 0.0
    matches = sum(1 for i in range(len(win) - 2) if win[i] == win[i + 2])
    return matches / (len(win) - 2)

def _hydro_var(win_vec):
    """Variance of hydropathy in window."""
    return float(np.var(win_vec))

def _hydro_mad(win_vec):
    """Median absolute deviation of hydropathy in window."""
    med = np.median(win_vec)
    return float(np.median(np.abs(win_vec - med)))

def _charge_transitions(win_vec):
    """Fraction of charge sign transitions in window."""
    if len(win_vec) < 2:
        return 0.0
    signs = np.sign(win_vec)
    transitions = np.sum(signs[1:] != signs[:-1])
    return float(transitions / (len(win_vec) - 1))

def _pos_patch_count(win_vec):
    """Fraction of consecutive positive residues (patches of +charge)."""
    if len(win_vec) == 0:
        return 0.0
    pos = (win_vec > 0).astype(int)
    return float(np.sum(pos)) / len(win_vec)

def _neg_patch_count(win_vec):
    """Fraction of consecutive negative residues."""
    if len(win_vec) == 0:
        return 0.0
    neg = (win_vec < 0).astype(int)
    return float(np.sum(neg)) / len(win_vec)

def _dis_block_frac(win):
    """Fraction of disorder-promoting AAs in window."""
    return sum(1 for c in win if c in DISORDER_PROMOTING) / max(1, len(win))

def _dis_transitions(win):
    """Fraction of transitions between disorder-promoting and order-promoting AAs."""
    if len(win) < 2:
        return 0.0
    trans = 0
    for i in range(len(win) - 1):
        a_dis = win[i] in DISORDER_PROMOTING
        b_dis = win[i + 1] in DISORDER_PROMOTING
        if a_dis != b_dis:
            trans += 1
    return trans / (len(win) - 1)

def _kappa_window(charge_vec_win):
    """Kappa (charge segregation) for a window: var(charge_block_means) / var(charge).
    Uses block size = 5."""
    n = len(charge_vec_win)
    if n < 5:
        return 0.0
    overall_var = np.var(charge_vec_win)
    if overall_var < 1e-12:
        return 0.0
    bsize = 5
    nblocks = n // bsize
    if nblocks < 2:
        return 0.0
    block_means = [charge_vec_win[i * bsize:(i + 1) * bsize].mean() for i in range(nblocks)]
    return float(np.var(block_means) / overall_var)

def _uversky_axis_window(hydro_win, charge_win):
    """Uversky axis: mean_hydropathy - 1.151*|mean_charge| - 0.693"""
    return float(np.mean(hydro_win) - 1.151 * abs(np.mean(charge_win)) - 0.693)

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
    """CV of spacing between aromatic residues. 0 if < 2 aromatics."""
    positions = [i for i, c in enumerate(seq) if c in AROMATIC]
    if len(positions) < 2:
        return 0.0
    spacings = np.diff(positions).astype(float)
    m = spacings.mean()
    if m < 1e-12:
        return 0.0
    return float(spacings.std() / m)

@dataclass
class FeatureBlock:
    base_df: pd.DataFrame
    pssm_df: pd.DataFrame
    all_df: pd.DataFrame
    has_pssm: bool

# ---------- main API ----------
def build_ewclv1_features(seq: str,
                          pssm: pd.DataFrame | None = None,
                          expand_aa_onehot: bool = False) -> FeatureBlock:
    """
    Build the feature matrix for EWCL v1. If `pssm` is None, PSSM fields will be zeroed and has_pssm_data=0.
    PSSM expected columns: 'A','R','N','D',...,'Y','V' plus we compute pssm_entropy, pssm_max_score, pssm_variance.
    """
    seq = seq.strip().upper()
    n = len(seq)
    idx = np.arange(1, n+1)

    # scalar per-residue tracks using EXACT names from schema
    hyd = _seq_to_vec(seq, KD)
    pol = _seq_to_vec(seq, POLARITY) 
    vdw = _seq_to_vec(seq, VDW)
    flex = _seq_to_vec(seq, FLEX)
    bulk = _seq_to_vec(seq, BULK)
    helx = _seq_to_vec(seq, HELIX)
    beta = _seq_to_vec(seq, SHEET)
    chg  = _seq_to_vec(seq, CHARGE7)

    # rolling stats for each track & each window - use EXACT prefixes from schema
    cols = {}
    def emit_track(prefix: str, x: np.ndarray):
        # Only emit windowed stats, not the base track itself
        for w in WINDOWS:
            stats = _rolling_stats(x, w)
            cols[f"{prefix}_w{w}_mean"] = stats["mean"]
            cols[f"{prefix}_w{w}_std"]  = stats["std"]
            cols[f"{prefix}_w{w}_min"]  = stats["min"]
            cols[f"{prefix}_w{w}_max"]  = stats["max"]

    # Use EXACT names expected by model - generate windowed stats only
    emit_track("hydro", hyd)        # window stats will be hydro_w5_mean, etc.
    emit_track("polar", pol)        # window stats will be polar_w5_mean, etc. (NOT polarity_)
    emit_track("vdw", vdw)          # window stats will be vdw_w5_mean, etc.
    emit_track("flex", flex)        # window stats will be flex_w5_mean, etc.
    emit_track("bulk", bulk)        # window stats will be bulk_w5_mean, etc.
    emit_track("helix_prop", helx)  # window stats will be helix_prop_w5_mean, etc.
    emit_track("sheet_prop", beta)  # window stats will be sheet_prop_w5_mean, etc.
    emit_track("charge", chg)       # window stats will be charge_w5_mean, etc.

    # Base single-residue features with EXACT schema names
    cols["hydropathy"] = hyd        # NOT hydro
    cols["polarity"] = pol          # keep as polarity (base feature)
    cols["vdw_volume"] = vdw        # NOT vdw
    cols["flexibility"] = flex      # NOT flex
    cols["bulkiness"] = bulk        # NOT bulk
    cols["helix_prop"] = helx       # keep as helix_prop
    cols["sheet_prop"] = beta       # keep as sheet_prop
    cols["charge_pH7"] = chg        # NOT charge

    # windowed entropies + low complexity + comp bias + Uversky distance
    for w in WINDOWS:
        Hw = _window_entropy(seq, w)
        cols[f"entropy_w{w}"]    = Hw
        cols[f"low_complex_w{w}"] = _low_complex_from_entropy(Hw)
        cols[f"comp_bias_w{w}"]   = _composition_bias(seq, w)
        cols[f"uversky_dist_w{w}"] = _uversky_distance(cols[f"hydro_w{w}_mean"], cols[f"charge_w{w}_mean"])

    # simple composition fractions (global per window is complex; we add global fractions once)
    counts = np.array([seq.count(a) for a in AA], dtype=float)
    frac = counts / max(1, counts.sum())
    comp_cols = {f"comp_{a}": np.full(n, frac[i]) for i, a in enumerate(AA)}
    cols.update(comp_cols)

    cols["comp_frac_aromatic"] = np.full(n, frac[AA.index('F')] + frac[AA.index('W')] + frac[AA.index('Y')])
    cols["comp_frac_positive"] = np.full(n, frac[AA.index('K')] + frac[AA.index('R')] + frac[AA.index('H')])
    cols["comp_frac_negative"] = np.full(n, frac[AA.index('D')] + frac[AA.index('E')])
    cols["comp_frac_polar"]    = np.full(n, sum(frac[AA.index(x)] for x in ['S','T','N','Q','C','Y','H','K','R','D','E']))
    cols["comp_frac_aliphatic"]= np.full(n, frac[AA.index('A')] + frac[AA.index('V')] + frac[AA.index('L')] + frac[AA.index('I')])
    cols["comp_frac_proline"]  = np.full(n, frac[AA.index('P')])
    cols["comp_frac_glycine"]  = np.full(n, frac[AA.index('G')])

    # ---- Position features (pos_) ----
    pos_rel = np.arange(n, dtype=float) / max(1, n - 1)  # 0..1
    pos_dist_nterm = np.arange(n, dtype=float)             # 0..n-1
    pos_dist_cterm = np.arange(n - 1, -1, -1, dtype=float) # n-1..0
    pos_dist_terminus = np.minimum(pos_dist_nterm, pos_dist_cterm)
    cols["pos_rel"] = pos_rel
    cols["pos_norm"] = pos_rel                                       # alias for EWCL-Disorder
    cols["pos_dist_terminus_norm"] = pos_dist_terminus / max(1, n - 1)
    cols["pos_log_dist_terminus"] = np.log1p(pos_dist_terminus)
    cols["pos_is_nterm"] = (pos_dist_nterm < max(1, n * 0.05)).astype(float)
    cols["pos_is_cterm"] = (pos_dist_cterm < max(1, n * 0.05)).astype(float)
    cols["pos_dist_nterm_norm"] = pos_dist_nterm / max(1, n - 1)
    cols["pos_dist_cterm_norm"] = pos_dist_cterm / max(1, n - 1)
    cols["pos_dist_n"] = cols["pos_dist_nterm_norm"]                 # alias
    cols["pos_dist_c"] = cols["pos_dist_cterm_norm"]                 # alias
    # Positional is_n/c at 10%/20%/30% thresholds
    for pct in [10, 20, 30]:
        cols[f"pos_is_n{pct}"] = (pos_dist_nterm < max(1, n * pct / 100)).astype(float)
        cols[f"pos_is_c{pct}"] = (pos_dist_cterm < max(1, n * pct / 100)).astype(float)
    # Exponential decay from termini
    nm1 = max(1, n - 1)
    for tau in [10, 30]:
        cols[f"pos_nexp{tau}"] = np.exp(-pos_dist_nterm / max(1, tau))
        cols[f"pos_cexp{tau}"] = np.exp(-pos_dist_cterm / max(1, tau))

    # ---- Sequence-level features (seq_) ----
    frac_pos_charge = frac[AA.index('K')] + frac[AA.index('R')] + frac[AA.index('H')]
    frac_neg_charge = frac[AA.index('D')] + frac[AA.index('E')]
    cols["seq_frac_charged"] = np.full(n, frac_pos_charge + frac_neg_charge)
    cols["seq_ncpr"] = np.full(n, frac_pos_charge - frac_neg_charge)  # net charge per residue

    # ---- Protein-level features (prot_) ----
    cols["prot_length_log"] = np.full(n, math.log(n) if n > 0 else 0.0)
    cols["prot_log_len"] = cols["prot_length_log"]                   # alias for EWCL-Disorder
    cols["prot_inv_len"] = np.full(n, 1.0 / max(1, n))              # for EWCL-Disorder
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
        # compute properly: var(block means) / var(charges)
        overall_var = float(np.var(chg))
        if overall_var > 1e-12:
            bsize = 5
            nblocks = n // bsize
            if nblocks >= 2:
                block_means = [float(chg[i * bsize:(i + 1) * bsize].mean()) for i in range(nblocks)]
                prot_kappa = float(np.var(block_means) / overall_var)
            else:
                prot_kappa = 0.0
        else:
            prot_kappa = 0.0
    else:
        prot_kappa = 0.0
    cols["prot_kappa"] = np.full(n, prot_kappa)

    # ---- LCD features (lcd_) - low complexity domain — vectorised ----
    # Encode sequence to int array once (each unique AA → unique int)
    _aa_to_int = {a: i for i, a in enumerate(sorted(set(seq)))}
    seq_int = np.array([_aa_to_int[c] for c in seq], dtype=np.int32)
    n_unique = len(_aa_to_int)

    for w in [5, 11, 25]:
        tag = f"w{w}"
        pad_lcd = w // 2
        # Pad sequence ints
        spad_int = np.pad(seq_int, (pad_lcd, pad_lcd), mode='edge')
        shape_lcd = (n, w)
        strides_lcd = (spad_int.strides[0], spad_int.strides[0])
        wins_int = np.lib.stride_tricks.as_strided(spad_int, shape=shape_lcd, strides=strides_lcd)

        # lcd_top1_dom: fraction of most common AA = max(count) / w
        # Use one-hot cumsum for per-window counts
        onehot_lcd = np.zeros((n + 2 * pad_lcd, n_unique), dtype=np.float64)
        for k, v in _aa_to_int.items():
            onehot_lcd[spad_int == v, v] = 1.0
        cs_lcd = np.vstack([np.zeros((1, n_unique)), np.cumsum(onehot_lcd, axis=0)])
        win_counts = cs_lcd[w:w + n] - cs_lcd[:n]  # (n, n_unique)
        cols[f"lcd_top1_dom_{tag}"] = win_counts.max(axis=1) / w

        # lcd_wf_complex: prod of (f^f) for each AA with count>0
        freqs = win_counts / w  # (n, n_unique)
        # f^f = exp(f * log(f)); 0^0 = 1 so contribution = 0 in log space
        log_terms = np.where(freqs > 0, freqs * np.log(freqs), 0.0)
        cols[f"lcd_wf_complex_{tag}"] = np.exp(log_terms.sum(axis=1))

        # lcd_lz_proxy: number of distinct bigrams / max_possible
        # Pad string for bigrams
        spad_str = seq[0] * pad_lcd + seq + seq[-1] * pad_lcd
        max_possible = min(400, max(1, w - 1))
        lz_arr = np.zeros(n)
        for i in range(n):
            ws = spad_str[i:i + w]
            bigrams = set()
            for j in range(len(ws) - 1):
                bigrams.add(ws[j:j + 2])
            lz_arr[i] = len(bigrams) / max_possible
        cols[f"lcd_lz_proxy_{tag}"] = lz_arr

        # lcd_longest_run: longest run of same AA in window — vectorised via stride_tricks
        # Compare adjacent elements: same[i,j] = (wins_int[i,j] == wins_int[i,j+1])
        same = (wins_int[:, 1:] == wins_int[:, :-1])  # (n, w-1)
        # For each window, compute longest consecutive True streak + 1
        longest_arr = np.ones(n, dtype=np.float64)
        cur_run = np.ones(n, dtype=np.float64)
        for j in range(w - 1):
            cur_run = np.where(same[:, j], cur_run + 1, 1.0)
            longest_arr = np.maximum(longest_arr, cur_run)
        cols[f"lcd_longest_run_{tag}"] = longest_arr

    # mean_run and frac_runs_ge4 for w11, w25
    for w in [11, 25]:
        tag = f"w{w}"
        pad_lcd = w // 2
        spad_int2 = np.pad(seq_int, (pad_lcd, pad_lcd), mode='edge')
        shape_lcd2 = (n, w)
        strides_lcd2 = (spad_int2.strides[0], spad_int2.strides[0])
        wins_int2 = np.lib.stride_tricks.as_strided(spad_int2, shape=shape_lcd2, strides=strides_lcd2)
        same2 = (wins_int2[:, 1:] == wins_int2[:, :-1])  # (n, w-1)
        # Process run statistics per window vectorised column-by-column
        # Track run lengths and accumulate mean_run / frac_runs_ge4
        # For each window: count runs and their lengths
        # Use column scan: cur_run tracks current run length
        cur_run2 = np.ones(n, dtype=np.float64)
        run_count = np.ones(n, dtype=np.float64)  # starts with 1 run
        total_len = np.zeros(n, dtype=np.float64)
        ge4_total = np.zeros(n, dtype=np.float64)
        for j in range(w - 1):
            continuing = same2[:, j]
            # Where run ends (not continuing), finalise current run
            ended = ~continuing
            # Add completed run length to total_len
            total_len += np.where(ended, cur_run2, 0.0)
            ge4_total += np.where(ended & (cur_run2 >= 4), cur_run2, 0.0)
            run_count += np.where(ended, 1.0, 0.0)
            # Reset or continue
            cur_run2 = np.where(continuing, cur_run2 + 1, 1.0)
        # Finalise last run
        total_len += cur_run2
        ge4_total += np.where(cur_run2 >= 4, cur_run2, 0.0)
        cols[f"lcd_mean_run_{tag}"] = total_len / run_count
        cols[f"lcd_frac_runs_ge4_{tag}"] = ge4_total / w

    # Global max run
    cols["lcd_max_run_global"] = np.full(n, float(_max_run_global(seq)))

    # Aromatic density for w11, w25 — vectorised via cumsum
    aro_mask = np.array([1.0 if c in AROMATIC else 0.0 for c in seq])
    for w in [11, 25]:
        pad_a = w // 2
        apad = np.pad(aro_mask, (pad_a, pad_a), mode='edge')
        cs_a = np.insert(np.cumsum(apad), 0, 0.0)
        cols[f"lcd_aro_density_w{w}"] = (cs_a[w:w + n] - cs_a[:n]) / w

    # Aromatic spacing CV (global)
    cols["lcd_aro_spacing_cv"] = np.full(n, _aro_spacing_cv(seq))

    # Repeat score for w11, w25 — vectorised via stride_tricks
    for w in [11, 25]:
        pad_r = w // 2
        spad_r = np.pad(seq_int, (pad_r, pad_r), mode='edge')
        shape_r = (n, w)
        strides_r = (spad_r.strides[0], spad_r.strides[0])
        wins_r = np.lib.stride_tricks.as_strided(spad_r, shape=shape_r, strides=strides_r)
        # repeat score: fraction of positions where win[i] == win[i+2]
        if w >= 3:
            matches_r = (wins_r[:, :-2] == wins_r[:, 2:]).sum(axis=1).astype(float)
            cols[f"lcd_repeat_score_w{w}"] = matches_r / (w - 2)
        else:
            cols[f"lcd_repeat_score_w{w}"] = np.zeros(n)

    # ---- Pattern features (pat_) ----
    # Hydropathy variance & MAD for w11, w25, w50 — vectorised via stride_tricks + cumsum
    for w in [11, 25, 50]:
        tag = f"w{w}"
        pad_h = w // 2
        hpad = np.pad(hyd, (pad_h, pad_h), mode='edge')
        # Variance via cumsum: Var = E[x^2] - E[x]^2
        cs1 = np.insert(np.cumsum(hpad), 0, 0.0)
        cs2 = np.insert(np.cumsum(hpad ** 2), 0, 0.0)
        rmean = (cs1[w:w + n] - cs1[:n]) / w
        rmean2 = (cs2[w:w + n] - cs2[:n]) / w
        cols[f"pat_hydro_var_{tag}"] = np.maximum(rmean2 - rmean ** 2, 0.0)
        # MAD: need strided windows for median
        shape = (n, w)
        strides = (hpad.strides[0], hpad.strides[0])
        wins = np.lib.stride_tricks.as_strided(hpad, shape=shape, strides=strides)
        medians = np.median(wins, axis=1)
        cols[f"pat_hydro_mad_{tag}"] = np.median(np.abs(wins - medians[:, None]), axis=1)

    # Charge transitions, pos/neg patch for w11, w25, w50 — vectorised
    for w in [11, 25, 50]:
        tag = f"w{w}"
        pad_c = w // 2
        cpad = np.pad(chg, (pad_c, pad_c), mode='edge')
        shape = (n, w)
        strides = (cpad.strides[0], cpad.strides[0])
        wins = np.lib.stride_tricks.as_strided(cpad, shape=shape, strides=strides)
        # charge transitions: sign changes / (w-1)
        signs = np.sign(wins)
        cols[f"pat_charge_trans_{tag}"] = np.sum(signs[:, 1:] != signs[:, :-1], axis=1) / max(1, w - 1)
        # pos patch: fraction of positive charges
        cols[f"pat_pos_patch_{tag}"] = np.sum(wins > 0, axis=1).astype(float) / w
        # neg patch: fraction of negative charges
        cols[f"pat_neg_patch_{tag}"] = np.sum(wins < 0, axis=1).astype(float) / w

    # Disorder block fraction & transitions for w11, w25 — vectorised via rolling counts
    dis_mask = np.array([1.0 if c in DISORDER_PROMOTING else 0.0 for c in seq])
    for w in [11, 25]:
        tag = f"w{w}"
        pad_d = w // 2
        dpad = np.pad(dis_mask, (pad_d, pad_d), mode='edge')
        cs_d = np.insert(np.cumsum(dpad), 0, 0.0)
        cols[f"pat_dis_block_{tag}"] = (cs_d[w:w + n] - cs_d[:n]) / w
        # transitions: count disorder/order boundaries
        spad_d = seq[0] * pad_d + seq + seq[-1] * pad_d
        dis_flags_pad = np.array([1.0 if c in DISORDER_PROMOTING else 0.0 for c in spad_d])
        shape_d = (n, w)
        strides_d = (dis_flags_pad.strides[0], dis_flags_pad.strides[0])
        wins_d = np.lib.stride_tricks.as_strided(dis_flags_pad, shape=shape_d, strides=strides_d)
        cols[f"pat_dis_trans_{tag}"] = np.sum(wins_d[:, 1:] != wins_d[:, :-1], axis=1).astype(float) / max(1, w - 1)

    # Kappa for w11, w25 — vectorised
    for w in [11, 25]:
        tag = f"w{w}"
        pad_k = w // 2
        kpad = np.pad(chg, (pad_k, pad_k), mode='edge')
        shape_k = (n, w)
        strides_k = (kpad.strides[0], kpad.strides[0])
        wins_k = np.lib.stride_tricks.as_strided(kpad, shape=shape_k, strides=strides_k)
        overall_var = np.var(wins_k, axis=1)
        # Block means: block_size=5
        bsize = 5
        nblocks = w // bsize
        if nblocks >= 2:
            # Reshape windows into blocks
            block_part = wins_k[:, :nblocks * bsize].reshape(n, nblocks, bsize)
            block_means = block_part.mean(axis=2)
            block_var = np.var(block_means, axis=1)
            kappa_arr = np.where(overall_var > 1e-12, block_var / overall_var, 0.0)
        else:
            kappa_arr = np.zeros(n)
        cols[f"pat_kappa_{tag}"] = kappa_arr

    # Uversky axis for w11, w25, w50 — vectorised via cumsum means
    for w in [11, 25, 50]:
        tag = f"w{w}"
        pad_u = w // 2
        hpad_u = np.pad(hyd, (pad_u, pad_u), mode='edge')
        cpad_u = np.pad(chg, (pad_u, pad_u), mode='edge')
        cs_h = np.insert(np.cumsum(hpad_u), 0, 0.0)
        cs_c = np.insert(np.cumsum(cpad_u), 0, 0.0)
        hmean = (cs_h[w:w + n] - cs_h[:n]) / w
        cmean = (cs_c[w:w + n] - cs_c[:n]) / w
        cols[f"pat_uversky_axis_{tag}"] = hmean - 1.151 * np.abs(cmean) - 0.693

    # poly runs (flags) - kept for OLD model compatibility
    for a in ['P','E','K','Q','S','G','D','N']:
        cols[f"in_poly_{a}_run_ge3"] = _poly_run_flags(seq, a, run_len=3)

    # unknown AA flag (should be 0 for canonical AA)
    cols["is_unknown_aa"] = np.array([0.0 if c in AA_SET else 1.0 for c in seq], dtype=float)

    # compute scd_local (local charge decoration; window=25)
    cols["scd_local"] = _scd_local(cols["charge_pH7"])

    # NOTE: aa_encoded is in base_features but not in all_features in the schema
    # Only add it if the model actually expects it
    # cols["aa_encoded"] = np.array([float(AA_TO_IDX.get(a, -1)) for a in seq])

    base_df = pd.DataFrame(cols, index=idx)

    # ---- PSSM block with EXACT names ----
    has_pssm = pssm is not None and all(k in pssm.columns for k in list(AA))
    if not has_pssm:
        # Zero-fill PSSM features to maintain exact 249 feature count
        pssm_cols = {a: np.zeros(n) for a in AA}  # A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
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
        H = -(np.where(probs>0, probs*np.log(probs), 0.0)).sum(axis=1)
        pssm_df = pd.DataFrame({a: P[:,i] for i,a in enumerate(AA)}, index=idx)
        pssm_df["pssm_entropy"] = H
        pssm_df["pssm_max_score"] = P.max(axis=1)
        pssm_df["pssm_variance"] = P.var(axis=1)
        pssm_df["has_pssm_data"] = 1.0

    # align indices and concat
    all_df = pd.concat([base_df, pssm_df], axis=1)
    return FeatureBlock(base_df=base_df, pssm_df=pssm_df, all_df=all_df, has_pssm=has_pssm)