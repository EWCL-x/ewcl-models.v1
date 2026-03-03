#!/usr/bin/env python3
"""Compare EWCL-Sequence and EWCL-Disorder predictions across:

1) Local models in this repo ("local")
2) Published model zips ("zip")
3) Deployed backend API (Railway) ("backend")

This is meant to be **adopter-friendly** and runnable by non-experts.

It checks:
- predictions length match
- numeric agreement between local and zip (should be identical)
- numeric agreement between local and backend (should be identical if deployed matches)

Usage (examples)
--------------
python tools/compare_local_zip_backend.py \
  --backend https://ewcl-api-production.up.railway.app \
  --uniprot P00441 --uniprot P04637

Notes
-----
- Uses backend endpoint `/api/predict` for both models.
- Requires internet access for backend and for downloading zips (if not cached).

"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import caid_dual_benchmark as cdb  # noqa: E402


@dataclass
class BackendResult:
    seq: np.ndarray
    dis: np.ndarray


def _http_post_multipart(url: str, fields: dict[str, str]) -> dict:
    """Tiny multipart/form-data POST without external deps."""
    boundary = "----ewclboundary7d0b8d06"
    body = bytearray()

    for k, v in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f"Content-Disposition: form-data; name=\"{k}\"\r\n\r\n".encode())
        body.extend(v.encode())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(url, data=bytes(body), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Accept", "application/json")

    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def fetch_backend_predictions(base_url: str, *, uniprot_id: str) -> BackendResult:
    base_url = base_url.rstrip("/")
    url = f"{base_url}/api/predict?{urllib.parse.urlencode({'uniprot_id': uniprot_id})}"
    data = _http_post_multipart(url, fields={})

    residues = data.get("residues") or []
    if not residues:
        raise RuntimeError(f"Backend returned no residues for {uniprot_id}")

    # Map model_id -> per-residue disorder score
    seq_scores: list[float] = []
    dis_scores: list[float] = []

    for r in residues:
        models = r.get("models") or []
        by_id = {m.get("model_id"): m for m in models}

        # Backend naming varies across deployments; accept both conventions.
        m_seq = by_id.get("OneEWCL_Sequence") or by_id.get("EWCL-Sequence")
        m_dis = by_id.get("OneEWCL_Disorder") or by_id.get("EWCL-Disorder")
        if m_seq is None or m_dis is None:
            raise RuntimeError(
                f"Backend residue missing expected model ids for {uniprot_id}: got {list(by_id)}"
            )

        seq_scores.append(float(m_seq.get("disorder")))
        dis_scores.append(float(m_dis.get("disorder")))

    return BackendResult(seq=np.asarray(seq_scores, dtype=float), dis=np.asarray(dis_scores, dtype=float))


def compare_arrays(a: np.ndarray, b: np.ndarray, *, label: str) -> tuple[float, float]:
    if a.shape != b.shape:
        raise RuntimeError(f"Shape mismatch for {label}: {a.shape} != {b.shape}")
    abs_max = float(np.max(np.abs(a - b)))
    abs_mean = float(np.mean(np.abs(a - b)))
    return abs_max, abs_mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend",
        default="https://ewcl-api-production.up.railway.app",
        help="Backend base URL (Railway).",
    )
    ap.add_argument(
        "--uniprot",
        action="append",
        default=[],
        help="UniProt ID(s) to test. Can be provided multiple times.",
    )
    args = ap.parse_args()

    uniprots = args.uniprot or ["P00441", "P04637"]

    local = cdb.load_local_models()
    zipm = cdb.load_zip_models()

    print("Comparing models: EWCL-Sequence, EWCL-Disorder")
    print(f"Backend: {args.backend}")

    for uid in uniprots:
        print(f"\n== {uid} ==")

        # Backend gives us the real sequence; reuse it for local/zip prediction
        backend_res = fetch_backend_predictions(args.backend, uniprot_id=uid)
        n = backend_res.seq.shape[0]

        # Get sequence from UniProt via backend output ordering isn’t enough for local.
        # We simply ask the backend again with include sequence by using its returned residues.
        # (aa field is present per residue)
        base_url = args.backend.rstrip("/")
        url = f"{base_url}/api/predict?{urllib.parse.urlencode({'uniprot_id': uid})}"
        data = _http_post_multipart(url, fields={})
        seq = "".join([r.get("aa", "X") for r in (data.get("residues") or [])])
        if len(seq) != n:
            raise RuntimeError(f"Sequence length mismatch from backend residues: {len(seq)} != {n}")

        preds_local = local.predict_all(seq)
        preds_zip = zipm.predict_all(seq)

        a1, m1 = compare_arrays(np.asarray(preds_local["EWCL-Sequence"]), np.asarray(preds_zip["EWCL-Sequence"]), label="local vs zip (sequence)")
        a2, m2 = compare_arrays(np.asarray(preds_local["EWCL-Disorder"]), np.asarray(preds_zip["EWCL-Disorder"]), label="local vs zip (disorder)")

        b1, bm1 = compare_arrays(np.asarray(preds_local["EWCL-Sequence"]), backend_res.seq, label="local vs backend (sequence)")
        b2, bm2 = compare_arrays(np.asarray(preds_local["EWCL-Disorder"]), backend_res.dis, label="local vs backend (disorder)")

        print(f"len={n}")
        print(f"EWCL-Sequence: local vs zip     abs_max={a1:.3e} abs_mean={m1:.3e}")
        print(f"EWCL-Sequence: local vs backend abs_max={b1:.3e} abs_mean={bm1:.3e}")
        print(f"EWCL-Disorder: local vs zip     abs_max={a2:.3e} abs_mean={m2:.3e}")
        print(f"EWCL-Disorder: local vs backend abs_max={b2:.3e} abs_mean={bm2:.3e}")

    print("\nOK")


if __name__ == "__main__":
    main()
