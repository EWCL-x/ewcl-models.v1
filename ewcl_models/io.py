"""I/O utilities for FASTA/PDB parsing and result output.

Provides file readers (FASTA, PDB/mmCIF) and output writers (CSV, Parquet,
JSONL) for the EWCL-Models command-line pipeline.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── FASTA reading ─────────────────────────────────────────────────────────────

def read_fasta(path: str | Path) -> List[Tuple[str, str]]:
    """Read a FASTA file and return ``[(id, sequence), …]``.

    Handles gzip-compressed files automatically.
    """
    path = Path(path)
    if path.suffix == ".gz":
        handle = io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    else:
        handle = open(path, encoding="utf-8")  # noqa: SIM115
    records: List[Tuple[str, str]] = []
    current_id: Optional[str] = None
    current_seq: List[str] = []
    try:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        if current_id is not None:
            records.append((current_id, "".join(current_seq)))
    finally:
        handle.close()
    return records


# ── Structure file (PDB / mmCIF) reading ─────────────────────────────────────

AA3_TO_1: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K", "HYP": "P",
    "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C",
}


def _maybe_gunzip(data: bytes) -> bytes:
    if len(data) >= 2 and data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def _sniff_format(data: bytes, filename: Optional[str]) -> str:
    name = (filename or "").lower()
    head = data[:4096].decode("latin-1", errors="ignore")
    if name.endswith(".cif") or "_atom_site." in head or "data_" in head:
        return "cif"
    if name.endswith(".pdb") or "ATOM  " in head or "HETATM" in head:
        return "pdb"
    return "cif" if "_atom_site." in head else "pdb"


def read_structure(
    path: str | Path,
) -> Tuple[pd.DataFrame, str, str]:
    """Parse a PDB or mmCIF file and return CA-atom records.

    Parameters
    ----------
    path : str or Path
        Path to ``.pdb``, ``.cif``, or gzipped variants.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``chain, resname, aa, bfactor``.  One row per residue
        (first model, polymer chains, amino-acid residues only, CA atom).
    sequence : str
        One-letter amino acid sequence extracted from the structure.
    parser_used : str
        ``"gemmi"`` or ``"biopython"`` — whichever parser succeeded.
    """
    path = Path(path)
    data = _maybe_gunzip(path.read_bytes())
    fmt = _sniff_format(data, path.name)

    # Try gemmi first
    try:
        import gemmi

        if fmt == "cif":
            doc = gemmi.cif.read_string(data.decode("utf-8", errors="ignore"))
            st = gemmi.make_structure_from_block(doc.sole_block())
        else:
            st = gemmi.read_pdb_string(data.decode("utf-8", errors="ignore"))
        rows: List[Dict[str, Any]] = []
        model = st[0]
        for chain in model:
            if not chain.is_polymer():
                continue
            for res in chain:
                if not res.is_amino_acid():
                    continue
                atom = None
                for name in ("CA", "N", "C", "CB"):
                    atom = res.find_atom(name, altloc="") or res.find_atom(
                        name, altloc="A"
                    )
                    if atom:
                        break
                if not atom:
                    continue
                aa = AA3_TO_1.get(res.name, "X")
                rows.append(
                    {
                        "chain": str(chain.name),
                        "resname": res.name,
                        "aa": aa,
                        "bfactor": float(atom.b_iso)
                        if hasattr(atom, "b_iso")
                        else 0.0,
                    }
                )
        if not rows:
            raise ValueError("No polymer residues found (gemmi)")
        df = pd.DataFrame(rows)
        seq = "".join(df["aa"].tolist())
        return df, seq, "gemmi"
    except Exception:
        pass

    # Fall back to BioPython
    from Bio.PDB import MMCIFParser, PDBParser, is_aa

    handle = io.StringIO(data.decode("utf-8", errors="ignore"))
    parser = MMCIFParser(QUIET=True) if fmt == "cif" else PDBParser(QUIET=True)
    st = parser.get_structure("input", handle)
    rows = []
    model = next(st.get_models())
    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            atom = res["CA"] if "CA" in res else None
            if atom is None:
                for alt in ("N", "C", "CB"):
                    if alt in res:
                        atom = res[alt]
                        break
            if atom is None:
                continue
            aa = AA3_TO_1.get(res.get_resname(), "X")
            rows.append(
                {
                    "chain": chain.id,
                    "resname": res.get_resname(),
                    "aa": aa,
                    "bfactor": float(atom.get_bfactor())
                    if hasattr(atom, "get_bfactor")
                    else 0.0,
                }
            )
    if not rows:
        raise ValueError("No polymer residues found (biopython)")
    df = pd.DataFrame(rows)
    seq = "".join(df["aa"].tolist())
    return df, seq, "biopython"


def extract_plddt(struct_df: pd.DataFrame) -> List[float]:
    """Extract per-residue pLDDT from structure DataFrame (AlphaFold b-factor convention)."""
    return struct_df["bfactor"].tolist()


# ── Output writers ────────────────────────────────────────────────────────────

def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write results to CSV."""
    df.to_csv(path, index=False, float_format="%.6f")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write results to Parquet."""
    df.to_parquet(path, index=False, engine="pyarrow")


def write_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    """Write results to JSONL (one JSON object per line)."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


WRITERS = {
    "csv": write_csv,
    "parquet": write_parquet,
    "jsonl": write_jsonl,
}
