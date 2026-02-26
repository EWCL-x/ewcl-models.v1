#!/usr/bin/env bash
# Example: Run EWCL-Sequence on the example FASTA file.
#
# Prerequisites:
#   pip install -e .
#   python tools/build_model_zip.py   (builds dist/*.zip from model weights)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== EWCL-Sequence ==="
ewcl-predict \
    --model "$ROOT/dist/EWCL-Sequence_v1.0.0.zip" \
    --fasta "$SCRIPT_DIR/example.fasta" \
    --out   "$SCRIPT_DIR/results_sequence.csv" \
    --verbose

echo ""
echo "=== EWCL-Disorder ==="
ewcl-predict \
    --model "$ROOT/dist/EWCL-Disorder_v1.0.0.zip" \
    --fasta "$SCRIPT_DIR/example.fasta" \
    --out   "$SCRIPT_DIR/results_disorder.csv" \
    --verbose

echo ""
echo "=== EWCL-Structure (no PDB → pLDDT defaults to 50) ==="
ewcl-predict \
    --model "$ROOT/dist/EWCL-Structure_v1.0.0.zip" \
    --fasta "$SCRIPT_DIR/example.fasta" \
    --out   "$SCRIPT_DIR/results_structure.csv" \
    --verbose

echo ""
echo "Results written to examples/results_*.csv"
