#!/bin/bash
#SBATCH --job-name=hpc_incremental
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --output=hpc_incremental_%j.out
#SBATCH --error=hpc_incremental_%j.err
#
# SLURM wrapper for append-only incremental toolkit runs.
#
# Mirrors the environment hardening in hpc_dataset_run.sh but invokes `toolkit.py --full-pipeline --skip-existing results.csv` instead of the full pipeline. 
# Does NOT call complex_resolver.py separately (the toolkit handles discovery + audit) and does NOT auto-render figures (figures must come fromthe merged CSV, not from incremental-only data).
#
# Step layout:
#   [0/4] pip check                              — Python package consistency
#   [1/4] data_registry                          — registered data deps present
#   [2/4] resolver-baseline preflight            — runs/<id>/ from a prior baseline
#   [3/4] historical-output sanity check         — results.csv ⊇ interfaces.jsonl
#   [4/4] toolkit --full-pipeline --skip-existing
#
# Override behaviour via env vars when re-submitting (e.g. for crash recovery):
#   RUN_STAMP, JOB_TAG, OUTPUT_CSV, INTERFACES_JSONL — reuse the same paths so RESUME=1 finds the existing checkpoint.
#   HISTORICAL_RESULTS_CSV, HISTORICAL_INTERFACES_JSONL — point at non-default baseline outputs.
#   RESUME=1 — append --resume to the toolkit invocation.
#   LIMIT — cap chunk size at N complexes (post-skip-existing, pre-resume).
#           Combine with successive submissions to walk the dataset in memory-bounded chunks; merge incremental CSV/JSONL into HISTORICAL_RESULTS_CSV / HISTORICAL_INTERFACES_JSONL between runs.
#           IMPORTANT: do NOT modify HISTORICAL_RESULTS_CSV between an original failed attempt and its RESUME=1 retry; the historical baseline must be the same on both runs or chunk boundaries shift. 
#           The job logs a sha256 of the baseline at start so you can confirm the resume baseline matches the original.

set -euo pipefail

export PROTEIN_TOOLKIT_PROJECT_ROOT=/scratch/prj/chmi_msa/protein-complexes-toolkit-hpc
export PROTEIN_COMPLEXES_ROOT=/scratch/prj/chmi_msa/Protein_Complexes
cd "$PROTEIN_TOOLKIT_PROJECT_ROOT"

module purge
module load python/3.11.6-gcc-13.2.0
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PROTEIN_TOOLKIT_PROJECT_ROOT/.matplotlib"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WORKERS="${SLURM_CPUS_PER_TASK:-16}"

# Env-overridable so the operator can resubmit a failed job pointing at the
# same OUTPUT_CSV / INTERFACES_JSONL and recover via RESUME=1 (which adds
# --resume to the toolkit call).
RUN_STAMP="${RUN_STAMP:-$(date +%Y-%m-%d_%H%M%S)}"
JOB_TAG="${JOB_TAG:-job_${SLURM_JOB_ID:-manual}}"

HISTORICAL_RESULTS_CSV="${HISTORICAL_RESULTS_CSV:-results.csv}"
HISTORICAL_INTERFACES_JSONL="${HISTORICAL_INTERFACES_JSONL:-interfaces.jsonl}"
export HISTORICAL_RESULTS_CSV HISTORICAL_INTERFACES_JSONL  # consumed by Python sanity heredoc

OUTPUT_CSV="${OUTPUT_CSV:-results_incremental_${RUN_STAMP}_${JOB_TAG}.csv}"
INTERFACES_JSONL="${INTERFACES_JSONL:-interfaces_incremental_${RUN_STAMP}_${JOB_TAG}.jsonl}"

RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_ARGS+=(--resume)
fi

# Fail fast on invalid LIMIT before SLURM commits compute; matches the Python-side parser.error("--limit must be a positive integer") check.
if [[ -n "${LIMIT:-}" ]]; then
    if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: LIMIT must be a positive integer; got '$LIMIT'" >&2
        exit 2
    fi
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS+=(--limit "$LIMIT")
fi

mkdir -p data/complex_manifest_audit "$MPLCONFIGDIR"

echo "============================================================"
echo "Incremental HPC run"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-manual}"
echo "Workers: $WORKERS"
echo "Historical results: $HISTORICAL_RESULTS_CSV"
echo "Historical interfaces: $HISTORICAL_INTERFACES_JSONL"
echo "Output CSV: $OUTPUT_CSV"
echo "Output interfaces JSONL: $INTERFACES_JSONL"
echo "Resume: ${RESUME:-0}"
echo "Limit: ${LIMIT:-unlimited}"
# Audit fingerprints for the historical baseline. Compare these between an original failed attempt and its RESUME=1 retry to confirm the baseline did not change underfoot; chunk membership depends on it.
if [[ -f "$HISTORICAL_RESULTS_CSV" ]]; then
    sha256sum "$HISTORICAL_RESULTS_CSV" || true
fi
if [[ -f "$HISTORICAL_INTERFACES_JSONL" ]]; then
    sha256sum "$HISTORICAL_INTERFACES_JSONL" || true
fi
echo "============================================================"

echo "[0/4] Checking Python package consistency..."
python -m pip check

echo "[1/4] Checking registered data dependencies..."
python -u data_registry.py --root "$PROTEIN_TOOLKIT_PROJECT_ROOT"

echo "[2/4] Checking pre-expansion resolver baseline exists..."
test -s data/complex_manifest_audit/latest/complex_manifest.tsv
test -s data/complex_manifest_audit/latest_run_id.txt

echo "[3/4] Checking historical baseline outputs..."
test -s "$HISTORICAL_RESULTS_CSV"
test -s "$HISTORICAL_INTERFACES_JSONL"

python - <<'PY'
import csv
import json
import os
from pathlib import Path

results_path = Path(os.environ["HISTORICAL_RESULTS_CSV"])
jsonl_path = Path(os.environ["HISTORICAL_INTERFACES_JSONL"])

with results_path.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    if "complex_name" not in (reader.fieldnames or []):
        raise SystemExit(f"{results_path} is missing required complex_name column")
    result_names = {r["complex_name"] for r in reader if r.get("complex_name")}

jsonl_names = set()
with jsonl_path.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            jsonl_names.add(json.loads(line)["complex_name"])

print(f"results complexes: {len(result_names)}")
print(f"interface records: {len(jsonl_names)}")
print(f"jsonl subset of results: {jsonl_names <= result_names}")
if not jsonl_names <= result_names:
    raise SystemExit(
        f"{jsonl_path} contains complex_name values absent from {results_path}"
    )
PY

echo "[4/4] Running incremental toolkit pipeline..."
python -u toolkit.py --full-pipeline \
    --dir "$PROTEIN_COMPLEXES_ROOT" \
    --workers "$WORKERS" \
    --skip-existing "$HISTORICAL_RESULTS_CSV" \
    --output "$OUTPUT_CSV" \
    --export-interfaces "$INTERFACES_JSONL" \
    "${RESUME_ARGS[@]}" \
    "${LIMIT_ARGS[@]}"

echo "============================================================"
echo "Incremental run complete"
echo "Output CSV: $OUTPUT_CSV"
echo "Output interfaces JSONL: $INTERFACES_JSONL"
echo "Next: merge historical + incremental CSV/JSONL into combined files,"
echo "promote the combined CSV as the new --skip-existing reference"
echo "before the next incremental run, then run visualise_results.py on the combined CSV."
echo ""
echo "Recommended production visualisation command (Decision #49 figure suite —"
echo "24 distinct titles / 36 PNGs at full settings):"
echo "    python visualise_results.py <merged.csv> --output-dir Output \\"
echo "        --full-figure-pack --human-supplement"
echo ""
echo "Override flags:"
echo "    --no-corpus-funnel        suppress Fig 0 Corpus Funnel"
echo "    --no-screening-figures    suppress Fig 17 Screening Landscape"
echo "    --nonhuman-supplement     add the _nonhuman dual-emit pass (Fig 9 force-skipped)"
echo "    --multimer-supplement     emit multimer-exploratory Fig 7 panels"
echo "    --legacy-mode             pre-2026-05-11 load_data() behaviour"
echo "    --density                 KDE density contour overlays on scatter figures"
echo "    --skip-diagnostics        suppress warn_missing_required_rows() per-figure summaries"
echo "Minimal main-text-only pack (Figs 0/1/3/4/5/7/8/12/16/17 on calibrated dimer scope):"
echo "    python visualise_results.py <merged.csv> --output-dir Output"
echo "============================================================"
