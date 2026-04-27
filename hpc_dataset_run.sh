#!/bin/bash
#SBATCH --job-name=hpc_dataset_run
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=hpc_dataset_run_%j.out
#SBATCH --error=hpc_dataset_run_%j.err

# Full-pipeline production run against the validated 41,196-complex HPC dataset.
# Reads .pdb.bz2 / .pkl.bz2 directly; no staging or manual decompression required.
#
# Submit:
#     sbatch /scratch/prj/chmi_msa/protein-complexes-toolkit-hpc/hpc_dataset_run.sh

set -euo pipefail

# -----------------------------
# Fixed HPC paths
# -----------------------------
export PROTEIN_TOOLKIT_PROJECT_ROOT=/scratch/prj/chmi_msa/protein-complexes-toolkit-hpc
export PROTEIN_COMPLEXES_ROOT=/scratch/prj/chmi_msa/Protein_Complexes

cd "$PROTEIN_TOOLKIT_PROJECT_ROOT"

# -----------------------------
# Python environment
# -----------------------------
module purge
module load python/3.11.6-gcc-13.2.0
source .venv/bin/activate

# -----------------------------
# Output filenames
# Keep these as variables so toolkit.py and visualise_results.py always agree.
# -----------------------------
OUTPUT_CSV="results.csv"
INTERFACES_JSONL="interfaces.jsonl"
OUTPUT_DIR="Output"

# -----------------------------
# HPC-safe runtime settings
# -----------------------------
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PROTEIN_TOOLKIT_PROJECT_ROOT/.matplotlib"

# Avoid CPU oversubscription from BLAS/OpenMP inside each Python worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Use one toolkit worker per allocated SLURM CPU.
# With #SBATCH --cpus-per-task=16, this becomes WORKERS=16.
# The :-16 fallback only applies when running outside SLURM (e.g. interactive testing).
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "$OUTPUT_DIR" data/complex_manifest_audit "$MPLCONFIGDIR"

echo "============================================================"
echo "HPC dataset run started: $(date)"
echo "Host:         $(hostname)"
echo "SLURM job ID:  ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "SLURM CPUs:    ${SLURM_CPUS_PER_TASK:-unset}"
echo "Project root: $PROTEIN_TOOLKIT_PROJECT_ROOT"
echo "Complex root: $PROTEIN_COMPLEXES_ROOT"
echo "Workers:      $WORKERS"
echo "Output CSV:   $OUTPUT_CSV"
echo "Interfaces:   $INTERFACES_JSONL"
echo "Output dir:   $OUTPUT_DIR"
echo "Python:       $(which python)"
python --version
echo "============================================================"

echo "[0/4] Checking Python package consistency..."
python -m pip check

echo "[1/4] Checking registered data dependencies..."
python -u data_registry.py --root "$PROTEIN_TOOLKIT_PROJECT_ROOT"

echo "[2/4] Resolving complex manifest..."
python -u complex_resolver.py

echo "[3/4] Running full toolkit pipeline..."
# If compute nodes are firewalled from outbound HTTPS and API calls cause delays,
# append --no-api to the command below for a fully offline run.
python -u toolkit.py \
    --full-pipeline \
    --dir "$PROTEIN_COMPLEXES_ROOT" \
    --workers "$WORKERS" \
    --output "$OUTPUT_CSV" \
    --export-interfaces "$INTERFACES_JSONL"

echo "[4/4] Generating figures from $OUTPUT_CSV..."
python -u visualise_results.py "$OUTPUT_CSV" --output-dir "$OUTPUT_DIR"

echo "============================================================"
echo "HPC dataset run completed: $(date)"
echo "Results:      $PROTEIN_TOOLKIT_PROJECT_ROOT/$OUTPUT_CSV"
echo "Interfaces:  $PROTEIN_TOOLKIT_PROJECT_ROOT/$INTERFACES_JSONL"
echo "Figures:     $PROTEIN_TOOLKIT_PROJECT_ROOT/$OUTPUT_DIR"
echo "Manifest:    $PROTEIN_TOOLKIT_PROJECT_ROOT/data/complex_manifest_audit/complex_manifest.tsv"
echo "Incomplete:  $PROTEIN_TOOLKIT_PROJECT_ROOT/data/complex_manifest_audit/incomplete_inputs.tsv"
echo "============================================================"
