#!/bin/bash
#SBATCH --job-name=mcpt_walkforward
#SBATCH --output=logs/walkforward_%j.out
#SBATCH --error=logs/walkforward_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# ============================================================================
# CESGA Finisterrae III - Walk-Forward MCPT Analysis
# ============================================================================
# Usage: sbatch run_walkforward.sbatch <strategy> [options]
# Example: sbatch run_walkforward.sbatch hawkes
#          sbatch run_walkforward.sbatch donchian --n-permutations 500
#
# Note: Walk-forward is more computationally intensive than in-sample,
# hence the longer default time limit (48h vs 24h).
#
# Check job status: squeue -u $USER
# Cancel job: scancel <job_id>
# View output: tail -f logs/walkforward_<job_id>.out
# ============================================================================

# Strict error handling
set -e

# Print job information
echo "=============================================="
echo "SLURM Job Information"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Project configuration - ADJUST THESE PATHS
export CESGA_PROJECT_ROOT="${CESGA_PROJECT_ROOT:-/mnt/lustre/scratch/nlsas/home/usc/ec/ahe/Trading-system}"
export CESGA_DATA_DIR="${CESGA_DATA_DIR:-$CESGA_PROJECT_ROOT/data}"
export CESGA_OUTPUT_DIR="${CESGA_OUTPUT_DIR:-$CESGA_PROJECT_ROOT/outputs}"

# Add project to PYTHONPATH
export PYTHONPATH="$CESGA_PROJECT_ROOT:$PYTHONPATH"

# Load required modules (adjust for your CESGA environment)
module purge
module load cesga/2020
module load python/3.10.8

# Activate virtual environment if exists
if [ -f "$CESGA_PROJECT_ROOT/venv/bin/activate" ]; then
    source "$CESGA_PROJECT_ROOT/venv/bin/activate"
    echo "Virtual environment activated"
fi

# Check arguments
if [ $# -lt 1 ]; then
    echo "ERROR: Strategy name required"
    echo "Usage: sbatch run_walkforward.sbatch <strategy> [options]"
    exit 1
fi

STRATEGY=$1
shift  # Remove first argument, pass rest to Python script

echo ""
echo "Running Walk-Forward MCPT for strategy: $STRATEGY"
echo "Additional arguments: $@"
echo ""

# Change to project directory
cd "$CESGA_PROJECT_ROOT"

# Run the analysis
python scripts/cesga/walkforward_permutation.py "$STRATEGY" \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    "$@"

# Print completion information
echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "=============================================="
