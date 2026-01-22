# CESGA Cluster Scripts

Scripts adapted for running on the CESGA Finisterrae III supercomputer using SLURM job scheduler.

## Setup

### 1. Configure Project Path

Edit the CESGA project root path in the sbatch scripts or set environment variables:

```bash
export CESGA_PROJECT_ROOT=/mnt/lustre/scratch/nlsas/home/usc/ec/ahe/Trading-system
export CESGA_DATA_DIR=$CESGA_PROJECT_ROOT/data
export CESGA_OUTPUT_DIR=$CESGA_PROJECT_ROOT/outputs
```

### 2. Create Virtual Environment (first time only)

```bash
module load python/3.10.8
python -m venv $CESGA_PROJECT_ROOT/venv
source $CESGA_PROJECT_ROOT/venv/bin/activate
pip install -r requirements.txt
```

### 3. Transfer Data

Copy your data files to the cluster:

```bash
rsync -avz data/ cesga:$CESGA_PROJECT_ROOT/data/
```

## Usage

### In-Sample MCPT Analysis

```bash
# Basic usage
sbatch run_insample.sbatch hawkes

# With custom parameters
sbatch run_insample.sbatch hawkes --n-permutations 2000 --ticker BTCUSD

# Full options
sbatch run_insample.sbatch hawkes \
    --ticker BTCUSD \
    --train-start 2016-01-01 \
    --train-end 2020-01-01 \
    --n-permutations 1000
```

### Walk-Forward MCPT Analysis

```bash
# Basic usage
sbatch run_walkforward.sbatch hawkes

# With custom parameters
sbatch run_walkforward.sbatch hawkes --n-permutations 500 --train-window-years 3

# Full options
sbatch run_walkforward.sbatch hawkes \
    --ticker BTCUSD \
    --data-start 2018-01-01 \
    --data-end 2024-01-01 \
    --train-window-years 4.0 \
    --n-permutations 200
```

## Job Management

```bash
# Check job status
squeue -u $USER

# View detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# View output in real-time
tail -f logs/insample_<job_id>.out

# View all your recent jobs
sacct -u $USER --starttime=2024-01-01 --format=JobID,JobName,State,Elapsed,MaxRSS
```

## Resource Configuration

The default SLURM configuration uses:
- **Partition**: shared
- **Nodes**: 1
- **CPUs per task**: 32
- **Memory**: 64GB
- **Time limit**: 24h (in-sample) / 48h (walk-forward)

To request different resources, edit the `#SBATCH` directives in the sbatch files or override on command line:

```bash
sbatch --cpus-per-task=64 --mem=128G run_insample.sbatch hawkes
```

## Output Files

Results are saved to:
```
$CESGA_OUTPUT_DIR/backtest/figures/<strategy>/
├── insample_mcpt.png
├── insample_cumulative_mcpt.png
├── insample_results_<job_id>.csv
├── walkforward_mcpt.png
├── walkforward_cumulative_mcpt.png
└── walkforward_results_<job_id>.csv
```

## Troubleshooting

### Module not found
Make sure PYTHONPATH includes the project root:
```bash
export PYTHONPATH=$CESGA_PROJECT_ROOT:$PYTHONPATH
```

### Out of memory
Increase memory allocation:
```bash
sbatch --mem=128G run_insample.sbatch hawkes
```

### Job killed due to time limit
Increase time limit:
```bash
sbatch --time=72:00:00 run_walkforward.sbatch hawkes
```
