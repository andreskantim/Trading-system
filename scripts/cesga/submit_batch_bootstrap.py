#!/usr/bin/env python3
"""
SLURM Job Submitter for Batch Bootstrap Analysis

Generates and submits SLURM jobs for batch bootstrap analysis.

Usage:
    python submit_batch_bootstrap.py --group crypto_10 --strategy hawkes --bootstrap-type circular_block
    python submit_batch_bootstrap.py --group crypto_25 --strategy donchian --bootstrap-type trade_based --partition gpu
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
import os

if 'TRADING_ROOT' in os.environ:
    project_root = Path(os.environ['TRADING_ROOT'])
else:
    project_root = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(project_root))

from config.tickers import TICKER_GROUPS


def generate_slurm_script(args) -> str:
    """Generate SLURM batch script content."""

    job_name = f"boot_{args.group}_{args.strategy}_{args.bootstrap_type}"
    log_dir = project_root / "logs" / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}

# Load modules
module load cesga/2020
module load python/3.9.1

# Activate virtual environment if exists
if [ -d "{project_root}/venv" ]; then
    source {project_root}/venv/bin/activate
fi

# Set environment
export TRADING_ROOT="{project_root}"
export PYTHONPATH="{project_root}:$PYTHONPATH"

# Run batch bootstrap
cd {project_root}
python scripts/cesga/batch_bootstrap.py \\
    --group {args.group} \\
    --strategy {args.strategy} \\
    --bootstrap-type {args.bootstrap_type} \\
    --n-iterations {args.n_iterations} \\
    --block-size {args.block_size}

echo "Job completed at $(date)"
"""
    return script


def main():
    parser = argparse.ArgumentParser(description='Submit SLURM job for batch bootstrap')

    parser.add_argument('--group', type=str, required=True, choices=list(TICKER_GROUPS.keys()))
    parser.add_argument('--strategy', type=str, required=True)
    parser.add_argument('--bootstrap-type', type=str, required=True,
                        choices=['circular_block', 'stationary', 'trade_based'])
    parser.add_argument('--n-iterations', type=int, default=1000)
    parser.add_argument('--block-size', type=int, default=20)

    # SLURM options
    parser.add_argument('--partition', type=str, default='shared', help='SLURM partition')
    parser.add_argument('--cpus', type=int, default=16, help='CPUs per task')
    parser.add_argument('--mem', type=str, default='32G', help='Memory')
    parser.add_argument('--time', type=str, default='12:00:00', help='Time limit')
    parser.add_argument('--dry-run', action='store_true', help='Print script without submitting')

    args = parser.parse_args()

    script_content = generate_slurm_script(args)

    if args.dry_run:
        print("=" * 70)
        print("SLURM SCRIPT (dry-run)")
        print("=" * 70)
        print(script_content)
        return

    # Write and submit
    script_dir = project_root / "scripts" / "cesga" / "slurm_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_file = script_dir / f"boot_{args.group}_{args.bootstrap_type}_{timestamp}.sh"

    with open(script_file, 'w') as f:
        f.write(script_content)

    print(f"Script written: {script_file}")
    print("Submitting job...")

    result = subprocess.run(['sbatch', str(script_file)], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Job submitted: {result.stdout.strip()}")
    else:
        print(f"Submission failed: {result.stderr}")


if __name__ == '__main__':
    main()
