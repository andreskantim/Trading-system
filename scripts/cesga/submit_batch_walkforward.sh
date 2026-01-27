#!/bin/bash
#SBATCH --job-name=batch_walkforward
#SBATCH --output=logs/slurm/walkforward_%j.out
#SBATCH --error=logs/slurm/walkforward_%j.err
#SBATCH -t 04:00:00
#SBATCH -c 64
#SBATCH --mem=32GB

# ============================================================================
# SLURM Script para Batch Walk-Forward Backtest en CESGA
# ============================================================================
#
# Uso:
#   sbatch submit_batch_walkforward.sh
#
# O con parámetros personalizados:
#   sbatch --export=GROUP=crypto_25,STRATEGY=donchian,PERMS=500 submit_batch_walkforward.sh
#
# ============================================================================

# Configuración por defecto (puede sobrescribirse con --export)
GROUP=${GROUP:-crypto_10}
STRATEGY=${STRATEGY:-bollinger_b2b}
PERMS=${PERMS:-200}
WORKERS=${WORKERS:-64}

# Directorio del proyecto
export TRADING_ROOT="${HOME}/Trading-system"
cd $TRADING_ROOT

# Cargar solo el módulo base de CESGA (NO miniconda del sistema)
module purge
module load cesga/2020

# Usar $STORE para conda (variable de entorno de CESGA)
source ${STORE}/miniconda3/etc/profile.d/conda.sh
conda activate trading_env

# Verificar que el entorno está activo
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
python -c "import numpy, pandas; print('✓ Dependencias disponibles')"

# Información del job
echo "=============================================="
echo "SLURM Job: $SLURM_JOB_ID"
echo "Nodo: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memoria: $SLURM_MEM_PER_NODE MB"
echo "=============================================="
echo "Grupo: $GROUP"
echo "Estrategia: $STRATEGY"
echo "Permutaciones: $PERMS"
echo "Workers: $WORKERS"
echo "=============================================="
echo ""

# Ejecutar batch
python scripts/cesga/batch_walkforward.py \
    --group $GROUP \
    --strategy $STRATEGY \
    --n-permutations $PERMS \
    --n-workers $WORKERS

# Mostrar estado final
echo ""
echo "=============================================="
echo "Job completado: $(date)"
echo "=============================================="
