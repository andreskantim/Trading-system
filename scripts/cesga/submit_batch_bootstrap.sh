#!/bin/bash
#SBATCH --job-name=batch_bootstrap
#SBATCH --output=logs/slurm/bootstrap_%j.out
#SBATCH --error=logs/slurm/bootstrap_%j.err
#SBATCH -t 04:00:00
#SBATCH -c 64
#SBATCH --mem=32GB

# ============================================================================
# SLURM Script para Batch Bootstrap Analysis en CESGA
# ============================================================================
#
# Uso:
#   sbatch submit_batch_bootstrap.sh
#
# O con parámetros personalizados:
#   sbatch --export=GROUP=crypto_25,STRATEGY=hawkes,BOOT_TYPE=circular_block submit_batch_bootstrap.sh
#   sbatch --export=GROUP=crypto_10,STRATEGY=donchian,BOOT_TYPE=stationary,ITERS=2000 submit_batch_bootstrap.sh
#   sbatch --export=GROUP=crypto_25,STRATEGY=hawkes,BOOT_TYPE=trade_based submit_batch_bootstrap.sh
#
# Bootstrap types: circular_block, stationary, trade_based
# ============================================================================

# Configuración por defecto (puede sobrescribirse con --export)
GROUP=${GROUP:-crypto_10}
STRATEGY=${STRATEGY:-hawkes}
BOOT_TYPE=${BOOT_TYPE:-circular_block}
ITERS=${ITERS:-1000}
BLOCK_SIZE=${BLOCK_SIZE:-20}

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
python -c "import numpy, pandas; print('Dependencias disponibles')"

# Información del job
echo "=============================================="
echo "SLURM Job: $SLURM_JOB_ID"
echo "Nodo: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memoria: $SLURM_MEM_PER_NODE MB"
echo "=============================================="
echo "Grupo: $GROUP"
echo "Estrategia: $STRATEGY"
echo "Bootstrap Type: $BOOT_TYPE"
echo "Iteraciones: $ITERS"
echo "Block Size: $BLOCK_SIZE"
echo "=============================================="
echo ""

# Ejecutar batch bootstrap
python scripts/cesga/batch_bootstrap.py \
    --group $GROUP \
    --strategy $STRATEGY \
    --bootstrap-type $BOOT_TYPE \
    --n-iterations $ITERS \
    --block-size $BLOCK_SIZE

# Mostrar estado final
echo ""
echo "=============================================="
echo "Job completado: $(date)"
echo "=============================================="
