#!/bin/bash
#SBATCH --job-name=batch_walkforward
#SBATCH --output=logs/slurm/walkforward_%j.out
#SBATCH --error=logs/slurm/walkforward_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

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
STRATEGY=${STRATEGY:-hawkes}
PERMS=${PERMS:-1000}
WORKERS=${WORKERS:-16}

# Directorio del proyecto
export TRADING_ROOT="${HOME}/Trading-system"
cd $TRADING_ROOT

# Cargar módulos (ajustar según configuración de CESGA)
module purge
module load python/3.11

# Activar entorno virtual
if [ -d "${HOME}/trading_env" ]; then
    source ${HOME}/trading_env/bin/activate
elif [ -d "${TRADING_ROOT}/venv" ]; then
    source ${TRADING_ROOT}/venv/bin/activate
fi

# Crear directorio de logs si no existe
mkdir -p logs/slurm

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
