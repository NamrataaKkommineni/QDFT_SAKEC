#!/bin/bash
#SBATCH --job-name=H2_22LDA
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00

echo "=========================================="
echo "SLURM_SUBMIT_DIR = $SLURM_SUBMIT_DIR"
echo "SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"
echo "SLURM_NTASKS = $SLURM_NTASKS"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Load cluster miniconda
module load miniconda3

# Activate YOUR prefix environment
conda activate /home/samritm/shreyas/envs/cpu_env

# Debug info (very important)
echo "Python path:"
which python
python -c "import sys; print(sys.executable)"

# Optional: CPU info
lscpu | grep "Model name"

# Run embedding
time python -u H2_LDA.py

echo "Job finished at: $(date)"