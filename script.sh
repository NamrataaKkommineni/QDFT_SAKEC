#!/bin/bash
#SBATCH --job-name=Tetracene_B3LYP_26
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
echo "SLURM_SUBMIT_DIR = $SLURM_SUBMIT_DIR"
echo "SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"
echo "SLURM_NTASKS = $SLURM_NTASKS"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

module load miniconda3
conda activate /home/samritm/tejjan/envs/cpu_env
echo "Python path:"
which python
python -c "import sys; print(sys.executable)"
lscpu | grep "Model name"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

time python -u "${SLURM_JOB_NAME}.py"

echo "Job finished at: $(date)"