#!/bin/bash
#SBATCH --job-name=hpml_lab4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:4  # Request 4 GPUs (e.g., RTX8000) on NYU HPC
#SBATCH --mem=32GB
#SBATCH --time=02:00:00       # Max wall time
#SBATCH --output=slurm_%j.out # Standard output and error log

# Load your Python environment on HPC here
# module purge
# module load anaconda3/2020.11
# source activate /path/to/your/env

echo "=================================="
echo "Starting Lab 4 DDP Experiments"
echo "=================================="

# Q1 Single GPU tests
echo "Running Q1: Single GPU Batch Size Sweep"
python lab4_ddp.py --run q1 --batch_size 128
python lab4_ddp.py --run q1 --batch_size 512

# Q2 Speedup Measurement tests
echo "Running Q2: DDP Speedup Measurement on 4 GPUs"
torchrun --nproc_per_node=4 lab4_ddp.py --run q2 --batch_size 128

# Q3 Computation vs Communication overhead
echo "Running Q3: Computation vs. Communication on 4 GPUs"
torchrun --nproc_per_node=4 lab4_ddp.py --run q3 --batch_size 128

# Q4 Large batch long training
echo "Running Q4: Large Batch Training for 5 Epochs"
torchrun --nproc_per_node=4 lab4_ddp.py --run q4 --batch_size 512 --epochs 5

echo "All Done!"
