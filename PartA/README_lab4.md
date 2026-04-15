# ECE-GY 9143 Lab 4 Part A: Distributed Deep Learning

## Files
- `lab4_ddp.py` — Main script for Q1–Q4 experiments (DDP-based)

## Requirements
- Python 3.8+, PyTorch 2.0+ with CUDA, torchvision
- Multi-GPU node (up to 4 GPUs of the same type)

## Commands for Each Experiment

### Q1: Computational Efficiency w.r.t Batch Size (Single GPU)
Run 2 epochs per batch size on single GPU. Reports 2nd epoch training time (excluding data I/O).

```bash
python lab4_ddp.py --run q1 --batch_size 32
python lab4_ddp.py --run q1 --batch_size 128
python lab4_ddp.py --run q1 --batch_size 512
python lab4_ddp.py --run q1 --batch_size 2048
# Keep increasing (4x) until GPU OOM
```

### Q2: Speedup Measurement (Multi-GPU DDP)
Run 2 epochs on 2/4 GPUs. Reports 2nd epoch total time (including all components).

```bash
# 2-GPU runs
torchrun --nproc_per_node=2 lab4_ddp.py --run q2 --batch_size 32
torchrun --nproc_per_node=2 lab4_ddp.py --run q2 --batch_size 128
torchrun --nproc_per_node=2 lab4_ddp.py --run q2 --batch_size 512

# 4-GPU runs
torchrun --nproc_per_node=4 lab4_ddp.py --run q2 --batch_size 32
torchrun --nproc_per_node=4 lab4_ddp.py --run q2 --batch_size 128
torchrun --nproc_per_node=4 lab4_ddp.py --run q2 --batch_size 512
```

### Q3: Computation vs Communication (Multi-GPU DDP)
Same as Q2 but prints additional info for bandwidth calculations.

```bash
# 2-GPU runs
torchrun --nproc_per_node=2 lab4_ddp.py --run q3 --batch_size 32
torchrun --nproc_per_node=2 lab4_ddp.py --run q3 --batch_size 128
torchrun --nproc_per_node=2 lab4_ddp.py --run q3 --batch_size 512

# 4-GPU runs
torchrun --nproc_per_node=4 lab4_ddp.py --run q3 --batch_size 32
torchrun --nproc_per_node=4 lab4_ddp.py --run q3 --batch_size 128
torchrun --nproc_per_node=4 lab4_ddp.py --run q3 --batch_size 512
```

**How to calculate compute vs comm time:**
- Compute time = Q1 single-GPU train_time for same batch_size
- Comm time = Q3 DDP train_time − Q1 single-GPU train_time

### Q4: Large Batch Training (4 GPUs, 5 epochs)
Use the largest batch size found in Q1.

```bash
torchrun --nproc_per_node=4 lab4_ddp.py --run q4 --batch_size 512 --epochs 5
```

## Hyper-parameters (all defaults from Lab 2)
- Optimizer: SGD
- Learning rate: 0.1
- Momentum: 0.9
- Weight decay: 5e-4
- DataLoader workers: 2

## Notes
- Q1 runs as a normal Python script (single GPU)
- Q2/Q3/Q4 must be launched with `torchrun` for DDP
- All experiments use 2 num_workers as required
- For NYU HPC: request a multi-GPU node with `--gres=gpu:4`
