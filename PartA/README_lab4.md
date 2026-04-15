# ECE-GY 9143 Lab 4 Part A: Distributed Deep Learning

## Files
- `lab4_ddp.py` — Main script for Q1–Q4 experiments (DDP-based).
- `lab4_experiments.ipynb` — Notebook runner for cloud/HPC execution.

## Environment Requirements
- Python 3.8+, PyTorch 2.0+ with CUDA, torchvision.
- 1 GPU for Q1, 2/4 GPUs for Q2/Q3, and 4 GPUs for Q4.
- Keep GPU type consistent across all experiments.

Quick sanity check:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

## Important CLI Note
Use positional run modes (`q1/q2/q3/q4`) with `torchrun`:
- Good: `torchrun ... lab4_ddp.py q2 --batch_size 128`
- Avoid: `torchrun ... lab4_ddp.py --run q2 ...` (some `torchrun` versions parse `--run` incorrectly).

## Common Setup
```bash
PROJECT_DIR=${PROJECT_DIR:-$PWD}
DATA_PATH=${DATA_PATH:-/workspace/data/cifar10}
LOG_DIR=${LOG_DIR:-$PROJECT_DIR/logs}
mkdir -p "$DATA_PATH" "$LOG_DIR"
cd "$PROJECT_DIR"
```

## Q1: Computational Efficiency vs Batch Size (Single GPU)
Run 2 epochs per batch size and report the 2nd-epoch `train_time` (excluding data loading). Increase by 4x until OOM.

```bash
for bs in 32 128 512 2048 8192 32768; do
  echo "=== Q1 | batch_size=${bs} ==="
  if ! python3 lab4_ddp.py q1 --batch_size "$bs" --data_path "$DATA_PATH" 2>&1 | tee "$LOG_DIR/q1_bs${bs}.log"; then
    echo "Q1 stopped at batch_size=${bs} (likely OOM). Use the previous successful batch size as MAX_BS."
    break
  fi
done
```

## Q2: Speedup Measurement (Multi-GPU DDP)
Main table runs (32/128/512), 2 epochs each, report 2nd-epoch `total_time`.

```bash
for bs in 32 128 512; do
  torchrun --nproc_per_node=2 lab4_ddp.py q2 --batch_size "$bs" --data_path "$DATA_PATH" 2>&1 | tee "$LOG_DIR/q2_g2_bs${bs}.log"
  torchrun --nproc_per_node=4 lab4_ddp.py q2 --batch_size "$bs" --data_path "$DATA_PATH" 2>&1 | tee "$LOG_DIR/q2_g4_bs${bs}.log"
done
```

Optional supplementary run at `MAX_BS`:
```bash
MAX_BS=8192
torchrun --nproc_per_node=2 lab4_ddp.py q2 --batch_size "$MAX_BS" --data_path "$DATA_PATH" 2>&1 | tee "$LOG_DIR/q2_g2_bs${MAX_BS}.log"
torchrun --nproc_per_node=4 lab4_ddp.py q2 --batch_size "$MAX_BS" --data_path "$DATA_PATH" 2>&1 | tee "$LOG_DIR/q2_g4_bs${MAX_BS}.log"
```

## Q3: Computation vs Communication (Multi-GPU DDP)
Main table runs (32/128/512), 2 epochs each, report 2nd-epoch `train_time` and `allreduce` metadata.

```bash
for bs in 32 128 512; do
  torchrun --nproc_per_node=2 lab4_ddp.py q3 --batch_size "$bs" --data_path "$DATA_PATH" 2>&1 | tee "$LOG_DIR/q3_g2_bs${bs}.log"
  torchrun --nproc_per_node=4 lab4_ddp.py q3 --batch_size "$bs" --data_path "$DATA_PATH" 2>&1 | tee "$LOG_DIR/q3_g4_bs${bs}.log"
done
```

Optional supplementary run at `MAX_BS`:
```bash
MAX_BS=8192
torchrun --nproc_per_node=2 lab4_ddp.py q3 --batch_size "$MAX_BS" --data_path "$DATA_PATH" 2>&1 | tee "$LOG_DIR/q3_g2_bs${MAX_BS}.log"
torchrun --nproc_per_node=4 lab4_ddp.py q3 --batch_size "$MAX_BS" --data_path "$DATA_PATH" 2>&1 | tee "$LOG_DIR/q3_g4_bs${MAX_BS}.log"
```

For Q3.2, `lab4_ddp.py` prints:
- Model parameter count.
- Model size in bytes/GB.
- AllReduce volume using `2*(N-1)/N*model_size`.

## Q4: Large Batch Training (4 GPUs, 5 epochs)
Use the largest successful per-GPU batch from Q1.

```bash
MAX_BS=8192
torchrun --nproc_per_node=4 lab4_ddp.py q4 --batch_size "$MAX_BS" --epochs 5 --data_path "$DATA_PATH" 2>&1 | tee "$LOG_DIR/q4_bs${MAX_BS}.log"
```

## Extract Key Results from Logs
```bash
grep -h "Q1 RESULT" logs/q1_*.log
grep -h "Q2 RESULT" logs/q2_*.log
grep -h "Q3 RESULT" logs/q3_*.log
grep -h "DDP train_time" logs/q3_*.log
grep -h "Q4.1 RESULT" logs/q4_*.log
```

## Hyper-parameters (Lab 2 defaults)
- Optimizer: SGD
- Learning rate: 0.1
- Momentum: 0.9
- Weight decay: 5e-4
- DataLoader workers: 2
