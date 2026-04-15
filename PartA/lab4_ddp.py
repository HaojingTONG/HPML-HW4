"""
ECE-GY 9143 - High Performance Machine Learning
Lab 4 Part A: Distributed Deep Learning with DistributedDataParallel

Based on Lab 2 code. Uses default SGD solver with default hyper-parameters
(lr=0.1, momentum=0.9, weight_decay=5e-4) and 2 num_workers.

Usage (single GPU - Q1):
    python lab4_ddp.py --run q1 --batch_size 32 --device cuda

Usage (multi-GPU - Q2/Q3/Q4):
    torchrun --nproc_per_node=2 lab4_ddp.py --run q2 --batch_size 32
    torchrun --nproc_per_node=4 lab4_ddp.py --run q2 --batch_size 32
    torchrun --nproc_per_node=4 lab4_ddp.py --run q4 --batch_size 512 --epochs 5
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as T


# =========================================================================
# Model: ResNet-18 for CIFAR-10  (identical to lab2.py)
# =========================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_bn: bool = True):
        super().__init__()
        bn = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = bn(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = bn(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                bn(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, use_bn: bool = True):
        super().__init__()
        self.in_ch = 64
        bn = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = bn(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1, use_bn=use_bn)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2, use_bn=use_bn)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2, use_bn=use_bn)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2, use_bn=use_bn)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_ch: int, num_blocks: int, stride: int, use_bn: bool) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.append(BasicBlock(self.in_ch, out_ch, stride=stride, use_bn=use_bn))
        self.in_ch = out_ch
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_ch, out_ch, stride=1, use_bn=use_bn))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =========================================================================
# Data
# =========================================================================
def build_train_dataset(data_path: str):
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])
    return torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)


def build_train_loader(
    data_path: str,
    batch_size: int,
    num_workers: int,
    sampler=None,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_train_dataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
    )


def build_train_loader_ddp(
    data_path: str,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DistributedSampler]:
    """Build train loader with DistributedSampler for DDP."""
    dataset = build_train_dataset(data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # sampler handles shuffling
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
    )
    return loader, sampler


# =========================================================================
# Utilities
# =========================================================================
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@dataclass
class EpochStats:
    data_time: float
    train_time: float
    total_time: float
    avg_loss: float
    avg_top1: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int = 200,
    rank: int = 0,
) -> EpochStats:
    """
    Train for one epoch, measuring:
      - data_time: time spent loading data from DataLoader
      - train_time: time for H2D transfer + forward + backward + optimizer step
      - total_time: wall-clock time for the full epoch
    """
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    data_time = 0.0
    train_time = 0.0

    t_total_start = time.perf_counter()

    it = iter(loader)
    num_batches = len(loader)

    for b in range(num_batches):
        # --- Data loading time (excluding H2D transfer) ---
        t0 = time.perf_counter()
        images, targets = next(it)
        t1 = time.perf_counter()
        data_time += (t1 - t0)

        # --- H2D transfer ---
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # --- Training time (H2D sync + compute + comm) ---
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        train_time += (t3 - t2)

        # --- Metrics ---
        bs = targets.size(0)
        acc = top1_accuracy(logits.detach(), targets)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_samples += bs

        if rank == 0 and ((b + 1) % log_interval == 0 or (b + 1) == num_batches):
            print(f"  batch {b+1:4d}/{num_batches} | loss {loss.item():.4f} | top1 {acc*100:.2f}%")

    t_total_end = time.perf_counter()
    total_time = t_total_end - t_total_start

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_top1 = total_acc / total_samples if total_samples > 0 else 0.0

    return EpochStats(
        data_time=data_time,
        train_time=train_time,
        total_time=total_time,
        avg_loss=avg_loss,
        avg_top1=avg_top1,
    )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters (useful for Q3.2 bandwidth calc)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================================
# DDP Setup / Cleanup
# =========================================================================
def setup_ddp():
    """Initialize DDP process group. Called by torchrun."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    """Destroy DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =========================================================================
# Q1: Computational Efficiency w.r.t Batch Size (single GPU)
# =========================================================================
def run_q1(args):
    """
    Q1: Measure training time (excluding data I/O) for 1 epoch on single GPU.
    Run 2 epochs; 1st epoch = warmup, report 2nd epoch training time.
    """
    device = torch.device(args.device)
    print("=" * 60)
    print("Q1: Computational Efficiency w.r.t Batch Size (Single GPU)")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    model = ResNet18CIFAR(num_classes=10, use_bn=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    loader = build_train_loader(args.data_path, args.batch_size, args.num_workers, shuffle=True)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Number of batches per epoch: {len(loader)}")

    for epoch in range(1, 3):
        print(f"\n--- Epoch {epoch} {'(warmup)' if epoch == 1 else '(measured)'} ---")
        stats = train_one_epoch(model, loader, optimizer, criterion, device, log_interval=200)
        print(f"  Data loading time : {stats.data_time:.3f}s")
        print(f"  Training time     : {stats.train_time:.3f}s  <-- Q1 metric (excl data I/O)")
        print(f"  Total epoch time  : {stats.total_time:.3f}s")
        print(f"  Avg loss          : {stats.avg_loss:.4f}")
        print(f"  Avg top-1 acc     : {stats.avg_top1*100:.2f}%")

        if epoch == 2:
            print(f"\n>>> Q1 RESULT: batch_size={args.batch_size}, "
                  f"train_time(epoch2)={stats.train_time:.3f}s (excluding data I/O)")


# =========================================================================
# Q2: Speedup Measurement (multi-GPU DDP)
# =========================================================================
def run_q2(args):
    """
    Q2: Measure running time on multi-GPU (DDP) and calculate speedup.
    Includes ALL training components (data loading, H2D, compute, comm).
    Run 2 epochs; report 2nd epoch.
    """
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("=" * 60)
        print("Q2: Speedup Measurement (DDP)")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"World size (num GPUs): {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print("=" * 60)

    model = ResNet18CIFAR(num_classes=10, use_bn=True).to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    loader, sampler = build_train_loader_ddp(
        args.data_path, args.batch_size, args.num_workers, rank, world_size
    )

    if rank == 0:
        print(f"Batches per GPU per epoch: {len(loader)}")

    for epoch in range(1, 3):
        sampler.set_epoch(epoch)  # ensure proper shuffling each epoch

        if rank == 0:
            print(f"\n--- Epoch {epoch} {'(warmup)' if epoch == 1 else '(measured)'} ---")

        stats = train_one_epoch(model, loader, optimizer, criterion, device, log_interval=200, rank=rank)

        if rank == 0:
            print(f"  Data loading time : {stats.data_time:.3f}s")
            print(f"  Training time     : {stats.train_time:.3f}s")
            print(f"  Total epoch time  : {stats.total_time:.3f}s  <-- Q2 metric (incl all)")
            print(f"  Avg loss          : {stats.avg_loss:.4f}")
            print(f"  Avg top-1 acc     : {stats.avg_top1*100:.2f}%")

            if epoch == 2:
                print(f"\n>>> Q2 RESULT: {world_size}-GPU, batch_size_per_gpu={args.batch_size}, "
                      f"total_time(epoch2)={stats.total_time:.3f}s")

    cleanup_ddp()


# =========================================================================
# Q3: Computation vs Communication (multi-GPU DDP)
# =========================================================================
def run_q3(args):
    """
    Q3.1: Report compute time and communication time for each setup.
    Strategy:
      - Compute time per GPU ≈ single-GPU train_time (from Q1, same batch size)
      - Communication time ≈ DDP_train_time - single_GPU_train_time

    This run measures the DDP training time. Compare with Q1 results to get comm time.
    Run 2 epochs; report 2nd epoch.
    """
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("=" * 60)
        print("Q3: Computation vs Communication (DDP)")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print("=" * 60)

    model = ResNet18CIFAR(num_classes=10, use_bn=True).to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    loader, sampler = build_train_loader_ddp(
        args.data_path, args.batch_size, args.num_workers, rank, world_size
    )

    num_params = count_parameters(model)

    for epoch in range(1, 3):
        sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n--- Epoch {epoch} {'(warmup)' if epoch == 1 else '(measured)'} ---")

        stats = train_one_epoch(model, loader, optimizer, criterion, device, log_interval=200, rank=rank)

        if rank == 0:
            print(f"  Train time (compute+comm): {stats.train_time:.3f}s")
            print(f"  Total epoch time          : {stats.total_time:.3f}s")

            if epoch == 2:
                print(f"\n>>> Q3 RESULT: {world_size}-GPU, bs_per_gpu={args.batch_size}")
                print(f"    DDP train_time(epoch2) = {stats.train_time:.3f}s")
                print(f"    (Subtract Q1 single-GPU train_time for same batch_size to get comm_time)")
                print(f"    comm_time = DDP_train_time - single_GPU_train_time")
                print(f"\n    For Q3.2 bandwidth utilization:")
                print(f"    Model params = {num_params:,}")
                model_size_bytes = num_params * 4  # float32 = 4 bytes
                model_size_gb = model_size_bytes / (1024**3)
                print(f"    Model size   = {model_size_bytes:,} bytes ({model_size_gb:.4f} GB)")
                print(f"    AllReduce data volume = 2 * (N-1)/N * model_size")
                allreduce_bytes = 2 * (world_size - 1) / world_size * model_size_bytes
                allreduce_gb = allreduce_bytes / (1024**3)
                print(f"    AllReduce volume = {allreduce_gb:.4f} GB (for {world_size} GPUs)")

    cleanup_ddp()


# =========================================================================
# Q4: Large Batch Training (multi-GPU DDP)
# =========================================================================
def run_q4(args):
    """
    Q4.1: Report avg training loss and accuracy for the 5th epoch
    using largest batch size per GPU on 4 GPUs.
    Compare with Lab 2 results (batch_size=128, single GPU).
    """
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("=" * 60)
        print("Q4: Large Batch Training (DDP)")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Epochs: {args.epochs}")
        print("=" * 60)

    model = ResNet18CIFAR(num_classes=10, use_bn=True).to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    loader, sampler = build_train_loader_ddp(
        args.data_path, args.batch_size, args.num_workers, rank, world_size
    )

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        stats = train_one_epoch(model, loader, optimizer, criterion, device, log_interval=200, rank=rank)

        # Gather loss and accuracy across all ranks for accurate reporting
        loss_tensor = torch.tensor([stats.avg_loss], device=device)
        acc_tensor = torch.tensor([stats.avg_top1], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = loss_tensor.item() / world_size
        global_avg_acc = acc_tensor.item() / world_size

        if rank == 0:
            print(f"  Avg loss  : {global_avg_loss:.4f}")
            print(f"  Avg top-1 : {global_avg_acc*100:.2f}%")
            print(f"  Train time: {stats.train_time:.3f}s")

            if epoch == args.epochs:
                print(f"\n>>> Q4.1 RESULT:")
                print(f"    Effective batch size = {args.batch_size * world_size}")
                print(f"    Epoch {epoch} avg loss     = {global_avg_loss:.4f}")
                print(f"    Epoch {epoch} avg accuracy = {global_avg_acc*100:.2f}%")
                print(f"    (Compare with Lab 2: batch_size=128, single GPU)")

    cleanup_ddp()


# =========================================================================
# Main
# =========================================================================
def main():
    p = argparse.ArgumentParser(description="ECE-GY 9143 Lab 4 Part A: Distributed Deep Learning")
    p.add_argument(
        "run_pos",
        nargs="?",
        choices=["q1", "q2", "q3", "q4"],
        help="Experiment mode as positional arg (q1/q2/q3/q4). Useful with some torchrun versions.",
    )
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--device", type=str, default="cuda", help="Device for single-GPU runs (Q1)")
    p.add_argument("--epochs", type=int, default=2, help="Number of epochs (default 2 for Q1-Q3, set 5 for Q4)")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (lab requirement: 2)")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--run", type=str, default=None,
                   choices=["q1", "q2", "q3", "q4"],
                   help="Which experiment: q1 (single-GPU timing) | q2 (speedup) | q3 (compute vs comm) | q4 (large batch)")

    args = p.parse_args()
    run_mode = args.run if args.run is not None else (args.run_pos if args.run_pos is not None else "q1")

    if run_mode == "q1":
        run_q1(args)
    elif run_mode == "q2":
        run_q2(args)
    elif run_mode == "q3":
        run_q3(args)
    elif run_mode == "q4":
        run_q4(args)


if __name__ == "__main__":
    main()
