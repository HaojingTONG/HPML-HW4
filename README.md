# ECE-GY 9143: High Performance Machine Learning - Lab 4 & Homework 4

This repository contains the implementations and reports for Lab 4 and Homework 4, focusing on **Distributed Deep Learning** and **Post-Training Quantization**.

## Directory Structure

- **`PartA/`**: Distributed Deep Learning
  - Explores data parallelism using PyTorch's `DistributedDataParallel` (DDP).
  - Benchmarks computational efficiency, multi-GPU speedups, and the balance between computation and communication overhead.
  - Investigates the challenges of large-batch training and its impact on accuracy.
  - *Refer to `PartA/README_lab4.md` for detailed distributed execution steps.*

- **`PartB/`**: Post-Training Quantization
  - Provides a deep dive into converting floating-point CNN inference to low-precision integer arithmetic.
  - Implements $3\sigma$ quantization strategies for $INT8$ Weights, $INT8$ Activations, and $INT32$ Biases.
  - Validates accuracy differences across stages of quantization.
  
- **`report.tex`**: The combined technical report covering both Part A and Part B.
- **`HPML_Lab_4.pdf`**: The assignment problem descriptions.

## Building the Report

To generate the final PDF report from the LaTeX source, run your preferred LaTeX engine (like `pdflatex`) from the root `lab4` folder. All image paths are configured relatively so they compile seamlessly:

```bash
pdflatex report.tex
```

## Environment Requirements

- **Frameworks:** Python 3.8+, PyTorch 2.0+ with CUDA, `torchvision`
- **Hardware:**
  - Part A requires a multi-GPU environment (e.g., NVIDIA A100, up to 4 GPUs) to produce the DDP scaling benchmarks.
  - Part B can run on standard CPU environments as it evaluates the inference accuracy in PyTorch after simulated quantization.

## Author
**Haojingtong Tong**
