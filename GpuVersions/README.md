# Quantum Embedding Benchmarking Framework

This repository documents the development, optimization, and benchmarking of a **Hybrid Quantum-Classical Embedding Framework** implemented using **PennyLane**, **Qiskit Nature**, and GPU-accelerated quantum simulation backends.

The project investigates the practical performance of **Variational Quantum Eigensolver (VQE)** workflows for quantum embedding calculations, focusing on:

- Ansatz construction strategies
- Fermion-to-qubit mapping techniques
- Analytical versus numerical gradient evaluation
- GPU acceleration
- JAX Just-In-Time (JIT) compilation
- Cross-framework validation between PennyLane and Qiskit

The repository captures the complete evolution of the embedding framework—from an initial proof-of-concept implementation to production-ready GPU workflows and architectural optimization studies.

---

# Repository Structure

```text
GpuVersions/
│
├── PennyLane Version 1/
│   ├── PLgpu_v1.py
│   ├── *.csv
│   ├── *.out
│   └── *.err
│
├── PennyLane Version 2/
│   │
│   ├── UCCSD/
│   │   ├── Modified v2/
│   │   ├── Result Files/
│   │   ├── PLcpu_v2.py
│   │   ├── PLgpu_v2.py
│   │   ├── PLcpu_v2_spin.py
│   │   ├── PLgpu_v2_spin.py
│   │   ├── Qkit_v2.py
│   │   └── Qkit_v2_spin.py
│   │
│   └── qUCCSD/
│       ├── Result Files/
│       ├── PLcpu_qUCCSD_spin.py
│       ├── PLgpu_qUCCSD_spin.py
│       └── Qkit_qUCCSD_spin.py
│
├── PennyLane Version 3/
│   ├── PLgpu_v3.py
│   ├── *.csv
│   ├── *.out
│   └── *.err
│
└── README.md
```

---

# Project Objectives

The primary objectives of this framework are to:

- Develop scalable VQE-based embedding workflows.
- Evaluate GPU acceleration for quantum chemistry simulations.
- Compare PennyLane and Qiskit implementations under identical physical conditions.
- Quantify performance improvements obtained from analytical adjoint gradients.
- Investigate the applicability of JAX Just-In-Time (JIT) compilation to self-consistent embedding calculations.
- Establish reproducible benchmark datasets for future quantum embedding research.

---

# Framework Evolution

The framework evolved through three major development stages, with each version addressing specific computational bottlenecks encountered during hybrid quantum embedding calculations.

---

# PennyLane Version 1 — Hybrid Prototype

## Overview

Version 1 represents the initial implementation of the hybrid embedding framework.

Quantum circuits were generated using **Qiskit Nature** and subsequently translated into **PennyLane** circuits for execution.

Although scientifically correct, repeated circuit translation introduced significant computational overhead during every embedding iteration.

---

## Core Architecture

| Component | Implementation |
|------------|---------------|
| Ansatz | Qiskit UCCSD |
| Mapper | ParityMapper |
| Backend | PennyLane |
| Gradient Evaluation | Numerical Finite Differences |

---

## Primary Bottleneck

Each SCF iteration required:

```python
bound_circuit.decompose()
qml.from_qiskit(...)
```

to convert Qiskit circuits into PennyLane-compatible circuits.

This repeated translation significantly limited GPU utilization and dominated the overall runtime.

---

# PennyLane Version 2 — Production Framework

Version 2 replaced the hybrid execution model with a fully native PennyLane implementation and became the production baseline for all subsequent developments.

---

## UCCSD Benchmark Suite

```text
PennyLane Version 2/UCCSD/
```

Implements the conventional **Unitary Coupled Cluster Singles and Doubles (UCCSD)** ansatz.

### Features

- Native PennyLane implementation
- Jordan-Wigner mapping
- Exact adjoint differentiation
- GPU execution (`lightning.gpu`)
- Active-space spin penalty Hamiltonian
- Cross-platform validation with Qiskit

---

## q-UCCSD Benchmark Suite

```text
PennyLane Version 2/qUCCSD/
```

Implements the **Quadratic UCCSD (q-UCCSD)** ansatz.

Compared with conventional UCCSD, q-UCCSD eliminates the long Jordan-Wigner parity strings, resulting in:

- Reduced circuit depth
- Lower CNOT count
- Improved compatibility between PennyLane and Qiskit
- Simplified wire ordering

---

## Major Improvements

### Native GPU Execution

Version 2 executes quantum circuits directly using

```text
lightning.gpu
```

without intermediate circuit translation.

---

### Exact Analytical Gradients

Numerical finite differences were replaced with

```python
diff_method="adjoint"
gradient_fn = qml.grad(cost_fn)
```

providing exact analytical derivatives and significantly reducing optimization cost.

---

### Active-Space Spin Hamiltonian

A dynamic spin penalty Hamiltonian

```math
H = H_0 + \beta \hat{S}^{2}
```

was introduced to suppress spin contamination and preserve the desired spin manifold throughout the embedding procedure.

---

### Cross-Framework Alignment

Additional patches aligned

- Qiskit block orbital ordering
- PennyLane Hartree-Fock state preparation
- Wire indexing conventions

allowing direct comparison between PennyLane and Qiskit implementations.

---

# PennyLane Version 3 — JAX-JIT Investigation

## Objective

Version 3 investigates the applicability of **JAX Just-In-Time (JIT)** compilation to self-consistent quantum embedding calculations.

---

## Findings

The study demonstrated that JIT compilation is fundamentally incompatible with self-consistent embedding workflows.

At every embedding iteration:

1. The classical DFT environment changes.
2. The molecular Hamiltonian changes.
3. The quantum circuit changes.
4. Previously compiled execution graphs become invalid.

Consequently, JAX repeatedly recompiles the computational graph, leading to:

- Extremely large memory consumption
- Longer execution times
- No practical performance improvement over Version 2

Version 2 therefore remains the recommended production implementation.

---

# Performance Benchmarks

## Version 1 vs Version 2 (Pyrene, CAS(6e,6o))

| Metric | Version 1 | Version 2 |
|----------|-----------:|----------:|
| Runtime | ~9.2 Hours | ~52 Minutes |
| Gradient Method | Finite Differences | Adjoint |
| GPU Utilization | 0–5% | 25–78% |
| Circuit Translation | Required | Eliminated |

---

## Version 2 vs Version 3 (Tetracene, CAS(6e,6o))

| Metric | Version 2 | Version 3 |
|----------|----------:|----------:|
| Runtime | ~6.1 Hours | ~7.1 Hours |
| Peak Memory | ~531 MB | ~62 GB |
| Primary Bottleneck | Python Loop | Graph Recompilation |

---

## Cross-Framework Comparison (Tetracene)

| Framework | Hardware | Runtime | Final Energy (Ha) |
|-----------|----------|---------:|------------------:|
| Qiskit CPU | Intel Xeon | ~5.23 Hours | -693.34300886 |
| PennyLane CPU | Intel Xeon | ~6.06 Hours | -693.18847956 |
| PennyLane GPU | NVIDIA A100 | ~6.14 Hours | -693.18844498 |

---

# Benchmark Results

Extensive benchmarking was performed on the **ParamPrabha Supercomputing Cluster**, equipped with:

- Intel Xeon Gold 6240R CPUs
- NVIDIA A100 Tensor Core GPUs

Representative benchmark results are summarized below.

| Framework | Ansatz | Runtime | Final Energy (Ha) | ⟨S²⟩ |
|-----------|--------|---------:|------------------:|------:|
| Qiskit CPU | UCCSD | 2.39 h | -693.271466 | 0.000 |
| Qiskit CPU | q-UCCSD | 2.73 h | -693.271491 | 0.000 |
| PennyLane GPU | UCCSD | 5.80 h | -693.190032 | 0.005 |
| PennyLane GPU | q-UCCSD | 4.95 h | -693.057367 | 0.531 |
| PennyLane CPU | q-UCCSD | 9.46 h | -692.550228 | 0.940 |

The repository also includes the corresponding:

- SLURM output logs (`.out`)
- Error logs (`.err`)
- Runtime summaries (`.csv`)

allowing complete reproduction of the reported benchmark studies.

---

# Known Limitations

## PennyLane Wire Ordering Limitation

### Error

```text
expected at least two wires representing the unoccupied orbitals; got 0
```

### Cause

Jordan-Wigner Z-string construction becomes invalid when directly applied to Qiskit's block-ordered orbitals, producing empty wire ranges.

---

## Qiskit AerEstimator Primitive Failure

### Error

```text
The primitive job failed!
```

### Cause

The Rust implementation underlying Qiskit Nature expects CPU-resident NumPy arrays.

Passing GPU-resident CuPy arrays across the Rust interface resulted in incompatible memory layouts and primitive execution failures.

---

# Current Project Status

## Completed

- Native PennyLane GPU implementation
- Native PennyLane CPU implementation
- Exact adjoint differentiation
- UCCSD benchmarking framework
- q-UCCSD benchmarking framework
- Cross-platform PennyLane/Qiskit validation
- Large-scale molecular benchmark studies
- JAX-JIT architectural investigation

---

# Future Work

Future developments may include:

- NVIDIA GH200 Grace Hopper deployment
- Unified CPU-GPU memory architectures
- Larger active-space calculations
- Multi-GPU quantum embedding
- Further optimization of q-UCCSD circuits
- Additional ansatz benchmarking studies
- Cross-validation with emerging quantum chemistry frameworks

---

# Notes

- **Version 2** represents the validated production implementation of the PennyLane-based quantum embedding framework.
- **Version 1** is retained for historical comparison and architectural benchmarking.
- **Version 3** documents the limitations of applying JAX-JIT compilation to self-consistent quantum embedding workflows.
- The benchmark outputs included throughout the repository provide reproducible performance data for all reported framework versions.
