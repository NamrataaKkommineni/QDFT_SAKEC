# Quantum Embedding Benchmarking Framework

This repository documents the development, optimization, and benchmarking of a **Hybrid Quantum-Classical Embedding Framework** implemented using **PennyLane**, **Qiskit Nature**, and GPU-accelerated quantum simulation backends.

The project investigates the practical performance limits of **Variational Quantum Eigensolver (VQE)** workflows for quantum embedding calculations, with particular emphasis on:

* Ansatz construction strategies
* Fermion-to-qubit mapping techniques
* Analytical versus numerical gradient evaluation
* GPU acceleration
* JAX Just-In-Time (JIT) compilation
* Cross-platform validation between PennyLane and Qiskit

The repository captures the complete architectural evolution from an initial proof-of-concept implementation to a production-grade GPU workflow, together with an extensive investigation into the limitations of JIT compilation for self-consistent quantum embedding calculations.

---

# Project Objectives

The primary objectives of this framework are to:

* Develop scalable VQE-based embedding workflows.
* Evaluate GPU acceleration for quantum chemistry simulations.
* Compare PennyLane and Qiskit implementations under identical physical conditions.
* Quantify performance improvements obtained from analytical adjoint gradients.
* Investigate the applicability of JAX JIT compilation to self-consistent embedding calculations.
* Establish reproducible benchmark datasets for future quantum embedding research.

---

# Architectural Evolution

The framework evolved through three major development stages, with each version addressing specific computational bottlenecks.

---

# Version 1 — Naive Hybrid Framework

## Overview

The initial implementation combined **Qiskit Nature** for ansatz generation with **PennyLane** for circuit execution.

Although scientifically correct, this architecture introduced substantial overhead arising from repeated circuit translation during every embedding iteration.

---

## Core Architecture

### Ansatz

* Qiskit Nature `QiskitUCCSD`

### Fermion-to-Qubit Mapping

* `ParityMapper`

Selected to reduce qubit count and memory overhead.

### Gradient Evaluation

* Numerical finite differences (SciPy)
* More than **150 circuit evaluations** per optimization step

---

## Execution Bottleneck

At every iteration of the embedding loop, the workflow executed

```python
bound_circuit.decompose()
qml.from_qiskit(...)
```

to translate Qiskit circuits into native PennyLane circuits.

Because this conversion occurred inside every SCF iteration, significant communication overhead was introduced between the classical optimization loop and the quantum backend, severely limiting GPU utilization.

---

# Version 2 — Native PennyLane Execution (Production Baseline)

## Overview

Version 2 replaced the hybrid execution pipeline with a fully native PennyLane implementation.

This architecture became the production baseline for all subsequent developments.

---

## Core Architecture

### Ansatz

```python
qml.UCCSD
```

### Fermion-to-Qubit Mapping

```text
JordanWignerMapper
```

using the standard mapping

```text
1 Spatial Orbital = 2 Qubits
```

### Execution Backend

```text
lightning.gpu
```

allowing direct GPU execution without intermediate circuit translation.

---

## Analytical Gradient Evaluation

Version 2 replaced numerical finite differences with exact analytical gradients using

```python
diff_method="adjoint"
gradient_fn = qml.grad(cost_fn)
```

This reduced optimization cost dramatically while improving convergence stability.

---

## Physics Improvements

### Active-Space Spin Penalty

A dynamic active-space spin penalty Hamiltonian

```math
H = H_{0} + \beta \hat{S}^{2}
```

was introduced to suppress spin contamination and enforce convergence toward the target singlet state.

---

### Spin Ordering Alignment

Additional patches aligned:

* Qiskit block orbital ordering
* PennyLane Hartree-Fock state preparation
* Wire indexing conventions

allowing consistent cross-framework comparisons.

---

# Version 3 — JAX-JIT Compilation Study

## Objective

The third development phase investigated whether **JAX Just-In-Time (JIT)** compilation could further accelerate embedding calculations by compiling large sections of the computational graph.

---

## Root Cause Analysis

The investigation demonstrated a fundamental incompatibility between JIT compilation and self-consistent embedding algorithms.

During every embedding iteration:

1. The classical DFT environment changes.
2. The molecular Hamiltonian changes.
3. The quantum circuit changes.
4. The compiled execution graph becomes invalid.

Consequently, JAX repeatedly discarded and reconstructed the compiled execution graph.

For large molecular systems containing more than **3,500 quantum gates**, this resulted in:

* Repeated graph recompilation
* Memory usage approaching **62 GB**
* Longer execution times than the native GPU implementation

---

## Conclusion

JIT compilation was determined to be unsuitable for self-consistent hybrid quantum embedding calculations where the Hamiltonian changes every iteration.

Version 2 therefore remains the production architecture.

---

# Hardware Performance Benchmarks

## Version 1 vs Version 2 (Pyrene, CAS(6e,6o))

| Metric              |          Version 1 |   Version 2 |
| ------------------- | -----------------: | ----------: |
| Execution Time      |         ~9.2 Hours | ~52 Minutes |
| Gradient Method     | Finite Differences |     Adjoint |
| GPU Utilization     |               0–5% |      25–78% |
| Circuit Translation |           Required |  Eliminated |

---

## Native GPU vs JAX-JIT (Tetracene, CAS(6e,6o))

| Metric             |  Native GPU |           JAX-JIT |
| ------------------ | ----------: | ----------------: |
| Runtime            |  ~6.1 Hours |        ~7.1 Hours |
| Gradient Method    |     Adjoint |           Adjoint |
| Peak Memory        |     ~531 MB |            ~62 GB |
| Primary Bottleneck | Python Loop | Graph Compilation |

---

## Framework Comparison (Tetracene)

| Framework     | Hardware    |     Runtime | Final Energy (Ha) |
| ------------- | ----------- | ----------: | ----------------: |
| Qiskit CPU    | Intel Xeon  | ~5.23 Hours |     -693.34300886 |
| PennyLane CPU | Intel Xeon  | ~6.06 Hours |     -693.18847956 |
| PennyLane GPU | NVIDIA A100 | ~6.14 Hours |     -693.18844498 |

---

# Recent Cluster Benchmark Results

Recent benchmarking was performed on the **ParamPrabha Supercomputing Cluster** equipped with:

* Intel Xeon Gold 6240R CPUs
* NVIDIA A100 Tensor Core GPUs

| Framework     | Ansatz    | Runtime | Final Energy (Ha) | ⟨S²⟩  |
| ------------- | --------- | ------  | -----------------:|  ---: |
| Qiskit CPU    | UCCSD     | 2.39 h  | -693.271466       | 0.000 |
| Qiskit CPU    | q-UCCSD   | 2.73 h  | -693.271491       | 0.000 |
| PennyLane GPU | q-UCCSD   | 4.95 h  | -693.057367       | 0.531 |
| PennyLane GPU | UCCSD     | 5.80 h  | -693.190032       | 0.005 |
| PennyLane CPU | q-UCCSD   | 9.46 h  | -692.550228       | 0.940 |

---

# Documented Issues

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

# Benchmark Suites

The repository is organized into two complementary benchmarking suites.

---

## UCCSD Benchmark Suite

```text
PennyLane Version 2/UCCSD/
```

Provides the baseline implementation using the conventional **Unitary Coupled Cluster Singles and Doubles (UCCSD)** ansatz.

Primary objectives include:

* Circuit depth benchmarking
* Energy convergence
* GPU performance characterization
* Cross-framework validation

---

## q-UCCSD Benchmark Suite

```text
PennyLane Version 2/qUCCSD/
```

Implements the **Quadratic UCCSD (q-UCCSD)** ansatz.

Unlike conventional UCCSD, q-UCCSD removes the long Jordan-Wigner Z-string ladders, enabling:

* Reduced circuit depth
* Lower CNOT count
* Consistent parameter layouts between PennyLane and Qiskit
* Elimination of wire-ordering inconsistencies

---

# Current Project Status

## Completed

* Native PennyLane GPU implementation
* Native PennyLane CPU implementation
* Exact adjoint gradient integration
* Cross-platform benchmarking infrastructure
* JAX-JIT architectural analysis
* q-UCCSD implementation
* Large-scale molecular benchmarking

---

# Future Outlook

The current implementation is constrained primarily by CPU–GPU communication overhead during self-consistent embedding iterations.

Future development can investigate deployment on the **NVIDIA GH200 Grace Hopper Superchip**, whose unified NVLink-C2C memory architecture removes the PCIe communication bottleneck between CPU and GPU.

This architecture is expected to substantially improve hybrid quantum-classical embedding performance by allowing both processors to share a unified memory space, reducing data-transfer latency during iterative embedding calculations.
