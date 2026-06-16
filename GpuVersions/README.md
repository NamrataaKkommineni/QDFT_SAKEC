# Quantum Embedding Benchmarking Framework

This repository documents the development, optimization, and benchmarking of a hybrid **Quantum-Classical Embedding Framework** implemented using **PennyLane**, **Qiskit Nature**, and GPU-accelerated quantum simulation backends.

The project investigates the practical performance limits of Variational Quantum Eigensolver (VQE) workflows for quantum embedding calculations, focusing on:

* Ansatz construction strategies
* Fermion-to-qubit mappings
* Gradient evaluation methods
* GPU acceleration
* JIT compilation performance
* Cross-platform validation between PennyLane and Qiskit

The repository captures the complete architectural evolution from an initial proof-of-concept implementation to a production-grade GPU workflow and an exploration of JAX-based acceleration limits.

---

# Project Objectives

The primary goals of this framework are:

* Develop scalable VQE-based embedding workflows.
* Evaluate GPU acceleration for quantum chemistry simulations.
* Compare PennyLane and Qiskit implementations under identical physical conditions.
* Quantify performance gains from analytical gradient methods.
* Investigate the applicability of JIT compilation to self-consistent embedding calculations.
* Establish benchmark datasets for future quantum embedding research.

---

# Architectural Evolution

The framework evolved through three major development stages.

---

# Version 1: Naive Hybrid Framework

## Overview

The initial implementation combined Qiskit Nature circuit generation with PennyLane execution.

While functional, the architecture suffered from substantial translation and optimization overhead.

---

## Core Architecture

### Ansatz

Constructed using Qiskit Nature:

```python
QiskitUCCSD
```

### Fermion-to-Qubit Mapping

```python
ParityMapper
```

The parity mapping was selected to reduce qubit counts and memory consumption.

---

## Execution Bottleneck

At every iteration of the embedding loop, the workflow required:

```python
bound_circuit.decompose()
qml.from_qiskit(...)
```

to convert a Qiskit circuit object into a PennyLane-compatible representation.

This conversion occurred repeatedly inside the self-consistent optimization cycle and became a major computational bottleneck.

---

## Gradient Evaluation

Optimization relied on numerical finite differences through SciPy.

Characteristics:

* Numerical gradient approximation
* More than 150 circuit evaluations per optimization step
* Significant computational overhead
* Poor GPU utilization

---

## Outcome

While scientifically correct, the implementation was not computationally viable for large-scale molecular systems.

---

# Version 2: Native PennyLane Execution

## Overview

Version 2 replaced the hybrid circuit construction pipeline with a fully native PennyLane implementation.

This architecture became the production baseline for all subsequent developments.

---

## Core Architecture

### Ansatz

Implemented directly through PennyLane:

```python
qml.UCCSD
```

### Fermion-to-Qubit Mapping

```python
JordanWignerMapper
```

This mapping was adopted to satisfy PennyLane's template requirements:

```text
1 spatial orbital = 2 qubits
```

---

## Execution Backend

All circuits execute directly on:

```python
lightning.gpu
```

Advantages:

* No circuit translation overhead
* No repeated decomposition steps
* Direct GPU execution
* Reduced memory consumption

---

## Analytical Gradients

Version 2 replaced numerical differentiation with exact analytical gradients.

Configuration:

```python
diff_method="adjoint"
```

Gradient evaluation:

```python
gradient_fn = qml.grad(cost_fn)
```

The analytical gradient function is passed directly to SciPy optimizers.

---

## Advantages

Compared with Version 1:

* Elimination of finite-difference overhead
* Exact gradient evaluation
* Improved optimization stability
* Dramatically reduced circuit executions
* Substantially improved GPU utilization

---

## Production Status

Version 2 was identified as the optimal architecture and became the foundation for all subsequent:

* Physics corrections
* Spin penalty implementations
* Scaling studies
* Benchmarking workflows

---

# Version 3: JAX-JIT Acceleration Study

## Objective

The third development phase investigated whether Just-In-Time (JIT) compilation could further accelerate embedding calculations.

The framework integrated:

```python
jax
jax.numpy
```

with computational kernels wrapped in:

```python
@jax.jit
```

decorators.

---

## Design Strategy

JAX tensors were interfaced with SciPy optimization routines through conversion bridges:

```python
scipy_cost(...)
scipy_grad(...)
```

The objective was to maximize GPU graph execution efficiency.

---

## Observed Behavior

For large molecular systems such as tetracene:

| Metric            | Result    |
| ----------------- | --------- |
| Peak Memory Usage | ~62 GB    |
| Runtime           | > 7 Hours |

Performance degraded dramatically compared with the production Version 2 workflow.

---

## Root Cause Analysis

The failure was traced to a fundamental incompatibility between JIT compilation and self-consistent embedding loops.

At each SCF iteration:

1. The embedding environment updates.
2. The molecular Hamiltonian changes.
3. The quantum circuit representation changes.
4. The JIT graph becomes invalid.

As a consequence, JAX was forced to:

* Discard the compiled execution graph
* Reconstruct the graph
* Recompile the entire quantum circuit

for every embedding iteration.

For circuits exceeding:

```text
3,500+ quantum gates
```

the recompilation overhead outweighed any potential acceleration benefits.

---

## Conclusion

JIT compilation was determined to be unsuitable for large-scale self-consistent quantum embedding workflows where the Hamiltonian changes at every iteration.

---

# Production Framework (Version 2)

Following the architectural evaluation, Version 2 was adopted as the production framework.

The repository is organized into two primary benchmarking suites.

---

# UCCSD Benchmark Suite

```text
PennyLane Version 2/UCCSD/
```

Production benchmarking framework using the standard **Unitary Coupled Cluster Singles and Doubles (UCCSD)** ansatz.

---

## Features

### Native GPU Execution

```python
lightning.gpu
```

### Active-Space Spin Penalty Patch

Custom Hamiltonian modification used to enforce spin-state constraints throughout embedding convergence.

### Cross-Platform Validation

Equivalent implementations are provided for:

* PennyLane GPU
* PennyLane CPU
* Qiskit CPU

---

## Benchmarking Goals

The suite is designed to verify:

### Numerical Consistency

Identical physical systems should produce equivalent:

* Ground-state energies
* Convergence trajectories
* Electronic properties

across PennyLane and Qiskit implementations.

### Performance Scaling

Measure acceleration achieved through:

* Analytical adjoint gradients
* GPU execution
* Native PennyLane workflows

---

# qUCCSD Benchmark Suite

```text
PennyLane Version 2/qUCCSD/
```

Alternative benchmarking framework based on the **quadratic UCCSD (qUCCSD)** ansatz.

---

## Purpose

Evaluate the influence of ansatz design on:

* Convergence behavior
* Optimization stability
* Computational scaling
* Embedding performance

---

## Features

* Native PennyLane GPU implementation
* PennyLane CPU implementation
* Qiskit CPU implementation
* Active-space spin penalty patch
* Cross-platform benchmarking support

The spin penalty implementation is identical to that used in the UCCSD framework, ensuring consistent physical constraints across all comparative studies.

---

# Current Project Status

## Development Status

⚠️ **Work in Progress**

The framework remains under active development.

---

## Completed

* Native PennyLane UCCSD implementation
* GPU-enabled execution workflows
* Analytical adjoint gradient integration
* Active-space spin penalty framework
* UCCSD and qUCCSD benchmarking infrastructure
* JAX-JIT architectural evaluation

---

## Ongoing Work

* Large-scale GPU acceleration studies
* Cross-platform validation between PennyLane and Qiskit
* Numerical consistency verification
* qUCCSD convergence benchmarking
* Extended molecular scaling analyses

---

## Current Assessment

Although Version 2 represents the most efficient architecture identified to date, the ultimate objective of achieving substantial GPU acceleration for large-scale quantum embedding calculations remains unresolved.

Furthermore, cross-framework verification is still ongoing, and definitive confirmation that PennyLane and Qiskit produce numerically identical results across all benchmark systems has not yet been established.

Future development efforts will focus on addressing these outstanding validation and performance objectives.
