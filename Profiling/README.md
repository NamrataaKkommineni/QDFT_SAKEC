# Runtime Performance Profiling Suite

This directory contains dedicated benchmarking and profiling utilities for the hybrid Quantum-Classical Density Functional Theory (QDFT) embedding framework.

The profiling suite is designed to characterize the computational cost of individual framework components and identify performance bottlenecks across self-consistent embedding calculations.

All implementations in this directory employ the standard **UCCSD Ansatz** together with random parameter initialization and **Linear Adaptive Damping** during convergence.

---

# Profiling Objectives

The primary goals of this suite are:

* Quantify total simulation runtime
* Measure memory consumption
* Track CPU and GPU workload distribution
* Isolate bottlenecks within the embedding workflow
* Evaluate scaling behavior across molecular systems

---

# Monitored Components

The profiling framework independently measures the computational cost of:

## Classical DFT Calculations

Tracks:

* SCF execution time
* Exchange-correlation evaluations
* Fock matrix construction
* Density matrix generation

---

## Embedding Operations

Tracks:

* Embedding matrix construction
* Active-space projections
* Density transfer operations
* Quantum-classical interface costs

---

## Quantum VQE Execution

Tracks:

* Circuit evaluations
* Optimizer overhead
* Gradient calculations
* Hamiltonian expectation value measurements

---

# Convergence Stabilization

All profiling implementations utilize **Linear Adaptive Damping** to reduce oscillatory behavior during embedding iterations.

The damping mechanism smooths updates to:

* Total energy
* Density matrices
* Embedding potentials

thereby improving convergence stability during performance measurements.

---

# Directory Contents

## `ProfiledCode_old_Basic.py`

### Version 1 Baseline

Original profiling implementation used during early development of the QDFT framework.

### Features

* Runtime tracking
* Memory monitoring
* Component-level timing diagnostics
* UCCSD Ansatz

### Known Limitation

A synchronization issue exists between:

* Standard output logs
* Generated output files
* Cluster `.err` files

This bug can produce inconsistent timing reports depending on the execution environment.

### Status

Maintained primarily for historical benchmarking comparisons.

---

## `ProfiledCode_new_Advanced.py`

### Version 2 Advanced

Enhanced profiling framework designed to address limitations identified in the original implementation.

### Improvements

* Corrected log synchronization behavior
* Improved runtime accounting
* Fine-grained timing instrumentation
* Enhanced CPU/GPU workload tracking

### Features

* Accurate wall-clock profiling
* Memory consumption tracking
* Quantum-classical workload decomposition
* UCCSD-based benchmarking
* Linear Adaptive Damping

### Status

Recommended profiling implementation for all future benchmarking studies.

---

# Recommended Usage

| Objective                         | Recommended Script             |
| --------------------------------- | ------------------------------ |
| Historical comparison studies     | `ProfiledCode_old_Basic.py`    |
| Accurate performance benchmarking | `ProfiledCode_new_Advanced.py` |
| CPU/GPU workload analysis         | `ProfiledCode_new_Advanced.py` |
| Runtime scaling studies           | `ProfiledCode_new_Advanced.py` |

---

# Production Notes

The profiling suite is intended exclusively for performance characterization and benchmarking.

These scripts should not be treated as production simulation workflows.

For production calculations and validated convergence procedures, refer to the primary QDFT implementations located in:

```text
Energy+Density/
```

which contain the most up-to-date convergence logic, callback definitions, damping mechanics, and spin-state management routines.

The profiling scripts are best viewed as diagnostic tools for understanding and optimizing the computational performance of the broader QDFT framework.
