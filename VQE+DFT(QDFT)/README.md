# QDFT (VQE + DFT) Simulation Framework

A hybrid **Quantum-Classical Density Functional Theory (QDFT)** framework that combines **Variational Quantum Eigensolver (VQE)** workflows with self-consistent **Density Functional Theory (DFT)** embedding loops.

This repository contains production implementations, and convergence strategies for performing quantum-enhanced electronic structure calculations within a DFT embedding framework.

---

# Framework Overview

The QDFT framework integrates:

* **Classical DFT calculations** for environment and embedding potentials.
* **Quantum VQE solvers** for active-space electronic structure problems.
* **Self-Consistent Field (SCF) iterations** to achieve convergence between quantum and classical subsystems.
* Multiple convergence schemes, and initialization strategies.

---

# Repository Structure

```text
├── Energy+Density/
├── Energy/
│   ├── HomoLumo/
│   ├── MP2init/
│   └── ROKS_RKS/
```

---

# 1. Energy + Density Convergence (`/Energy+Density`)

Production-grade implementations using the **UCCSD Ansatz** and a **dual-convergence criterion**:

1. Total Energy convergence
2. Electron Density Matrix convergence (Frobenius norm)

Both conditions must be satisfied before self-consistency is declared.

## `LDA-RS_MP2_spin.py`

Implements the **Local Density Approximation with Range Separation (LDA-RS)**.

Features:

* Explicit range-separation parameter `ω`
* Active-space spin Hamiltonian penalty correction
* MP2-based state initialization
* Energy + density convergence monitoring

## `OtherFunctional_MP2_spin.py`

General workflow for standard exchange-correlation functionals, including:

* B3LYP
* CAM-B3LYP
* LRC-ωPBE
* PBE
* LDA

Features:

* Active-space spin penalty correction
* MP2 initialization
* Energy + density convergence

No explicit `ω` parameter is required.

## `tuned_MP2_spin.py`

Implementation for a tuned **CAM-B3LYP** functional.

Features:

* Active-space spin penalty correction
* MP2 initialization
* Energy + density convergence

## `tuned_MP2.py`

Control/baseline implementation using:

* Tuned CAM-B3LYP
* MP2 initialization

Unlike `tuned_MP2_spin.py`, this version does **not** apply active-space spin penalties.

---

# 2. Energy-Only Convergence (`/Energy`)

Scripts in this directory perform self-consistent embedding calculations while monitoring **total energy only** as the SCF convergence criterion. Collection of specialized workflows built around the UCCSD Ansatz.

---

## A. HOMO-LUMO Gap Analysis (`/HomoLumo`)

Tools for extracting:

* Ground-state energies
* HOMO-LUMO energy gaps

### `HomoLumo_v1.py`

Uses a standard Hermitian eigensolver:

```python
numpy.linalg.eigvalsh()
```

Assumes:

* Orthogonal basis functions
* Negligible atomic orbital overlap

### `HomoLumo_v2.py`

Extends the workflow to non-orthogonal atomic orbital basis sets.

Retrieves the overlap matrix:

```python
mf.get_ovlp()
```

and solves the generalized eigenvalue problem:

```python
scipy.linalg.eigh(
    fock_total,
    b=overlap_matrix,
    eigvals_only=True
)
```

---

## B. MP2 Initialization Baseline (`/MP2init`)

### `DFT+VQE_MP2_Init.py`

Reference implementation used to evaluate the impact of MP2-based initialization on convergence behavior when only energy convergence is monitored.

---

## C. Restricted vs Restricted Open-Shell Drivers (`/ROKS_RKS`)

Support for:

* Restricted Kohn-Sham (RKS) calculations
* Restricted Open-Shell Kohn-Sham (ROKS) calculations

Designed primarily for open-shell systems such as molecular oxygen (`O₂`).

---

# Functional Constraints

## LDA-RS Workflows

All LDA-RS implementations require explicit specification of the range-separation parameter:

```python
ω
```

## Standard Functional Workflows

For conventional exchange-correlation functionals such as:

* B3LYP
* PBE
* LDA
* CAM-B3LYP

the range-separation parameter `ω` should **not** be provided unless explicitly required by the functional definition.

---

# Convergence and Damping Reference

## VQE Callbacks

Several callback implementations exist throughout the repository due to historical development and testing workflows.

## Production Reference

For the most stable and up-to-date implementations of:

* VQE callbacks
* Density matrix damping
* Energy damping
* Dual-convergence logic

always refer to the production scripts located in:

```text
Energy+Density/
```

These files contain the current validated convergence infrastructure used throughout the QDFT framework.

---

# Recommended Production Workflow

For all new developments and benchmark studies:

1. Use implementations in `Energy+Density/`.
2. Prefer dual energy-density convergence over energy-only convergence.
3. Use MP2 initialization when available.
4. Reference production callback and damping implementations from the `Energy+Density` directory.

```
```
