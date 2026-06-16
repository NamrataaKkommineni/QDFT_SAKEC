# DFT + FCI Embedding Framework

This directory contains the core implementations of a hybrid **Density Functional Theory + Full Configuration Interaction (DFT+FCI) Embedding Framework**.

Unlike the VQE-based workflows within the broader QDFT ecosystem, these implementations replace variational quantum optimization with a classical **exact diagonalization** strategy. The active-space electronic structure problem is solved using the **NumPyMinimumEigensolver**, enabling exact Full Configuration Interaction (FCI) energies to be obtained within the self-consistent embedding loop.

As a result, these workflows serve as a high-accuracy reference for evaluating and benchmarking hybrid DFT+VQE embedding methodologies.

---

# Framework Overview

The DFT+FCI framework combines:

* Classical Density Functional Theory (DFT)
* Active-space Full Configuration Interaction (FCI)
* Self-consistent quantum-classical embedding cycles
* Exact active-space diagonalization via NumPyMinimumEigensolver
* Energy and density convergence monitoring

By eliminating variational optimization error, these implementations provide an exact active-space baseline against which VQE-based workflows can be compared.

---

# Repository Structure

```text
├── Energy+Density/
└── Energy/
```

---

# 1. Energy + Density Convergence (`/Energy+Density`)

Scripts in this directory employ a dual-convergence criterion based on:

1. Total Energy Convergence
2. Electron Density Matrix Convergence

These implementations primarily focus on convergence stabilization techniques and spin-state preservation methodologies.

---

## `FCI_ChupinDiis_NoSpinProtection.py`

Implementation of an adaptive **Direct Inversion in the Iterative Subspace (DIIS)** damping strategy inspired by the Chupin framework.

Unlike traditional DIIS approaches that activate predefined subspaces after fixed iteration counts, this implementation dynamically determines:

* DIIS activation criteria
* Subspace construction
* Error-vector management

during runtime.

### Features

* Adaptive DIIS damping
* Dynamic subspace allocation
* Energy and density convergence monitoring
* No spin filtering
* No spin Hamiltonian penalty corrections

### Purpose

Designed for investigating advanced SCF stabilization techniques independent of spin-symmetry enforcement.

---

## `FCI_spinFilter.py`

Implementation of a spin-symmetry preservation strategy based on post-solution state filtering.

Rather than modifying the Hamiltonian directly, candidate states are evaluated after diagonalization and filtered according to their spin properties.

### Features

* Post-diagonalization spin filtering
* Energy and density convergence monitoring
* Exact FCI active-space solver

### Notes

Although mathematically rigorous, post-processing spin filters are generally more computationally expensive than Hamiltonian-based spin penalties because unwanted states are still generated and evaluated before filtering.

This implementation does **not** utilize an active-space spin Hamiltonian penalty.

---

## `FCI_spinHamiltonian.py`

Production implementation incorporating the optimized **Spin Hamiltonian Penalty Patch**.

Spin symmetry is enforced directly within the active-space Hamiltonian, enabling the solver to preferentially converge toward the desired spin manifold during diagonalization.

### Supported Functionals

* B3LYP
* PBE
* LRC-ωPBE
* Standard LDA
* Other conventional exchange-correlation functionals

### Features

* Active-space spin penalty correction
* Energy and density convergence monitoring
* Exact FCI solver
* Standard exchange-correlation functional support

### Functional Constraint

No range-separation parameter `ω` is required.

---

## `FCI_spinHamiltonian_LDA-RS.py`

Extension of the spin Hamiltonian workflow to the **Range-Separated Local Density Approximation (LDA-RS)** framework.

### Features

* Active-space spin Hamiltonian penalty
* Energy and density convergence monitoring
* Exact FCI solver
* LDA-RS functional support

### Functional Constraint

Requires explicit specification of the range-separation parameter:

```python
ω
```

---

## `FCI_tuned_spinHamiltonian.py`

Implementation of the spin Hamiltonian penalty framework combined with a customized, optimally tuned **CAM-B3LYP** exchange-correlation functional.

### Features

* Tuned CAM-B3LYP workflow
* Active-space spin Hamiltonian penalty
* Exact FCI active-space solver
* Energy and density convergence monitoring

### Purpose

Provides a high-accuracy reference implementation for tuned range-separated hybrid functionals while preserving spin-state fidelity throughout the embedding cycle.

---

# 2. Energy-Only Convergence (`/Energy`)

Scripts in this directory employ a simplified convergence criterion based exclusively on total energy.

Unlike the workflows in `Energy+Density`, density matrix convergence is not explicitly monitored.

---

# Spin Symmetry Note

None of the implementations in this directory include:

* Spin filtering procedures
* Spin Hamiltonian penalty corrections

These workflows are intended for energy-focused convergence studies and baseline comparisons.

---

## `DFT+FCI_LDA-RS.py`

Energy-only convergence workflow configured for the **LDA-RS** functional.

### Features

* Exact FCI active-space solver
* Energy-only convergence
* LDA-RS exchange-correlation treatment

### Functional Constraint

Requires explicit specification of:

```python
ω
```

---

## `DFT+FCI_LDA.py`

General-purpose energy-only workflow for conventional exchange-correlation functionals.

### Supported Functionals

* LDA
* B3LYP
* CAM-B3LYP
* PBE
* Other standard functionals

### Features

* Exact FCI active-space solver
* Energy-only convergence monitoring
* Standard exchange-correlation functional support

### Functional Constraint

No range-separation parameter `ω` is used.

---

## `FCI_camb3lyp_tuned.py`

Energy-only convergence workflow dedicated to a tuned CAM-B3LYP exchange-correlation functional.

### Features

* Tuned CAM-B3LYP implementation
* Exact FCI active-space solver
* Energy-only convergence monitoring

### Purpose

Provides a direct FCI reference for tuned CAM-B3LYP embedding calculations without density-based convergence constraints.

---

# Functional Parameter Rules

## LDA-RS Configurations

All implementations utilizing the LDA-RS exchange-correlation framework must explicitly declare and process the range-separation parameter:

```python
ω
```

This includes both energy-only and energy-density convergence workflows.

---

## Standard Functional Configurations

Implementations based on conventional exchange-correlation functionals, including:

* LDA
* B3LYP
* PBE
* CAM-B3LYP
* LRC-ωPBE

do not require an explicit `ω` input unless it is inherently part of the functional definition.

---

# Recommended Usage

| Workflow Type                               | Recommended Scripts                                                                       |
| ------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Spin-symmetry-preserving production runs    | `FCI_spinHamiltonian.py`, `FCI_spinHamiltonian_LDA-RS.py`, `FCI_tuned_spinHamiltonian.py` |
| Spin-filter methodology studies             | `FCI_spinFilter.py`                                                                       |
| Chupin DIIS stabilization                   | `FCI_ChupinDiis_NoSpinProtection.py`                                                      |
| Exact FCI baseline comparisons              | All workflows                                                                             |
| Energy-only benchmarking                    | `DFT+FCI_LDA.py`, `DFT+FCI_LDA-RS.py`, `FCI_camb3lyp_tuned.py`                            |

---

# Production Notes

For production calculations requiring robust spin-state preservation, the preferred workflows are:

```text
FCI_spinHamiltonian.py
FCI_spinHamiltonian_LDA-RS.py
FCI_tuned_spinHamiltonian.py
```

These implementations enforce spin symmetry directly within the active-space Hamiltonian and generally offer superior computational efficiency compared to post-processing spin filtering approaches.

The DFT+FCI framework serves as the exact active-space benchmark layer of the broader QDFT ecosystem and should be used as the primary reference when assessing the accuracy of VQE-based embedding calculations.
