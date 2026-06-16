# QDFT Anion Embedding Module

## Doublet Open-Shell Framework

This directory contains production-grade implementations of a hybrid **Quantum-Classical Density Functional Theory (QDFT) Embedding Framework** designed for **anion molecular systems** with **open-shell doublet electronic configurations** (`S = 1/2`).

The framework combines:

* **Qiskit Nature** quantum workflows
* Native **UCCSD Ansatz**
* **L-BFGS-B** variational optimization
* **PySCF** classical electronic structure calculations
* Self-consistent quantum-classical embedding loops

Unlike conventional embedding implementations that monitor total energy alone, self-consistency is determined through a **dual-convergence criterion**:

1. **Total Energy Convergence**
2. **Electron Density Matrix Convergence**

Both criteria must be simultaneously satisfied before convergence is declared.

---

# Framework Compatibility

Although this module is configured for **open-shell anion calculations**, the underlying framework is fully compatible with **cationic molecular systems**.

Converting an anion workflow to a cation workflow requires modifying the physical system parameters within the `main()` execution block.

## Required Parameter Adjustments

### Electron Count

Reduce the total number of electrons by updating:

```python
active_num_electrons
```

### Spin Configuration

Update the alpha and beta electron populations:

```python
num_alpha
num_beta
```

and ensure the particle specification reflects the desired charge and spin state:

```python
num_particles
```

### Molecular Charge

Within the PySCF driver initialization, modify:

```python
charge = 1
```

(or the appropriate positive integer charge state).

### System Spin

Update the spin quantum number to remain consistent with the modified electron configuration and target multiplicity.

---

# Directory Contents

This module contains two primary production workflows that differ in their treatment of exchange-correlation functionals and spin-state management.

---

## `anion_otherFunctional.py`

General-purpose implementation for standard exchange-correlation functionals supported through PySCF.

Supported examples include:

* B3LYP
* LRC-ωPBE
* Other LibXC-compatible functionals

### Features

* Restricted Open-Shell Kohn-Sham (ROKS) formalism
* Open-shell doublet support
* Standard PySCF exchange-correlation handling
* Dual energy-density convergence monitoring
* Native UCCSD VQE workflow
* L-BFGS-B optimization

### Purpose

Recommended for benchmarking and production calculations using established exchange-correlation functionals without custom parameter tuning.

---

## `anion_tuned_spin.py`

Advanced implementation featuring a fully customizable **Tuned CAM-B3LYP** exchange-correlation framework.

Rather than relying on predefined LibXC functional definitions, this workflow directly constructs the exchange-correlation functional by injecting user-defined parameters into the PySCF kernel.

### Custom Functional Features

* User-defined short-range Hartree-Fock exchange
* User-defined long-range Hartree-Fock exchange
* Custom Becke88 exchange weights
* Dynamic CAM-B3LYP parameter tuning
* Direct PySCF kernel injection
* Bypasses standard LibXC parameterization constraints

### Spin Penalty Hamiltonian Patch

This implementation additionally incorporates an **Active-Space Spin Penalty Hamiltonian** designed to suppress spin contamination.

Open-shell doublet anions are particularly susceptible to artificial mixing with higher-spin states during variational optimization. To mitigate this effect, the script evaluates the total spin operator:

```math
\hat{S}^2
```

and augments the active-space Hamiltonian with a spin-dependent penalty term.

The penalty is constructed such that the target doublet state remains energetically unaffected.

For a doublet state:

```math
S(S+1) = \frac{1}{2}\left(\frac{1}{2}+1\right)=0.75
```

An identity offset corresponding to this value is included in the penalty operator, ensuring:

* The physical doublet ground state receives zero penalty.
* Higher-spin contaminants incur an energetic cost.
* Variational optimization is biased toward the correct spin manifold.

### Features

* Tuned CAM-B3LYP implementation
* Custom exchange-correlation parameterization
* Active-space spin penalty correction
* Open-shell doublet stabilization
* UCCSD-based VQE workflow
* L-BFGS-B optimization
* Dual energy-density convergence monitoring

---

# Recommended Usage

| Workflow                   | Recommended Use Case                                                                                                               |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `anion_otherFunctional.py` | Standard exchange-correlation functional studies and benchmarking                                                                  |
| `anion_tuned_spin.py`      | Tuned CAM-B3LYP calculations requiring enhanced control over exchange-correlation parameters and suppression of spin contamination |

---

# Production Notes

For open-shell anion systems, the preferred production workflow is:

```text
anion_tuned_spin.py
```

due to its explicit spin-contamination mitigation strategy and flexible tuned CAM-B3LYP implementation.

For validation studies, benchmarking, and comparisons against conventional exchange-correlation functionals, use:

```text
anion_otherFunctional.py
```

which provides a cleaner reference implementation using standard PySCF-supported functionals.
