# QDFT (VQE + DFT) Simulation Framework

A hybrid **Quantum-Classical Density Functional Theory (QDFT)** framework that integrates **Variational Quantum Eigensolver (VQE)** workflows with self-consistent **Density Functional Theory (DFT)** embedding calculations.

The repository contains production implementations, benchmarking utilities, convergence studies, and supporting tools for performing quantum-enhanced electronic structure calculations on strongly correlated molecular systems.

---

# Framework Overview

The QDFT framework combines

- Classical Density Functional Theory (DFT) for the environment and embedding potential
- Variational Quantum Eigensolver (VQE) for active-space electronic structure
- Self-Consistent Field (SCF) embedding cycles
- Spin-state preservation through active-space spin Hamiltonians
- Multiple initialization, damping, and convergence strategies

The framework has been developed to investigate practical quantum embedding workflows while maintaining compatibility with modern quantum computing software stacks such as **Qiskit Nature**, **PySCF**, and **PennyLane**.

---

# Repository Structure

```text
QDFT/
│
├── Energy+Density/
│   ├── README.md
│   ├── LDA-RS_MP2_spin.py
│   ├── OtherFunctional_MP2_spin.py
│   ├── tuned_MP2_spin.py
│   └── tuned_MP2.py
│
└── Energy/
    ├── HomoLumo/
    ├── MP2init/
    └── ROKS_RKS/
```

---

# Repository Components

## Energy+Density/

Contains the **validated production implementation** of the QDFT framework.

Features include

- Dual Energy + Density convergence
- UCCSD Ansatz
- MP2 initialization
- Active-space spin Hamiltonian correction
- Adaptive DIIS damping
- Production benchmark database

This directory should be used for all production VQE+DFT calculations.

**See:** `Energy+Density/README.md`

---

## Energy/

Contains specialized workflows that employ **energy-only convergence**.

These scripts primarily support algorithm development, benchmarking, and methodological studies.

Subdirectories include:

### HomoLumo/

Utilities for computing

- HOMO energies
- LUMO energies
- HOMO-LUMO gaps

for both orthogonal and non-orthogonal basis sets.

---

### MP2init/

Reference implementation for studying the effect of MP2 initialization on VQE convergence.

---

### ROKS_RKS/

Restricted and Restricted Open-Shell Kohn-Sham workflows for open-shell molecular systems.

---

# Production Workflow

The recommended production workflow is

1. Generate the required spin-penalty coefficient using the **Spin-Penalty β Calculator**.
2. Select the appropriate production implementation from `Energy+Density/`.
3. Perform calculations using the default production parameters.
4. Validate convergence of both energy and density.
5. Compare results with the production benchmark database when available.

---

# Notes

- The production implementation of the QDFT framework resides entirely within the `Energy+Density` directory.
- Energy-only workflows are retained for benchmarking, algorithm development, and methodological comparisons.
- The framework supports multiple exchange-correlation functionals, active-space configurations, and convergence strategies through the specialized modules distributed throughout the repository.
