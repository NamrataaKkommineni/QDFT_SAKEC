# Energy + Density Production Framework

This directory contains the **production implementation** of the hybrid **Quantum-Classical Density Functional Theory (QDFT)** embedding framework.

These scripts constitute the validated production workflows for **VQE+DFT embedding calculations** and represent the most mature implementations within the repository.

Unlike the experimental and benchmarking codes distributed throughout the project, every implementation in this directory employs **dual self-consistency**, requiring simultaneous convergence of both

- **Total Electronic Energy**
- **Electron Density Matrix** (measured using the Frobenius norm)

before terminating the self-consistent embedding cycle.

---

# Production Configuration

Unless otherwise specified, all production calculations use the following computational configuration.

| Parameter | Value |
|-----------|-------|
| Active Space | CAS(6e,6o) |
| Quantum Ansatz | UCCSD |
| Classical Driver | PySCF |
| Optimizer | L-BFGS-B |
| Initial Guess | MP2 |
| Spin Hamiltonian | Enabled (except `tuned_MP2.py`) |
| Convergence Criterion | Energy + Density |
| Density Metric | Frobenius Norm |
| Adaptive Damping | DIIS |
| `base_alpha` | 0.75 |
| `diis_start` | 9 |
| `diis_space` | 3 |

These parameters were selected through extensive convergence testing and are recommended for all production calculations.

---

# Directory Contents

## `LDA-RS_MP2_spin.py`

Production implementation using the **Range-Separated Local Density Approximation (LDA-RS)**.

**Features**

- UCCSD Ansatz
- MP2 initialization
- Active-space spin Hamiltonian correction
- Adaptive DIIS damping
- Dual Energy + Density convergence
- Explicit range-separation parameter (`ω`)

---

## `OtherFunctional_MP2_spin.py`

General production workflow supporting conventional exchange-correlation functionals.

**Supported Functionals**

- LDA
- PBE
- B3LYP
- CAM-B3LYP
- LRC-ωPBE

**Features**

- UCCSD Ansatz
- MP2 initialization
- Active-space spin Hamiltonian correction
- Adaptive DIIS damping
- Dual Energy + Density convergence

---

## `tuned_MP2_spin.py`

Production implementation using a **tuned CAM-B3LYP** exchange-correlation functional.

**Features**

- Tuned CAM-B3LYP
- UCCSD Ansatz
- MP2 initialization
- Active-space spin Hamiltonian correction
- Adaptive DIIS damping
- Dual Energy + Density convergence

---

## `tuned_MP2.py`

Baseline implementation identical to `tuned_MP2_spin.py`, except that the active-space spin Hamiltonian penalty is disabled.

This script serves as the control implementation for evaluating the influence of spin-state preservation on convergence behavior and electronic energies.

---

# Convergence Strategy

At every embedding iteration, the framework simultaneously evaluates

1. Total electronic energy
2. Electron density matrix

Self-consistency is declared only when **both quantities** satisfy their respective convergence thresholds.

Compared with conventional energy-only convergence, this dual criterion improves numerical stability and minimizes false convergence arising from oscillatory density updates.

---

# Spin-State Preservation

All production workflows, with the exception of `tuned_MP2.py`, employ an active-space spin Hamiltonian of the form

```math
H_{\mathrm{penalized}}
=
H_{\mathrm{native}}
+
\beta
\left(
\hat{S}^{2}
-
\langle S^{2}\rangle_{\mathrm{target}}
\right)
```

to suppress artificial spin contamination while preserving the desired spin multiplicity throughout the embedding procedure.

The required molecule-specific **β** coefficients can be generated using the **Automated Spin-Penalty β Coefficient Calculator** included elsewhere in this repository.

---

# Benchmark Dataset

The production implementations have been benchmarked on a diverse set of molecular systems, including

- Benzene
- Pyridine
- Naphthalene
- Anthracene
- Phenanthrene
- Tetracene
- Pentacene
- Pyrene
- Chrysene
- Triphenylene
- Benz[a]anthracene
- Perylene
- Porphyrin

using multiple exchange-correlation functionals:

- Tuned CAM-B3LYP
- B3LYP
- CAM-B3LYP
- LRC-ωPBE
- LDA-RS

The benchmark database includes

- QDFT (VQE + DFT) energies
- DFT+FCI reference calculations
- Classical DFT reference calculations
- CCSD(T) reference energies
- Experimental excitation energies
- Active-space (CAS) comparisons
- Excitation-gap analysis
- Runtime measurements
- Functional performance comparisons

---

# Complete Benchmark Database

The complete benchmark dataset is maintained separately as a Google Sheets document to allow continuous updates as additional molecular systems and validation studies are completed.

**QDFT Benchmark Database**

📊 https://docs.google.com/spreadsheets/d/1jqu6-lq_od3toY4zbfOZyQ7kjna7TH3r5ojPXu2AEIE

---

# Recommended Workflow

For production calculations:

1. Select the desired exchange-correlation functional.
2. Compute the appropriate **β** coefficient using the Spin-Penalty Calculator.
3. Use the default production parameters unless benchmarking alternative configurations.
4. Verify convergence of both the total energy and density matrix.
5. Compare the results with the benchmark database whenever applicable.

---

# Notes

- All production calculations employ the **UCCSD** ansatz.
- MP2 initialization is used throughout all production workflows.
- The default optimizer is **L-BFGS-B**.
- Adaptive DIIS parameters (`base_alpha = 0.75`, `diis_start = 9`, `diis_space = 3`) were determined through empirical convergence studies.
- The implementations in this directory constitute the validated production reference for the broader QDFT framework.
