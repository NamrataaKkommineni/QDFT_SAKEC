# Energy + Density Production Framework

This directory contains the **production implementation** of the hybrid Quantum-Classical Density Functional Theory (QDFT) embedding framework.

These scripts represent the most mature implementation of the repository and are recommended for all production VQE+DFT embedding calculations.

Unlike the experimental workflows contained elsewhere in the repository, every implementation in this directory employs **dual self-consistency**, requiring simultaneous convergence of both

- Total electronic energy
- Electron density matrix (Frobenius norm)

before terminating the embedding cycle.

---

# Production Configuration

Unless otherwise specified, all production calculations employ the following computational settings.

| Parameter | Value |
|-----------|-------|
| Active Space | CAS(6e,6o) |
| Ansatz | UCCSD |
| Optimizer | L-BFGS-B |
| Initial Guess | MP2 |
| Spin Hamiltonian | Enabled (except tuned_MP2.py) |
| Convergence | Energy + Density |
| Density Metric | Frobenius Norm |
| Adaptive Damping | DIIS |
| base_alpha | 0.75 |
| diis_start | 9 |
| diis_space | 3 |

---

# Directory Contents

## `LDA-RS_MP2_spin.py`

Production implementation using the **Range-Separated Local Density Approximation (LDA-RS)**.

Features

- UCCSD Ansatz
- MP2 initialization
- Spin Hamiltonian correction
- Adaptive DIIS damping
- Dual Energy + Density convergence
- Explicit range-separation parameter ω

---

## `OtherFunctional_MP2_spin.py`

Production implementation supporting conventional exchange-correlation functionals.

Supported functionals include

- LDA
- PBE
- B3LYP
- CAM-B3LYP
- LRC-ωPBE

Features

- UCCSD Ansatz
- MP2 initialization
- Spin Hamiltonian correction
- Adaptive DIIS damping
- Dual Energy + Density convergence

---

## `tuned_MP2_spin.py`

Production implementation using a tuned CAM-B3LYP functional.

Features

- Tuned CAM-B3LYP
- UCCSD Ansatz
- MP2 initialization
- Spin Hamiltonian correction
- Adaptive DIIS damping
- Dual Energy + Density convergence

---

## `tuned_MP2.py`

Baseline implementation identical to `tuned_MP2_spin.py` except that the active-space spin Hamiltonian penalty is disabled.

This script serves as the reference implementation for evaluating the influence of spin-state constraints on embedding calculations.

---

# Convergence Strategy

Every embedding iteration simultaneously evaluates

1. Total electronic energy
2. Electron density matrix

Self-consistency is declared only when **both quantities** satisfy their respective convergence thresholds.

Compared with energy-only convergence, this dual criterion significantly improves numerical stability and reduces false convergence arising from oscillating density matrices.

---

# Spin-State Preservation

Production workflows (except `tuned_MP2.py`) employ an active-space spin Hamiltonian

\[
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
\]

to suppress artificial spin contamination during the embedding procedure.

Molecule-specific β coefficients are generated using the **Automated Spin-Penalty β Calculator** provided elsewhere in this repository.

---

# Benchmark Dataset

The production implementation has been benchmarked across a diverse collection of molecular systems, including

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

The benchmark dataset includes

- QDFT total energies
- DFT+FCI reference calculations
- CCSD(T) reference energies
- Experimental excitation energies
- CAS active-space comparisons
- Excitation-gap analysis
- Runtime measurements
- Functional performance comparisons

---

# Complete Benchmark Database

The complete benchmark dataset is maintained separately as a Google Sheet and is continuously updated as additional molecules and benchmark studies are completed.

**Benchmark Database**

> **[<Insert Google Sheets Link Here>](https://docs.google.com/spreadsheets/d/1jqu6-lq_od3toY4zbfOZyQ7kjna7TH3r5ojPXu2AEIE/edit?usp=sharing)**

---

# Recommended Workflow

For production calculations

1. Select the desired exchange-correlation functional.
2. Compute the appropriate β coefficient.
3. Use the default production parameters unless benchmarking alternative configurations.
4. Verify convergence of both energy and density before accepting results.
5. Compare results with the benchmark database whenever possible.

---

# Notes

- All production calculations employ the UCCSD ansatz.
- MP2 initialization is used throughout.
- The default optimizer is L-BFGS-B.
- Adaptive DIIS parameters (`base_alpha = 0.75`, `diis_start = 9`, `diis_space = 3`) were selected through empirical convergence studies.
- These implementations constitute the validated production reference for the broader QDFT framework.
