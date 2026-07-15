# Adaptive Excitation-Selected IITB Ansatz Framework

This directory contains implementations of the **IITB Compass Ansatz** within the hybrid Quantum-Classical Density Functional Theory (QDFT) embedding framework.

Unlike conventional UCCSD-based workflows that optimize the complete excitation manifold simultaneously, the IITB framework employs an **adaptive excitation-selection strategy** to construct compact, chemically motivated variational circuits. The methodology first identifies the most energetically significant excitations through a sequence of screening VQE calculations before assembling the final ansatz.

The framework has since evolved into two complementary implementations:

* **Singles–Doubles (SD) Ansatz**, based on adaptive excitation pruning and ranking.
* **Pseudo-Triples Ansatz**, which approximates triple excitations through the coupling of carefully selected double excitations.

Both workflows operate within the same self-consistent DFT embedding framework and employ **energy-only convergence** during embedding iterations.

---

# Framework Overview

The IITB Ansatz framework combines:

* Classical Density Functional Theory (DFT)
* Variational Quantum Eigensolver (VQE)
* IITB Compass Ansatz
* Adaptive excitation screening
* Importance-based excitation ranking
* Self-consistent embedding cycles

The primary objective of this directory is to investigate whether compact, chemically informed ansatz constructions can reduce optimization complexity while maintaining the dominant electron-correlation effects of conventional UCCSD approaches.

---

# Convergence Strategy

Unlike the production implementations located in:

```text
VQE+DFT(QDFT)/Energy+Density/
```

the IITB workflows employ an **energy-only convergence criterion**.

Embedding iterations continue until the prescribed total energy convergence threshold is satisfied. Electron density convergence is not explicitly monitored.

---

# Directory Structure

```text
IITB Ansatz/
│
├── singles_doubles/
│   ├── IITB_sd_otherFunctional.py
│   ├── IITB_sd_lda_rs.py
│   └── IITB_camb3lyp_tuned.py
│
└── Triples/
    ├── IITB_otherFunctional.py
    └── IITB_Lda_rs.py
```

---

# 1. Singles–Doubles Framework (`singles_doubles/`)

This implementation constructs a compact variational ansatz through an adaptive excitation-selection procedure.

Rather than optimizing every double excitation simultaneously, each double excitation is first evaluated independently. Only those that produce meaningful energy lowering are retained for the final variational circuit.

---

## Adaptive Excitation Screening Algorithm

### Stage 1 – Double Excitation Pool Generation

A complete doubles excitation manifold is generated using

```python
UCC(..., excitations="d")
```

These operators form the initial candidate pool.

---

### Stage 2 – Individual Excitation Optimization

Each double excitation is optimized independently using a one-parameter VQE calculation.

For every excitation the framework evaluates:

* Initial expectation value
* Optimized energy
* Optimal variational parameter

---

### Stage 3 – Excitation Pruning

Excitations producing negligible energy lowering are discarded according to

```math
|E_{\mathrm{initial}}-E_{\mathrm{optimized}}|
```

Only energetically significant operators are retained.

---

### Stage 4 – Importance Ranking

The remaining double excitations are ranked according to

```math
|\theta_{\mathrm{optimal}}|
```

where larger optimized amplitudes indicate greater correlation importance.

The excitation pool is sorted before constructing the final variational circuit.

---

### Stage 5 – Singles Recovery

Following double-excitation screening, the complete set of single excitations is appended.

This preserves:

* Orbital relaxation
* Variational flexibility
* Reference-state accuracy

while maintaining a reduced doubles space.

---

### Stage 6 – Final Ansatz Assembly

The final ansatz consists of:

* Ranked double excitations
* Complete single excitations

The optimized amplitudes obtained during the screening stage are reused as the initial parameter vector for the final VQE optimization.

This provides a physically motivated initial guess and substantially improves convergence compared to random initialization.

---

## Directory Contents

### `IITB_sd_otherFunctional.py`

General-purpose adaptive Singles–Doubles implementation for conventional exchange-correlation functionals.

Supported examples include:

* B3LYP
* PBE
* LDA
* CAM-B3LYP
* Other PySCF-supported functionals

---

### `IITB_sd_lda_rs.py`

Adaptive Singles–Doubles implementation configured for the range-separated Local Density Approximation (LDA-RS).

This workflow explicitly requires the range-separation parameter

```text
ω
```

during Hamiltonian construction.

---

### `IITB_camb3lyp_tuned.py`

Adaptive Singles–Doubles implementation configured for a tuned CAM-B3LYP exchange-correlation functional.

This version combines:

* Adaptive excitation screening
* Tuned CAM-B3LYP
* Energy-only embedding convergence

within the same reduced variational framework.

---

# 2. Pseudo-Triples Framework (`Triples/`)

The Triples implementation extends the adaptive Singles–Doubles methodology by introducing an approximate treatment of triple excitations.

Instead of explicitly constructing conventional UCC triple excitation operators—which dramatically increase circuit depth and optimization cost—the framework synthesizes the physical effect of triple excitations by coupling carefully selected double excitations.

This approach recovers additional electron correlation while avoiding the computational expense associated with true UCC triples.

---

## Pseudo-Triples Construction Algorithm

### Stage 1 – Double Excitation Screening

The workflow begins with the same adaptive pruning and ranking procedure employed by the Singles–Doubles implementation.

Only the most energetically important double excitations are retained.

---

### Stage 2 – Custom Excitation Generation

Two specialized excitation libraries are constructed:

* **Sh operators**
* **Sp operators**

These custom excitations are represented as additional double excitations and transformed into Pauli operators for subsequent variational optimization.

---

### Stage 3 – Orbital Coupling

Each retained double excitation is compared against the custom Sh and Sp excitation pools.

Coupling is permitted only when predefined occupied- or virtual-orbital overlap conditions are satisfied.

This orbital-overlap criterion produces the effective behavior of triple excitations while maintaining the computational complexity of coupled double excitations.

---

### Stage 4 – Secondary VQE Screening

Every candidate coupled excitation undergoes an additional two-parameter VQE optimization.

Only those combinations that further reduce the total energy beyond the predefined threshold are incorporated into the final ansatz.

This secondary screening prevents unnecessary circuit growth while preserving only beneficial coupled excitations.

---

### Stage 5 – Final Ansatz Assembly

The completed variational circuit consists of:

* Screened double excitations
* Validated Sh operators
* Validated Sp operators
* Complete single excitations

The optimized amplitudes obtained during the screening stages are reused to initialize the final VQE optimization, reducing convergence time and improving optimization stability.

---

## Directory Contents

### `IITB_otherFunctional.py`

Pseudo-Triples implementation configured for standard exchange-correlation functionals.

Supported examples include:

* B3LYP
* PBE
* LDA
* CAM-B3LYP
* Other PySCF-supported functionals

---

### `IITB_Lda_rs.py`

Pseudo-Triples implementation configured for the LDA-RS exchange-correlation functional.

This workflow explicitly requires the range-separation parameter

```text
ω
```

during Hamiltonian construction.

---

# Comparison of Available Frameworks

| Feature                          | Singles–Doubles | Pseudo-Triples |
| -------------------------------- | :-------------: | :------------: |
| Adaptive excitation screening    |        ✓        |        ✓       |
| Double-excitation pruning        |        ✓        |        ✓       |
| Importance-based ranking         |        ✓        |        ✓       |
| Singles recovery                 |        ✓        |        ✓       |
| Sh/Sp custom operators           |        ✗        |        ✓       |
| Orbital-overlap coupling         |        ✗        |        ✓       |
| Secondary VQE validation         |        ✗        |        ✓       |
| Triple-correlation approximation |        ✗        |        ✓       |
| Circuit depth                    |      Lower      |    Moderate    |
| Correlation recovery             |       Good      |     Higher     |

---

# Advantages of the IITB Ansatz Framework

Compared with conventional UCCSD workflows, the IITB implementations provide:

* Reduced variational parameter count
* Smaller optimization landscape
* Physically motivated parameter initialization
* Automatic elimination of chemically insignificant excitations
* Lower circuit complexity
* Faster convergence behavior
* Flexible extension toward higher-order correlation through the Pseudo-Triples framework

---

# Production Notes

The IITB implementations are intended as alternative ansatz-development and benchmarking workflows within the broader QDFT ecosystem.

While these approaches significantly reduce optimization complexity through adaptive excitation selection, the production reference implementations for robust energy-density convergence remain those located in:

```text
VQE+DFT(QDFT)/Energy+Density/
```

which contain the most mature convergence protocols, damping strategies, spin-state corrections, and self-consistency procedures.

The Singles–Doubles implementation is recommended for compact ansatz construction and efficient variational optimization, whereas the Pseudo-Triples framework extends this methodology by recovering additional electron-correlation effects through adaptive coupling of screened double excitations, providing a computationally efficient approximation to higher-order excitation physics.
