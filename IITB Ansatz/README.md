# Adaptive Excitation-Selected IITB Ansatz Framework

This directory contains implementations of the **IITB Compass Ansatz** within the hybrid Quantum-Classical Density Functional Theory (QDFT) embedding framework.

Unlike conventional UCCSD-based workflows that include the complete excitation manifold from the outset, these implementations employ an **adaptive excitation-screening and ranking procedure** to construct a compact, chemically motivated ansatz. By identifying and prioritizing the most energetically significant excitations before the final Variational Quantum Eigensolver (VQE) optimization, the framework reduces the effective parameter space while preserving the dominant correlation effects of the active-space electronic structure.

The IITB Ansatz workflows operate within the same self-consistent DFT embedding framework used throughout the broader QDFT ecosystem and utilize **energy-only convergence criteria** during embedding iterations.

---

# Framework Overview

The IITB Ansatz framework combines:

* Classical Density Functional Theory (DFT)
* Variational Quantum Eigensolver (VQE)
* IITB Compass Ansatz
* Adaptive excitation screening
* Importance-based excitation ranking
* Self-consistent embedding cycles

The primary objective of this directory is to investigate whether a reduced, chemically informed ansatz can achieve comparable accuracy while lowering optimization complexity relative to full UCCSD implementations.

---

# Convergence Strategy

Unlike the production workflows located in the `VQE+DFT(QDFT)/Energy+Density` directory, IITB Ansatz calculations employ an **energy-only convergence criterion**.

Convergence is achieved when successive embedding iterations satisfy the prescribed total energy threshold.

The electron density matrix is not explicitly included in the convergence evaluation.

---

# Adaptive Excitation Screening Algorithm

A distinguishing feature of this framework is its automated excitation-selection procedure.

Rather than optimizing all excitations simultaneously, the workflow first evaluates the importance of each double excitation independently before constructing the final ansatz.

---

## Stage 1: Double Excitation Pool Generation

The framework begins by generating the complete set of double excitations from a UCC excitation manifold:

```python
UCC(..., excitations='d')
```

These operators form the candidate excitation pool.

---

## Stage 2: Individual Excitation Optimization

Each double excitation is optimized independently using a single-parameter VQE calculation.

For every excitation, the algorithm evaluates:

* Initial expectation value
* Optimized energy
* Optimal variational parameter

The energy reduction produced by each excitation is then calculated.

---

## Stage 3: Excitation Pruning

Excitations that provide negligible energy lowering are automatically discarded.

The pruning criterion is based on:

```math
|E_{\mathrm{initial}} - E_{\mathrm{optimized}}|
```

Only excitations producing a meaningful energetic contribution are retained.

This process eliminates chemically insignificant operators from the final ansatz.

---

## Stage 4: Importance Ranking

The surviving excitations are ranked according to the magnitude of their optimized amplitudes:

```math
|\theta_{\mathrm{optimal}}|
```

The rationale is that larger optimized amplitudes generally correspond to more important electron-correlation effects.

The excitation pool is sorted in descending order of parameter magnitude before ansatz construction.

---

## Stage 5: Singles Recovery

Following double-excitation screening, the complete set of single excitations is appended.

This preserves:

* Orbital relaxation effects
* Reference-state flexibility
* Variational completeness

while retaining the compactness gained from double-excitation pruning.

---

## Stage 6: Final Ansatz Construction

The final variational circuit is constructed using:

1. Ranked double excitations
2. Complete single-excitation operators

The optimized amplitudes obtained during the screening phase are used as the initial parameter guess for the final VQE optimization.

This provides a physically motivated starting point compared to random initialization strategies.

---

# Advantages of the Screening Strategy

Compared with a conventional UCCSD workflow, the excitation-selection framework offers:

* Reduced parameter count
* Smaller optimization landscape
* Faster convergence behavior
* Physically motivated parameter initialization
* Elimination of chemically insignificant excitations
* Lower computational overhead during optimization

The resulting ansatz retains the dominant correlation mechanisms while reducing unnecessary variational degrees of freedom.

---

# Directory Contents

## `IITB_Ansatz.py`

General-purpose implementation of the adaptive IITB Ansatz framework configured for standard exchange-correlation functionals.

### Supported Functionals

* B3LYP
* PBE
* LDA
* CAM-B3LYP
* Other PySCF-supported functionals

### Features

* IITB Compass Ansatz
* Adaptive excitation screening
* Importance-ranked doubles excitations
* Singles recovery procedure
* Energy-only convergence
* Self-consistent DFT embedding

### Purpose

Serves as the primary implementation for evaluating excitation-selected ansatz behavior across conventional exchange-correlation functionals.

---

## `VQE_camb3lyp_tuned.py`

Specialized implementation configured for a tuned CAM-B3LYP exchange-correlation functional.

### Features

* IITB Compass Ansatz
* Adaptive excitation ranking
* Tuned CAM-B3LYP functional
* Importance-selected doubles excitations
* Energy-only convergence

### Purpose

Designed for studies involving tuned range-separated hybrid functionals and direct comparison against equivalent UCCSD-based workflows.

---

# Comparison with Standard UCCSD

| Feature                  | UCCSD Workflow           | IITB Framework                      |
| ------------------------ | ------------------------ | ----------------------------------- |
| Excitation Pool          | Full excitation manifold | Importance-selected excitation pool |
| Double Excitations       | All included             | Screened and ranked                 |
| Parameter Initialization | Zero/random/MP2          | Physically optimized amplitudes     |
| Optimization Complexity  | Higher                   | Reduced                             |
| Circuit Compactness      | Lower                    | Higher                              |
| Convergence Efficiency   | System-dependent         | Improved for many systems           |

---

# Recommended Usage

Use this directory when:

* Benchmarking alternative ansatz architectures
* Investigating parameter-reduction techniques
* Comparing IITB and UCCSD convergence behavior
* Studying excitation-selection methodologies
* Evaluating compact ansatz constructions for larger active spaces

---

# Production Notes

The IITB implementations should be viewed as ansatz-development and benchmarking workflows within the broader QDFT framework.

While they can substantially reduce optimization complexity through excitation screening, the production reference implementations for robust energy-density convergence remain those located in:

```text
VQE+DFT(QDFT)/Energy+Density/
```

which contain the most mature convergence protocols, damping schemes, spin-state corrections, and self-consistency procedures currently available within the QDFT ecosystem.

The IITB framework is best suited for investigating compact variational circuit constructions and importance-selected excitation strategies for quantum embedding calculations.
