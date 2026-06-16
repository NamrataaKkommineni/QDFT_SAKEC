# Automated Spin-Penalty β Coefficient Calculator

This directory contains a standalone utility for automatically determining the optimal **spin-penalty scaling coefficient** (`β`) used in active-space spin-constrained Hamiltonians.

The script evaluates a set of molecular systems—including **Polycyclic Aromatic Hydrocarbons (PAHs)** and related structural isomers—to estimate spin-state energy separations prior to launching hybrid quantum-classical embedding calculations.

The resulting `β` values can be directly incorporated into production DFT+VQE and DFT+FCI workflows to improve spin-state fidelity and suppress artificial spin contamination.

---

# Purpose

Open-shell molecular systems, such as:

* Radical anions
* Radical cations
* Doublet states
* Higher-spin excited configurations

are particularly susceptible to **spin contamination** when modeled within a restricted active space.

To enforce convergence toward the desired spin multiplicity, production embedding workflows augment the electronic Hamiltonian with a spin-dependent penalty term:

```math
H_{\mathrm{penalized}}
=
H_{\mathrm{native}}
+
\beta
\left(
\hat{S}^{2}
-
\langle S^{2} \rangle_{\mathrm{target}}
\right)
```

where:

* `H_native` is the original electronic Hamiltonian
* `β` is the spin-penalty coefficient
* `Ŝ²` is the total spin operator
* `⟨S²⟩target` is the desired spin expectation value

The penalty shifts unwanted spin states upward in energy while leaving the target spin state unaffected.

---

# Why β Calibration Matters

The effectiveness of the spin-penalty Hamiltonian depends critically on the choice of the scaling coefficient:

```math
\beta
```

If `β` is too small:

* Spin contamination may persist.
* Unwanted spin states remain energetically accessible.

If `β` is too large:

* Artificial distortions may be introduced into the active-space spectrum.
* Numerical stability may deteriorate.

This utility automates the selection of an appropriate coefficient using physically motivated spin-state energy differences.

---

# Methodology

The implementation follows the parameterization strategy proposed by **Kuroiwa and Nakagawa**, in which the spin-penalty coefficient is derived from the neutral molecule's singlet-triplet excitation energy.

The relevant energy difference is:

```math
\Delta E_{S-T}
=
E_{\mathrm{triplet}}
-
E_{\mathrm{singlet}}
```

The penalty coefficient is then computed as:

```math
\beta
=
\frac{
E_{\mathrm{triplet}}
-
E_{\mathrm{singlet}}
}{
C_{\mathrm{min}}^{2}
}
```

For a singlet-to-triplet transition:

```math
C_{\mathrm{min}}^{2}
=
0.5625
```

yielding:

```math
\beta
=
\frac{
\Delta E_{S-T}
}{
0.5625
}
```

This formulation ensures that higher-spin contaminants are shifted sufficiently above the target spin manifold while preserving the correct ground-state energetics.

---

# Computational Workflow

The script automatically performs two independent self-consistent field (SCF) calculations using PySCF.

---

## 1. Singlet Reference State

A closed-shell Restricted Kohn-Sham calculation:

```python
dft.RKS
```

with:

```python
spin = 0
```

This calculation provides:

```math
E_{\mathrm{singlet}}
```

---

## 2. Triplet Reference State

An open-shell Unrestricted Kohn-Sham calculation:

```python
dft.UKS
```

with:

```python
spin = 2
```

corresponding to two unpaired electrons.

This calculation provides:

```math
E_{\mathrm{triplet}}
```

---

## 3. Gap and β Evaluation

After both SCF calculations complete, the script:

1. Extracts the singlet energy.
2. Extracts the triplet energy.
3. Computes the singlet-triplet gap.
4. Evaluates the corresponding spin-penalty coefficient.
5. Generates a formatted summary table.

---

# Exchange-Correlation Functional Configuration

The default implementation employs:

```text
CAM-B3LYP
```

which provides a balanced description of:

* Short-range exchange interactions
* Long-range exchange interactions
* Delocalized π-electron systems
* Extended conjugated molecular frameworks

This choice is particularly appropriate for:

* Polycyclic Aromatic Hydrocarbons (PAHs)
* Acenes
* Radical ions
* Extended conjugated systems

The functional specification can be replaced with alternative standard or custom exchange-correlation functional strings when required.

---

# Output Format

Upon completion, the script generates a markdown-ready summary table directly to standard output.

The table includes:

| Quantity       | Description                          |
| -------------- | ------------------------------------ |
| Singlet Energy | Ground-state closed-shell energy     |
| Triplet Energy | Ground-state triplet energy          |
| ΔE(S–T)        | Singlet-triplet energy gap           |
| β              | Recommended spin-penalty coefficient |

The resulting output can be copied directly into reports, notebooks, or benchmarking datasets.

---

# Integration with Production Workflows

The calculated coefficient should be transferred into production active-space embedding codes that implement spin-constrained Hamiltonians.

In general, this includes any workflow containing:

```text
spin
```

within the filename, including:

* DFT+VQE spin-penalty implementations
* DFT+FCI spin-Hamiltonian workflows
* Anion embedding frameworks
* Tuned CAM-B3LYP spin-constrained calculations

---

# Recommended Usage

For each molecular system:

1. Run this utility prior to production simulations.
2. Record the generated `β` value.
3. Update the corresponding spin-penalty parameter within the production workflow.
4. Execute the hybrid embedding calculation using the calibrated coefficient.

Because spin splittings are molecule-dependent, a unique `β` value should generally be determined for every molecular system under investigation.

---

# Notes

* The script is intended as a preprocessing utility and is not part of the self-consistent embedding loop itself.
* The generated coefficient is molecule-specific and should not be assumed transferable across unrelated systems.
* CAM-B3LYP serves as the default reference functional but can be replaced when alternative exchange-correlation treatments are desired.
* The resulting `β` values are designed for use with the active-space spin-penalty Hamiltonian implementations deployed throughout the QDFT framework.

By automating spin-gap analysis and coefficient generation, this utility provides a consistent and reproducible approach for configuring spin-constrained quantum embedding calculations across a wide range of open-shell molecular systems.
