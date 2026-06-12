# Quantum Chemistry Reference Calculations 

This repository contains benchmarking scripts for electronic structure methods using PySCF via Qiskit Nature. These scripts calculate reference energies and timings across various methods (HF, MP2, DFT, CCSD, CASSCF, etc.).

## Files Overview

* **`ExactCode_v1.py` (Basic Restricted Systems)**
    * Calculates reference energies and timings for restricted (closed-shell) systems.
    * Pre-configured with multiple linear alkane geometries (H2, propane, pentane, heptane, nonane) for easy scaling tests.
    * Runs standard methods including HF, MP2, DFT (LDA, VWN), CCSD, CASCI, and CASSCF.

* **`ExactCode_v2.py` (Advanced Open-Shell & Custom Systems)**
    * A more flexible, configurable template that supports open-shell systems.
    * Dynamically handles charge and spin (switching to ROHF/ROKS when needed).
    * Includes advanced methods like CCSD(T) and Range-Separated DFT (LDA-RS).
    * Outputs a summarized benchmark table comparing all methods for easy data collection.
