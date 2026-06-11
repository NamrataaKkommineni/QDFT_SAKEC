QDFT Anion Embedding Module (Doublet Open-Shell Framework)

This directory contains production-grade implementations of a hybrid Quantum-Classical DFT Embedding (QDFT) framework explicitly tailored for anion molecular states (open-shell doublet configurations, $S = 1/2$). These scripts integrate Qiskit Nature (using a native UCCSD ansatz optimized via L_BFGS_B) with a classical PySCF driver. Rather than checking energy alone, self-consistency is governed by a dual-metric convergence suite: Energy and Density Criterion.


While this module is explicitly configured for open-shell anion configurations, the underlying framework is fully compatible with cation molecular runs as well. Transitioning the code to model a cation requires modifying a few key physical and electronic parameters within the simulation setup of the _main() execution block. Specifically, you must reduce the total electron count to reflect the positive charge state by adjusting active_num_electrons, along with updating the specific alpha and beta spin counts (num_alpha, num_beta) and the num_particles tuple to match the target doublet or singlet state. Additionally, within the PySCF driver initialization, the charge argument must be toggled to 1 (or the appropriate positive integer), and the overall system spin must be updated to align with the new electron distribution.

Folder Contents & Functional Distinctions


This folder contains two main executable scripts, separated by how they construct the classical exchange-correlation (XC) functional matrix and handle spin contamination:


anion_otherFunctional.py: Configured for standard exchange-correlation functionals via PySCF (B3LYP, lrc_wpbe, etc). It utilizes the ROKS (Restricted Open-Shell Kohn-Sham) method to manage the open-shell system.


anion_tuned_spin.py: Features a fully manual, universally parameterized Tuned CAM-B3LYP string initialization. It explicitly bypasses the standard LibXC driver limits by dynamically injecting custom short-range/long-range Hartree-Fock fractions and Becke88 weights directly into the PySCF kernel. Integrates an active-space Spin Penalty Hamiltonian Patch. Because doublet anions are highly prone to artificial spin-state contamination, this script evaluates the total spin angular momentum operator ($\hat{S}^2$) and appends an energy penalty to the Hamiltonian for excited spin states. The penalty is offset by an identity shift ($s(s+1) = 0.75$) so the true doublet ground state experiences zero penalty. 


