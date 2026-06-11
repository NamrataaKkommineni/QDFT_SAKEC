QDFT (VQE + DFT) Simulation Framework


This repository contains the core implementations, ansatz configurations, profiling utilities, and convergence mechanics for a hybrid Quantum-Classical DFT Embedding (QDFT) framework. The codebase bridges Variational Quantum Eigensolver (VQE) workflows with self-consistent field (SCF) classical density functional theory (DFT) loops.


Core Modules & Functional Breakdown


1. Energy + Density Convergence (/Energy+Density/)

All production-grade scripts in this directory utilize the UCCSD Ansatz and enforce a strict dual-convergence metric, checking both Total Energy and the Electron Density Matrix Convergence (via Frobenius Norm) to establish self-consistency.

LDA-RS_MP2_spin.py: Implements the Local Density Approximation with Range Separation (LDA-RS). Incorporates the range-separation parameter $\omega$, a custom active-space spin Hamiltonian penalty patch, and Second-Order Møller–Plesset (MP2) state initialization.

OtherFunctional_MP2_spin.py: General pipeline for standard exchange-correlation functionals (e.g., B3LYP, CAM-B3LYP, LRC_wPBE, PBE, standard LDA). Includes the spin penalty patch and MP2 initialization. No range-separation parameter $\omega$ is required.

tuned_MP2_spin.py: Specifically optimized for a tuned CAM-B3LYP functional, leveraging the spin penalty patch and MP2 initialization.

tuned_MP2.py: A control baseline variant using tuned CAM-B3LYP with MP2 initialization, but without the active-space spin penalty constraints.



2. Energy-Only Convergence Suites (/Energy/)

Scripts within this directory execute self-consistent embedding loops that track and check only Total Energy for SCF convergence criteria.

A. IITB Ansatz (/Energy/IITB Ansatz/)

Departs from standard unitary coupled cluster expansions to implement the hardware-efficient IITB Compass Ansatz combined with a custom spatial state initialization workflow.

IITB_Ansatz.py: Framework configured for standard exchange-correlation functionals.

VQE_camb3lyp_tuned.py: Framework configured specifically for the tuned CAM-B3LYP range-separated hybrid functional.



B. Runtime Performance Profiling (/Energy/Profiling/)

Dedicated benchmarking scripts using the UCCSD Ansatz and standard random parameter initialization. These track wall-clock time and memory consumption profiles across individual components: Classical DFT, Embedding Matrix Operations, and Quantum VQE steps. Both files utilize Linear Adaptive Damping to smooth out optimization steps.

ProfiledCode_old_Basic.py (v1 Baseline): Original tracking script. Contains a known profiling bug resulting in time reporting discrepancies between stdout/out files and cluster .err logs.

ProfiledCode_new_Advanced.py (v2 Advanced): Upgraded implementation. Resolves log synchronization errors and introduces granular tracking metrics to accurately balance CPU-to-GPU timeline logs.

C. UCCSD Ansatz Sub-Suites (/Energy/UCCSD Ansatz/)

Homo-Lumo Gap Analysis (/HomoLumo/): Scripts dedicated to extracting ground-state energies alongside the Highest Occupied / Lowest Unoccupied Molecular Orbital gap.

HomoLumo_v1.py: Employs a standard hermitian solver (numpy.linalg.eigvalsh). This assumes a mathematically orthogonal basis set where atomic orbitals do not spatially overlap.

HomoLumo_v2.py: Incorporates non-orthogonal atomic orbital basis math. It retrieves the non-local overlap matrix from the classical driver (mf.get_ovlp()) and passes it as a metric to a generalized eigensolver:
scipy.linalg.eigh(fock_total, b=overlap_matrix, eigvals_only=True)


MP2 Initialization Baseline (/MP2init/):

DFT+VQE_MP2_Init.py: A clean implementation testing the baseline impact of MP2 state initialization on convergence speeds when tracking energy metrics exclusively.


Restricted vs. Restricted Open-Shell Drivers (/ROKS_RKS/):

Handles operational modes for spin-restricted closed-shell systems (RKS) and spin-restricted open-shell configurations (ROKS), specifically tailored for open-shell systems like molecular Oxygen ($O_2$).Functional Constraints (Strict Rule): For all LDA-RS variations, the range-separation parameter $\omega$ must be explicitly passed. For standard alternative functionals (B3LYP, PBE, etc.), $\omega$ is excluded.


Critical Algorithmic Notes


VQE Callbacks & Damping Mechanics
The repository contains varied implementations of VQE callback functions and mathematical damping algorithms distributed across older scripts and testing directories.

Important Deployment Guardrail: For the most accurate, stable, and up-to-date VQE callback definitions and density matrix/energy damping mechanics, always reference the production codes located in the Energy+Density folder. These scripts contain the optimized convergence routines developed for this framework.
