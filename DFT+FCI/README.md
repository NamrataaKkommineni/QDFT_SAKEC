DFT + FCI Embedding Framework


This repository folder hosts the core implementations of a hybrid Density Functional Theory + Full Configuration Interaction (DFT+FCI) embedding framework.


Unlike the VQE-based execution streams, all files in this module swap out the variational quantum optimization loop for a classical, exact diagonalization approach. They utilize the NumPyMinimumEigensolver to capture the exact active space ground state energy within the self-consistent field (SCF) embedding loop, serving as an exact baseline for your hybrid embedding workflows.

Core Modules & Functional Breakdown


1. Energy + Density Convergence (/Energy+Density/)


Scripts within this folder check both Total Energy and the Electron Density Matrix Convergence to establish self-consistency. They are primarily focused on implementing and comparing different spin-state preservation methodologies and advanced stabilization techniques.


FCI_ChupinDiis_NoSpinProtection.py: Employs an advanced Direct Inversion in the Iterative Subspace (DIIS) damping scheme based on the Chupin paper architecture. This acts as an adaptive DIIS framework where the triggering criteria and dynamic subspace allocation are calculated on the fly rather than using fixed iteration milestones. It does not include any spin penalty or spin filtering adjustments.


FCI_spinFilter.py: Handles spin symmetry preservation by implementing a post-evaluation state filtering patch. While mathematically valid, this filtering technique is computationally slower than applying an active-space penalty directly to the Hamiltonian operator. It does not use the spin Hamiltonian penalty patch.


FCI_spinHamiltonian.py: Integrates the optimized spin Hamiltonian penalty code patch to smoothly enforce spin-state constraints during convergence. This script handles standard exchange-correlation functionals (e.g., B3LYP, PBE, LRC_wPBE, standard LDA) and does not include a range-separation parameter $\omega$.


FCI_spinHamiltonian_LDA-RS.py: Combines the active-space spin Hamiltonian penalty patch with the range-separated Local Density Approximation (LDA-RS), explicitly enforcing the inclusion of the range-separation parameter $\omega$.


FCI_tuned_spinHamiltonian.py: Applies the active-space spin Hamiltonian penalty patch to a customized, optimally tuned CAM-B3LYP functional pipeline.




2. Energy-Only Convergence Suites (/Energy/)


Scripts within this folder optimize the embedding loop by tracking and evaluating only Total Energy for SCF convergence criteria.


Symmetry Note: No scripts in this directory contain spin filtering patches or spin Hamiltonian penalty constraints.


DFT+FCI_LDA-RS.py: Tailored for the LDA-RS functional configuration. It strictly requires and includes the range-separation parameter $\omega$.


DFT+FCI_LDA.py: General pipeline for handling standard exchange-correlation functionals (such as standard LDA, B3LYP, standard CAM-B3LYP, PBE, etc.) that do not utilize the range-separation parameter $\omega$.


FCI_camb3lyp_tuned.py: Dedicated energy-only convergence pipeline configured for the tuned CAM-B3LYP functional.


Functional-Specific Parameter Rules


LDA-RS Configurations: Every script incorporating the LDA-RS functional matrix must declare and process the range-separation parameter $\omega$.


Alternative Functional Configurations: All alternative implementations (LDA, B3LYP, PBE, etc.) bypass the $\omega$ input.


