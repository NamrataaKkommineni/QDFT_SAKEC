Automated Spin-Penalty $\beta$-Coefficient Calculator


This directory contains a standalone utility script designed to automate the calculation of the optimal shifting weight parameter ($\beta$) for active-space spin-penalty Hamiltonians. This code evaluates a series of Polycyclic Aromatic Hydrocarbons (PAHs) and structural isomers to pre-determine individual spin splitting profiles before launching hybrid quantum-classical embedding loops.


When modeling open-shell molecular states (such as radical doublet anions or cations) in a restricted active space, artificial spin contamination frequently compromises the numerical accuracy of the quantum solver. To force the solver to track the true target spin multiplicity, an energy penalty operator can be appended to the active-space electronic Hamiltonian:

$$H_{\text{penalized}} = H_{\text{native}} + \beta (\hat{S}^2 - \langle S^2 \rangle_{\text{target}}) $$


To ensure that the ground state experiences zero penalty while unphysical higher-spin contaminants are aggressively shifted upward, the scaling factor $\beta$ must be carefully calibrated. This script automates that calibration by calculating the Singlet-Triplet vertical excitation gap ($\Delta E_{\text{S-T}}$) of the neutral system.


Following the established Kuroiwa and Nakagawa parameterization methodology, the coefficient is defined by mapping the energy gap to the minimum squared spin configuration boundary ($C_{\text{min}}^2 = 0.5625$ for an $S=0$ to $S=1$ transition):


$$\beta = \frac{E_{\text{triplet}} - E_{\text{singlet}}}{C_{\text{min}}^2} = \frac{\Delta E_{\text{S-T}}}{0.5625}$$


Dual-State Quantum Chemistry Driver: The execution script automatically provisions and runs two separate classical self-consistent field (SCF) kernels via PySCF:A Restricted Kohn-Sham (dft.RKS) framework tracking the closed-shell Singlet state ($\text{spin} = 0$).An Unrestricted Kohn-Sham (dft.UKS) framework tracking the open-shell Triplet state ($\text{spin} = 2$, representing 2 unpaired electrons).


Configurable Functional Matrix: Defaults to the range-separated hybrid functional CAM-B3LYP to ensure a balanced description of exchange interactions across large conjugated rings. This can be directly swapped out for specialized custom functional strings if required.


Upon execution, the script performs sequential SCF energy evaluations and outputs a cleanly formatted markdown-ready results table directly to stdout. Table prints out the absolute ground energies of both spin states, isolates the energy gap, and computes the exact $\beta$ value to be hardcoded into your DFTEmbeddingSolver patches. Use the generated values in the final column to overwrite the beta inside your production active-space scripts (any file than has 'spin' in name) corresponding to the molecule being simulated.