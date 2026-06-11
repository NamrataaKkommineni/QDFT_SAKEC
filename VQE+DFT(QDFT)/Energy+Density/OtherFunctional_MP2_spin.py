# Developed By Namrataa Kkommineni under AICTE Industry Fellowship Program SH-2025
# Inspiration derived from Rossmanek 2020 quantum embedding paper https://arxiv.org/pdf/2009.01872

# ================================================================
#  IMPORTS & GLOBAL SETTINGS
# ================================================================
import numpy as np
import scipy.linalg
import matplotlib
# Use non-interactive backend for headless environments (CI / servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Qiskit / Qiskit Nature and PySCF imports (only those used)
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, BasisTransformer
from qiskit_nature.second_q.mappers import TaperedQubitMapper, ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.second_q.properties import ElectronicDensity
from qiskit_nature.settings import settings
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint
# from qiskit_nature.second_q.circuit.library import UCC
# from qiskit.circuit.library import EvolvedOperatorAnsatz

# Keep original (important) settings from your script
settings.tensor_unwrapping = False
settings.use_pauli_sum_op = False
settings.use_symmetry_reduced_integrals = True

### >>> ADDED: PROFILING IMPORTS
import time
import psutil
from collections import defaultdict

# ================================================================
#  PROFILING GLOBAL STATE
# ================================================================
### >>> ADDED: PROFILING STATE
PROCESS = psutil.Process()
PROFILE = defaultdict(float)
MEMORY_SNAPSHOTS = {}

### >>> ADDED: PROFILING HELPERS
def _mem_mb():
    return PROCESS.memory_info().rss / 1024**2

def _tic(label):
    PROFILE[f"{label}_start"] = time.perf_counter()
    MEMORY_SNAPSHOTS[f"{label}_start_MB"] = _mem_mb()

def _toc(label):
    PROFILE[label] += time.perf_counter() - PROFILE[f"{label}_start"]
    MEMORY_SNAPSHOTS[f"{label}_end_MB"] = _mem_mb()


# ================================================================
#  CLASS: DFTEmbeddingSolver
# ================================================================
class DFTEmbeddingSolver:

    def __init__(
        self,
        active_space: ActiveSpaceTransformer,
        solver: GroundStateEigensolver,
        *,
        max_iter: int = 100,
        threshold: float = 1e-6,
    ) -> None:

        self.active_space = active_space
        self.solver = solver
        self.max_iter = max_iter
        self.threshold = threshold

    def solve(self, driver: PySCFDriver, mol_name: str = "Molecule", active_space_str: str = "CAS"):

        # ---- Step 1: reference DFT run (classical DFT energy baseline) ----
        ### >>> ADDED: DFT REFERENCE TIMING
        _tic("DFT_reference")
        driver.run_pyscf()
        _toc("DFT_reference")
        #driver.run_pyscf()
        E_DFT_full = driver._calc.e_tot
        print("\n=== Classical DFT Reference Energy (PySCF) ===")
        print(f"Total DFT Energy: {E_DFT_full:.8f} Ha")
        print("==============================================\n")

        # ---- Step 2: fix MO coefficients for consistent basis transformations
        (
            mo_coeff,
            mo_coeff_b,
        ) = driver._expand_mo_object(
            driver._calc.mo_coeff, array_dimension=3
        )
        basis_trafo = BasisTransformer(
            ElectronicBasis.AO,
            ElectronicBasis.MO,
            ElectronicIntegrals.from_raw_integrals(mo_coeff, h1_b=mo_coeff_b),
        )

        # ---- Step 3: construct the MO-basis 'problem' using range-separated Coulomb
        problem = driver.to_problem(basis=ElectronicBasis.MO, include_dipole=False)

        # Build total MO density (spin-adapted). We attach it to problem properties so later steps may use it.
        total_mo_density = ElectronicDensity.from_orbital_occupation(
            problem.orbital_occupations,
            problem.orbital_occupations_b,
            include_rdm2=False,
        )
        problem.properties.electronic_density = total_mo_density

        # ---- Step 4: initialize active space transformer with global sizes
        self.active_space.prepare_active_space(
            problem.num_particles,
            problem.num_spatial_orbitals,
            occupation_alpha=problem.orbital_occupations,
            occupation_beta=problem.orbital_occupations_b,
        )

        # ---- Step 5: set up initial densities

        active_density_history = [
            self.active_space.active_basis.transform_electronic_integrals(
                total_mo_density
            )
        ]

        # The inactive (environment) AO density is built by taking the total MO
        # density (transformed to AO) and subtracting the AO representation of
        # the active density. The inactive density remains fixed in our loop.
        inactive_ao_density = basis_trafo.invert().transform_electronic_integrals(
            total_mo_density
            - self.active_space.active_basis.invert().transform_electronic_integrals(
                active_density_history[-1]
            )
        )

        # ---- bookkeeping variables for iteration and outputs ----
        e_nuc = problem.hamiltonian.nuclear_repulsion_energy  # nuclear repulsion
        e_tot = driver._calc.e_tot  # current total energy (starts with reference)
        e_next = float("NaN")
        e_prev = float("NaN")
        converged = False
        n_iter = 0
        embedding_energies = []

        # ---- Step 6: iterative embedding loop ----
        while n_iter < self.max_iter:
            n_iter += 1
            ### >>> ADDED: TOTAL EMBEDDING LOOP TIMER
            _tic("Embedding_loop_total")

            # a) expand active density (reduced active-basis representation)
            #    back into the full-system MO basis so we can combine with the
            #    inactive density (which is stored in AO).
            active_mo_density = (
                self.active_space.active_basis.invert().transform_electronic_integrals(
                    active_density_history[-1]
                )
            )

            # b) transform the active MO density to AO basis for addition
            #    to the inactive AO density.
            active_ao_density = basis_trafo.invert().transform_electronic_integrals(
                active_mo_density
            )

            # c) total AO density is the sum of the inactive (environment) AO
            #    density and the active-subsystem AO density.
            total_ao_density = inactive_ao_density + active_ao_density

            # d) convert to PySCF DM (density matrix) format expected by driver._calc
            #    Handling spin-restricted vs spin-unrestricted structures:
            if basis_trafo.coefficients.beta.is_empty():
                # spin-restricted (alpha == beta)
                rho = np.asarray(total_ao_density.trace_spin()["+-"])
            else:
                # spin-unrestricted; driver expects list/array of alpha and beta
                rho = np.asarray(
                    [total_ao_density.alpha["+-"], total_ao_density.beta["+-"]]
                )

            # e) evaluate total DFT energy at the new total density.
            #    This uses PySCF's energy functional evaluated at DM `rho`.
            #e_tot = driver._calc.energy_tot(dm=rho)
            ### >>> ADDED: DFT ENERGY
            _tic("DFT_energy")
            e_tot = driver._calc.energy_tot(dm=rho)
            _toc("DFT_energy")

            # f) evaluate Fock operator (effective one-electron operator) at the current density; used to build the active-space references.
            _tic("DFT_fock")
            (
                fock_a,
                fock_b,
            ) = driver._expand_mo_object(
                driver._calc.get_fock(dm=rho),
                array_dimension=3,
            )
            _toc("DFT_fock")
            # g) update active-space transformer with current active density
            #
            self.active_space.active_density = active_mo_density
            self.active_space.reference_inactive_energy = e_tot - e_nuc
            self.active_space.reference_inactive_fock = (
                basis_trafo.transform_electronic_integrals(
                    ElectronicIntegrals.from_raw_integrals(fock_a, h1_b=fock_b)
                )
            )

            # h) reduce the full MO-basis problem to the active subsystem problem
            as_problem = self.active_space.transform(problem)

            # ==========================================================
            # 🔹 SPIN PENALTY HAMILTONIAN BUILD (Fix for VQE Contamination)
            # ==========================================================
            from qiskit_nature.second_q.properties import AngularMomentum

            # 1. Extract the native Hamiltonian Fermionic Operator
            h_op = as_problem.hamiltonian.second_q_op()

            # 2. Generate the S^2 (Angular Momentum) Fermionic Operator
            s2_property = AngularMomentum(as_problem.num_spatial_orbitals)
            s2_op = s2_property.second_q_ops()["AngularMomentum"]

            # 3. Build the Penalized Hamiltonian (beta = 0.5 is a standard weight)
            #Benzene: 0.33828
            #Naphthalene: 0.21432
            #Anthracene : 0.14648
            #Tetracene  :0.10110
            #Pentacene: 0.06941
            #Pyrene: 0.16656
            #Chrysene    :  0.20187
            #Triphenylene : 0.27670
            #Phenanthrene  :0.21763
            #Perylene : 0.12863
            beta = 0.14648
            penalized_h_op = h_op + (beta * s2_op)

            # 4. Override the problem's Hamiltonian generation before passing to the solver
            as_problem.hamiltonian.second_q_op = lambda: penalized_h_op
            # ==========================================================

            # i) solve the active-space problem
            _toc("Embedding_loop_total")
            result = self.solver.solve(as_problem)

            # ==========================================================
            # >>> ADDED: Extract Occupation Numbers from 1-RDM
            # ==========================================================
            try:
                # Extract the spatial 1-RDM (alpha + beta) for the active space
                active_1rdm = np.asarray(result.electronic_density.trace_spin()["+-"])

                # The natural occupation numbers are the eigenvalues of the 1-RDM
                occupations = scipy.linalg.eigvalsh(active_1rdm)

                # Sort in descending order (from most occupied to least occupied)
                occupations = np.sort(occupations)[::-1]

                print(f"  [Active Space Occupations]: {np.round(occupations, 4)}")
            except Exception as e:
                print(f"  [Occupation Extraction Failed]: {e}")

            # j) add the newly-computed active density to history and apply damping/mixing
            new_damped_density = self.damp_active_density(
                active_density_history + [result.electronic_density]
            )
            active_density_history.append(new_damped_density)

            # ==========================================================
            # k) DUAL CONVERGENCE CHECK (Energy + Density)
            # ==========================================================
            e_prev = e_next
            e_next = result.total_energies[0]
            embedding_energies.append(e_next)

            if n_iter > 1:
                # Helper to extract density matrix for the norm calculation
                def get_mat(dens):
                    return dens.alpha["+-"] + (dens.beta["+-"] if not dens.beta.is_empty() else 0)

                # Calculate Frobenius norm of the density change (Δρ)
                prev_mat = get_mat(active_density_history[-2])
                curr_mat = get_mat(active_density_history[-1])
                res_norm = np.linalg.norm(curr_mat - prev_mat, 'fro')

                delta_e = np.abs(e_prev - e_next)

                print(f"[Iteration {n_iter}] E = {e_next:.8f} Ha | ΔE = {delta_e:.3e} | Δρ = {res_norm:.3e}")

                # Check both conditions
                density_threshold = 1e-4
                energy_converged = delta_e < self.threshold
                density_converged = res_norm < density_threshold

                if energy_converged and density_converged:
                    print(f"[Converged] Both Energy (ΔE < {self.threshold}) and Density (Δρ < {density_threshold}) criteria met!")
                    break
                elif energy_converged:
                    print(f"  -> [Info] Energy converged, but Density (Δρ = {res_norm:.3e}) is still sloshing. Continuing...")
            else:
                print(f"[Iteration {n_iter}] E = {e_next:.8f} Ha")

        # ================================================================
        #  >>> ADDED: HOMO-LUMO Analysis Patch
        # ================================================================
        try:
            print("\n=== HOMO-LUMO Analysis ===")
            mf = driver._calc
            nocc = mf.mol.nelec[0]
            orbital_energies = mf.mo_energy

            # Adjusted to safely check for 'use_range_separation' in case it wasn't defined in __init__
            method_name = 'DFT' if getattr(self, 'use_range_separation', True) else 'HF'
            print(f"[Classical {method_name}] HOMO = {orbital_energies[nocc-1]:.6f} Ha, "
                  f"LUMO = {orbital_energies[nocc]:.6f} Ha, "
                  f"Gap = {orbital_energies[nocc]-orbital_energies[nocc-1]:.6f} Ha")

            # Embedding HOMO/LUMO from Fock_total
            total_ao_density = inactive_ao_density + active_ao_density
            rho_embed = np.asarray(total_ao_density.trace_spin()["+-"])
            fock_total = mf.get_fock(dm=rho_embed)

            overlap_matrix = mf.get_ovlp()
            eigvals = scipy.linalg.eigh(fock_total, b=overlap_matrix, eigvals_only=True)
            print(f"[Embedding] HOMO = {eigvals[nocc-1]:.6f} Ha, LUMO = {eigvals[nocc]:.6f} Ha, Gap = {eigvals[nocc]-eigvals[nocc-1]:.6f} Ha")
        except Exception as e:
            print(f"[HOMO-LUMO] Analysis failed: {e}")

        # Try to write cube files for HOMO and LUMO using PySCF cubegen
        try:
            from pyscf.tools import cubegen

            # Dynamic filenames
            homo_filename = f"{mol_name}_HOMO_{active_space_str}.cube"
            lumo_filename = f"{mol_name}_LUMO_{active_space_str}.cube"

            print("\nAttempting to write HOMO/LUMO cube files and analyze PR...")
            nocc = driver._calc.mol.nelec[0]
            homo_coeff = driver._calc.mo_coeff[:, nocc-1]
            lumo_coeff = driver._calc.mo_coeff[:, nocc]

            cubegen.orbital(driver._calc.mol, homo_filename, homo_coeff)
            cubegen.orbital(driver._calc.mol, lumo_filename, lumo_coeff)
            print(f"[Saved] {homo_filename}, {lumo_filename}")

            # ================================================================
            #  >>> ADDED: Participation Ratio (PR) Calculation
            # ================================================================
            # 1. Get the overlap matrix (S) to account for non-orthogonal basis sets
            S = driver._calc.get_ovlp()

            # 2. Compute S^(1/2) for Löwdin orthogonalization
            S_half = scipy.linalg.fractional_matrix_power(S, 0.5).real

            # 3. Transform MO coefficients to the orthogonalized basis
            homo_ortho = S_half.dot(homo_coeff)
            lumo_ortho = S_half.dot(lumo_coeff)

            # 4. Calculate PR = (sum(|c|^2))^2 / sum(|c|^4)
            # Since Löwdin orbitals are normalized, sum(|c|^2) = 1
            pr_homo = 1.0 / np.sum(np.abs(homo_ortho)**4)
            pr_lumo = 1.0 / np.sum(np.abs(lumo_ortho)**4)

            print(f"\n[Participation Ratio] HOMO = {pr_homo:.2f} (Effective orbitals participating)")
            print(f"[Participation Ratio] LUMO = {pr_lumo:.2f} (Effective orbitals participating)")

        except Exception as e:
            print(f"[Cube/PR Analysis] Failed: {e}")

        # Return active-space result (last iteration) and the energy history
        return result, embedding_energies

    # === damping helper ===
    @staticmethod
    def damp_active_density(density_history, base_alpha=0.75, diis_start=10, diis_space=4):
        """
        Simplified, customizable damping for DFT embedding.
        - base_alpha: The starting mixing value for early iterations.
        - diis_start: Iteration number where DIIS is allowed to turn on.
        - diis_space: Number of previous densities to include in the DIIS matrix.
        """
        if len(density_history) < 2:
            return density_history[-1]

        prev_density = density_history[-2]
        new_density = density_history[-1]

        # ==========================================================
        # 1. OPTIONAL DIIS ACCELERATION (Pure math, no tuning parameters)
        # ==========================================================
        if len(density_history) >= diis_start and len(density_history) >= (diis_space + 1):
            try:
                def get_mat(dens):
                    mat = dens.alpha["+-"]
                    if not dens.beta.is_empty():
                        mat = mat + dens.beta["+-"]
                    return mat

                densities = density_history[-diis_space:]
                prev_densities = density_history[-(diis_space + 1):-1]
                n = len(densities)

                errors = [get_mat(d) - get_mat(pd) for d, pd in zip(densities, prev_densities)]

                B = np.zeros((n + 1, n + 1))
                for i in range(n):
                    for j in range(n):
                        B[i, j] = np.trace(errors[i] @ errors[j].T)

                B[n, :n] = -1.0
                B[:n, n] = -1.0
                B[n, n] = 0.0

                rhs = np.zeros(n + 1)
                rhs[n] = -1.0

                B[np.diag_indices(n)] += 1e-8

                coeffs = np.linalg.solve(B, rhs)[:n]

                diis_density = None
                for c, d in zip(coeffs, densities):
                    if diis_density is None:
                        diis_density = c * d
                    else:
                        diis_density = diis_density + (c * d)

                print(f"[DIIS] Extrapolating density (coeffs: {np.round(coeffs, 2)})")
                return diis_density
            except Exception:
                pass  # Silently fall back to linear damping if matrix math fails

        # ==========================================================
        # 2. GENERALIZED LINEAR DAMPING
        # ==========================================================
        alpha = max(0.05, base_alpha / np.sqrt(len(density_history)))
        print(f"[Damping] Linear mix with alpha={alpha:.3f}")

        try:
            return (1.0 - alpha) * prev_density + alpha * new_density
        except Exception:
            return new_density

# ============================ END CLASS ============================


# ================================================================
#  MAIN: simulation setup, VQE initialization, run embedding
# ================================================================
def _main():
    """Main entry: define geometry/parameters, build driver/active-space/VQE,
    run the DFT embedding iterations, and produce plots.

    The function is intentionally structured and partitioned so it can be
    easily pasted into Overleaf verbatim with minimal additional commentary.
    """

    # ----------------------------------------------------------------
    # (A) Simulation parameters & geometries
    # ----------------------------------------------------------------
    #omega = 5.0  # range-separation parameter used in PySCF's with_range_coulomb

    # Active-space specification (tune for your system)
    active_num_spatial_orbitals = 6
    active_num_electrons = 6

    # VQE optimizer max iterations (passed to L-BFGS-B)
    #vqe_maxiter = 50

    # Geometry options (kept compact for documentation). Select the one to use.
    geometry_C6H6 = (
    "C  0.000000  1.397532  0.000000;"
    "C  1.209441  0.698476  0.000000;"
    "C  1.209441 -0.698476  0.000000;"
    "C  0.000000  -1.397532  0.000000;"
    "C  -1.209441  -0.698476  0.000000;"
    "C  -1.209441  0.698476  0.000000;"
    "H  0.000000  2.484546  0.000000; "
    "H  2.150927  1.241781  0.000000; "
    "H  2.150927  -1.241781  0.000000; "
    "H  0.000000  -2.484546  0.000000;"
    "H  -2.150927  -1.241781  0.000000;"
    "H  -2.150927  1.241781  0.000000"
    ) #benzene
    geometry_C14H10= (
    "C  5.890096  -0.722620  0.000000;"
    "C  5.890096  0.722620  0.000000;"
    "C  3.442033  -0.722608  0.000000;"
    "C  3.442033  0.722608  0.000000;"
    "C  4.666079  -1.403534  0.000000;"
    "C  4.666079  1.403534  0.000000;"
    "C  7.146040  -1.407540  0.000000;"
    "C  7.146040  1.407540  0.000000;"
    "C  2.186103  -1.407521  0.000000;"
    "C  2.186103  1.407521  0.000000;"
    "C  8.327468  -0.713406  0.000000;"
    "C  8.327468  0.713406  0.000000;"
    "C  1.004660  -0.713397  0.000000;"
    "C  1.004660  0.713397  0.000000;"
    "H  4.666140  -2.491880  0.000000;"
    "H  4.666140  2.491880  0.000000;"
    "H  7.143168  -2.495025  0.000000;"
    "H  7.143168  2.495025  0.000000;"
    "H  2.188982  -2.495007  0.000000;"
    "H  2.188982  2.495007  -0.000000;"
    "H  9.274249  -1.246632  0.000000;"
    "H  9.274249  1.246632  0.000000; "
    "H  0.057891  -1.246645  0.000000;"
    "H  0.057891  1.246645  -0.000000"
    ) #anthracene
    geometry_C10H8 = (
    "C  0.000000  0.717021  0.000000;"
    "C  0.000000  -0.717021  0.000000;"
    "C  1.245022  1.402853  0.000000;"
    "C  1.245022  -1.402853  0.000000;"
    "C  2.434130  0.708608  0.000000;"
    "C  2.434130  -0.708608  0.000000; "
    "C  -1.245022  1.402853  0.000000;"
    "C  -1.245022  -1.402853  0.000000;"
    "C  -2.434130  0.708608  0.000000;"
    "C  -2.434130  -0.708608  0.000000;"
    "H  1.242357  2.490515  0.000000;"
    "H  1.242357  -2.490515  0.000000; "
    "H  3.378831  1.245703  0.000000; "
    "H  3.378831  -1.245703  0.000000; "
    "H  -1.242357  2.490515  0.000000; "
    "H  -1.242357  -2.490515  0.000000;"
    "H  -3.378831  1.245703  0.000000;"
    "H  -3.378831  -1.245703  0.000000"
    ) #napthelene
    tetracene = (
    "C     5.570766    -0.726233     0.000000; "
    "C     5.570766     0.726233     0.000000; "
    "C     8.021910    -0.726281     0.000000; "
    "C     8.021910     0.726281     0.000000; "
    "C     3.119566    -0.725776     0.000000; "
    "C     3.119566     0.725776     0.000000; "
    "C     4.335378    -1.406442     0.000000; "
    "C     4.335378     1.406442     0.000000; "
    "C     6.806488    -1.406864     0.000000; "
    "C     6.806488     1.406864     0.000000; "
    "C     9.283097    -1.409750     0.000000; "
    "C     9.283097     1.409750     0.000000; "
    "C     1.858911    -1.409579     0.000000; "
    "C     1.858911     1.409579     0.000000; "
    "C    10.460901    -0.715530     0.000000; "
    "C    10.460901     0.715530     0.000000; "
    "C     0.680625    -0.715534     0.000000; "
    "C     0.680625     0.715534     0.000000; "
    "H     4.335391    -2.494544     0.000000; "
    "H     4.335391     2.494544     0.000000; "
    "H     6.806136    -2.494957     0.000000; "
    "H     6.806136     2.494957     0.000000; "
    "H     9.280956    -2.497175     0.000000; "
    "H     9.280956     2.497175     0.000000; "
    "H     1.861193    -2.496953     0.000000; "
    "H     1.861193     2.496953     0.000000; "
    "H    11.408401    -1.247230     0.000000; "
    "H    11.408401     1.247230     0.000000; "
    "H    -0.266731    -1.247523    -0.000000; "
    "H    -0.266731     1.247523    -0.000000"
    )  # Tetracene_optimized
    pentacene = (
    "C    7.767609    0.728831    0.000000; "
    "C    7.767950   -0.728108    0.000000; "
    "C    5.313886    0.728245    0.000000; "
    "C    5.314234   -0.728685    0.000000; "
    "C    6.540596    1.408688    0.000000; "
    "C    6.541266   -1.408559    0.000000; "
    "C   10.220343    0.728839    0.000000; "
    "C   10.220695   -0.726933    0.000000; "
    "C    2.861158    0.727098    0.000000; "
    "C    2.861501   -0.728663    0.000000; "
    "C    9.008805    1.408922    0.000000; "
    "C    9.009468   -1.407609    0.000000; "
    "C    4.072385    1.407761    0.000000; "
    "C    4.073045   -1.408764    0.000000; "
    "C   11.483662    1.412113    0.000000; "
    "C   11.484354   -1.409595    0.000000; "
    "C    1.597532    1.409778    0.000000; "
    "C    1.598207   -1.411959    0.000000; "
    "C   12.660108    0.718168    0.000000; "
    "C   12.660479   -0.715140    0.000000; "
    "C    0.421468    0.715182    0.000000; "
    "C    0.421840   -0.718122    0.000000; "
    "H    6.540319    2.496662    0.000000; "
    "H    6.541508   -2.496533    0.000000; "
    "H    9.008415    2.497000    0.000000; "
    "H    9.009600   -2.495687    0.000000; "
    "H    4.072266    2.495839    0.000000; "
    "H    4.073413   -2.496841    0.000000; "
    "H   11.481437    2.499497    0.000000; "
    "H   11.482631   -2.496988    0.000000; "
    "H    1.599197    2.497168    0.000000; "
    "H    1.600507   -2.499515    0.000000; "
    "H   13.607936    1.249340    0.000000; "
    "H   13.608552   -1.245844    0.000000; "
    "H   -0.526633    1.245865    0.000000; "
    "H   -0.525984   -1.249280    0.000000"
    ) # Pentacene_optimized (B3LYP/6-31g*)

    # Friendly filename token used in saved plots
    geometry_scan = geometry_C14H10
    # ----------------------------------------------------------------
    # (B) PySCF driver setup
    # ----------------------------------------------------------------
    # The driver wraps PySCF DFT and produces integrals / occupations used by Qiskit Nature.
    # xc_functional uses a short-range LDA + a long-range HF component parameterized by omega.
    driver = PySCFDriver(
        atom=geometry_scan,
        basis="6-31g*",
        method=MethodType.RKS,
        xc_functional="B3LYP",
        xcf_library="xcfun",
    )

    # ----------------------------------------------------------------
    # (C) Active-space transformer and VQE setup
    # ----------------------------------------------------------------
    active_space = ActiveSpaceTransformer(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_electrons=active_num_electrons,
    )

    # Qubit mapper used for the active-space qubit Hamiltonian (parity mapping +
    # tapering to exploit symmetries). This is the mapper passed to GroundStateEigensolver.
    mapper = TaperedQubitMapper(ParityMapper())


    # Build Hartree-Fock initial state and q-UCCSD ansatz sized for the active problem.
    n = active_num_electrons
    num_particles = (n // 2, n - (n // 2))

    # ---------- 🔹 CHANGE 1: Use L_BFGS_B with internal evaluation limits ----------
    from qiskit_algorithms.optimizers import L_BFGS_B
    optimizer = L_BFGS_B(maxiter=2000, maxfun=10000, ftol=1e-6)
# (keeps COBYLA option commented for reference)
# optimizer = COBYLA(maxiter=200, tol=1e-4)

# ---------- 🔹 CHANGE 2: Add a small random initial point to avoid flat gradient ----------
    #initial_point = 0.05 * (2 * np.random.rand(ansatz.num_parameters) - 1)

# Build Hartree-Fock initial state (HF) and UCCSD ansatz
    initial_state = HartreeFock(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
    )
    ansatz = UCCSD(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=initial_state,
        generalized=False,
        preserve_spin=True,
        reps=1,
    )
    #initial_point = 0.05 * (2 * np.random.rand(ansatz.num_parameters) - 1)
    #initial_point = np.zeros(ansatz.num_parameters)
    # np.random.seed(42)
    # sigma = 0.001
    # initial_point = sigma * np.random.randn(ansatz.num_parameters)
    mp2_init = MP2InitialPoint()
    mp2_init.compute(ansatz=ansatz, problem=driver.run())
    initial_point = mp2_init.to_numpy_array()

# # ---------- 🔹 CHANGE 3: Use AerEstimator (compatible with your setup) ----------
#     from qiskit_aer.primitives import Estimator as AerEstimator
    #estimator = AerEstimator()
    from qiskit.primitives import Estimator
    estimator = Estimator()     # exact expectation values, no Aer limitations
    # ---------- 🔹 VQE callback (original normal behavior) ----------
# ---------- 🔹 VQE callback with quantum timing & convergence tracking ----------

    _vqe_last_time = {"t": None}
    vqe_maxiter = 2000
    vqe_maxfun = 10000
    vqe_ftol = 1e-6
    # Track VQE history for the current embedding loop step
    vqe_history = {"evals": [], "energies": [], "step_energies": []}
    evals_per_iteration = ansatz.num_parameters + 1
    vqe_ftol = 1e-6 # Matching the ftol defined in your L_BFGS_B optimizer

    def vqe_callback(eval_count, parameters, mean, std):
        """
        VQE callback that records energy history, tracks convergence per
        optimizer iteration, and accumulates wall-clock quantum time.
        """
        # --- 1. Profiling Logic ---
        now = time.perf_counter()
        if _vqe_last_time["t"] is not None:
            PROFILE["Quantum_VQE"] += now - _vqe_last_time["t"]
        _vqe_last_time["t"] = now

        # --- 2. Iteration Tracking Logic ---
        # Reset tracking for each new VQE run within the embedding loop
        if eval_count == 1:
            vqe_history["evals"].clear()
            vqe_history["energies"].clear()
            vqe_history["step_energies"].clear()
            print("  --- Starting VQE Optimization Step ---")

        vqe_history["evals"].append(eval_count)
        vqe_history["energies"].append(mean)

        # Only print at the start of a new optimizer iteration (after gradient evaluations)
        if eval_count >= vqe_maxfun:
            print(f"  [WARNING] maxfun limit ({vqe_maxfun}) hit! Optimizer is force-quitting.")

        # ... (keep your existing eval_count % evals_per_iteration math here) ...

        if (eval_count - 1) % evals_per_iteration == 0:
            iteration = (eval_count - 1) // evals_per_iteration

            if iteration >= vqe_maxiter:
                print(f"  [WARNING] maxiter limit ({vqe_maxiter}) hit! Optimizer is force-quitting.")

            if iteration == 0:
                print(f"  [VQE] Iteration: {iteration:4d} | Energy: {mean:12.8f} Ha | Rel ΔE: N/A")
            else:
                prev_mean = vqe_history["step_energies"][-1]
                abs_delta_e = abs(mean - prev_mean)
                rel_delta_e = abs_delta_e / max(abs(prev_mean), abs(mean), 1.0)

                print(f"  [VQE] Iteration: {iteration:4d} | Energy: {mean:12.8f} Ha | Rel ΔE: {rel_delta_e:10.8f}")

                if rel_delta_e < vqe_ftol:
                    print(f"  [VQE Converged] Rel ΔE ({rel_delta_e:.3e}) is below threshold ({vqe_ftol})!")

            vqe_history["step_energies"].append(mean)

    # Build the VQE solver
    vqe_solver = VQE(
        ansatz=ansatz,
        optimizer=optimizer,
        estimator=estimator,
        callback=vqe_callback,
        initial_point=initial_point,
    )

    # Wrap VQE into the Ground State Solver
    ground_state_solver = GroundStateEigensolver(mapper, vqe_solver)

    # Instantiate the DFT embedding solver
    dft_solver = DFTEmbeddingSolver(active_space, ground_state_solver)

    # ================================================================
    #  (D) Run embedding normally
    # ================================================================
    try:
        mol_name = "Anthracene_B3LYP"  # Change this based on the geometry used (e.g., "H2O", "CO2", "Benzene", etc.)
        active_space_str = f"CAS_{active_num_electrons}e_{active_num_spatial_orbitals}o"

        result, embedding_energies = dft_solver.solve(driver, mol_name=mol_name, active_space_str=active_space_str)

        # ✅ Add this to print the final formatted result like in COBYLA
        print("\n=== DFT Embedding + VQE Results ===")
        print(result)

        # ==========================================================
        # >>> INLINE DENSITY DIFFERENCE PLOT
        # ==========================================================
        if result is not None:
            print(f"\n>>> Generating Inline Density Difference Plot for {mol_name}...")

            # 1. Recreate basis_trafo since it was deleted after the solver finished
            mo_coeff, mo_coeff_b = driver._expand_mo_object(driver._calc.mo_coeff, array_dimension=3)
            basis_trafo = BasisTransformer(
                ElectronicBasis.AO, ElectronicBasis.MO,
                ElectronicIntegrals.from_raw_integrals(mo_coeff, h1_b=mo_coeff_b)
            )

            # 2. Reconstruct Classical DFT Total Density
            problem = driver.to_problem(basis=ElectronicBasis.MO, include_dipole=False)
            total_mo_density = ElectronicDensity.from_orbital_occupation(
                problem.orbital_occupations, problem.orbital_occupations_b, include_rdm2=False
            )

            # Active space density from the VQE result
            active_mo_density = result.electronic_density

            # Transform all to AO basis
            total_ao_density = basis_trafo.invert().transform_electronic_integrals(total_mo_density)

            # Properly map the active space back to full AO
            active_ao_density = basis_trafo.invert().transform_electronic_integrals(
                active_space.active_basis.invert().transform_electronic_integrals(active_mo_density)
            )

            # Fixed inactive density (environment) from classical DFT
            dft_active_mo = active_space.active_basis.transform_electronic_integrals(total_mo_density)
            dft_active_ao = basis_trafo.invert().transform_electronic_integrals(
                active_space.active_basis.invert().transform_electronic_integrals(dft_active_mo)
            )
            inactive_ao_density = total_ao_density - dft_active_ao

            # CRITICAL: Density difference (Total Embedded - Inactive - Active VQE)
            total_embedded_ao = inactive_ao_density + active_ao_density
            delta_rho = total_embedded_ao - inactive_ao_density - active_ao_density

            # Extract the full spatial difference matrix
            full_delta_matrix = np.asarray(delta_rho.trace_spin()["+-"])

            # Sum the absolute values of the mismatch to get the total magnitude
            total_mismatch_magnitude = np.sum(np.abs(full_delta_matrix))
            print(f"  [Sanity Check] Total Density Mismatch Magnitude: {total_mismatch_magnitude:.4e}")

            # Plot on molecular plane
            plt.figure(figsize=(10, 4))

            # Ensure we don't slice out of bounds
            matrix_size = min(50, delta_rho.trace_spin()["+-"].shape[0])
            Z = delta_rho.trace_spin()["+-"][:matrix_size, :matrix_size]

            # Adjust X and Y to match Z's shape
            X, Y = np.meshgrid(np.linspace(-3, 3, matrix_size), np.linspace(-2, 2, matrix_size))

            # ==========================================================
            # 🔹 SYMMETRIC COLORMAP FIX
            # ==========================================================
            # Find the absolute maximum value to make bounds perfectly symmetric
            limit = np.max(np.abs(Z))
            if limit == 0: limit = 1e-16 # prevent division by zero

            # Use vmin and vmax to lock 0.0 to pure white
            plt.contourf(X, Y, Z, levels=20, cmap='RdBu_r', vmin=-limit, vmax=limit)
            plt.colorbar(label='Δρ (e/bohr³)')

            # 🔹 WIRED mol_name INTO TITLE AND FILENAME HERE 🔹
            plt.title(f'Density Difference: {mol_name}')
            plt.xlabel('x (bohr)'); plt.ylabel('z (bohr)')

            plt.tight_layout()
            filename = f'{mol_name}_density_diff.png'
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"[Saved] {filename}")

        # ================================================================
        #  >>> ADDED: Quantum Eigenstate Participation Ratio (PR)
        # ================================================================
        try:
            print("\n=== Quantum Eigenstate PR Analysis ===")
            from qiskit.quantum_info import Statevector

            # 1. Extract the raw VQE result from the GroundStateEigensolver wrapper
            vqe_result = result.raw_result

            # 2. Bind the optimal parameters found by VQE to the UCCSD ansatz
            optimal_circuit = ansatz.assign_parameters(vqe_result.optimal_parameters)

            # 3. Simulate the full statevector of the optimized circuit
            # Note: This works for manageable active spaces like (8,6) where qubits <= ~20.
            state = Statevector(optimal_circuit)

            # 4. Get the probabilities (|c_gamma|^2) for every basis state
            probabilities = state.probabilities()

            # 5. Calculate Eigenstate PR: 1 / sum( (|c_gamma|^2)^2 ) = 1 / sum( p^2 )
            ipr = np.sum(probabilities**2)
            pr_eigenstate = 1.0 / ipr

            print(f"[Eigenstate PR] {pr_eigenstate:.4f} (Effective configurations participating)")

            # Optional: Print the top contributing electron configurations to see the entanglement
            print("Top contributing configurations (>1% probability):")
            top_indices = np.argsort(probabilities)[::-1]
            for idx in top_indices:
                if probabilities[idx] > 0.01:
                    # Format index as a binary string representing the qubit state (e.g., |001100>)
                    bin_str = format(idx, f"0{ansatz.num_qubits}b")
                    print(f"  |{bin_str}> : Probability = {probabilities[idx]*100:.2f}%")

        except Exception as e:
            print(f"[Eigenstate PR Analysis] Failed: {e}\n(Ensure you are running a solver that supports statevector extraction)")

    except Exception as e:
        print(f"[Warning] VQE encountered issue ({e}); continuing with last results.")
        result, embedding_energies = None, []


   # ============================================================
    #  FINAL PROFILING REPORT (CORRECT DECOMPOSITION)
    # ============================================================
    hybrid_overhead = (
        PROFILE["Embedding_loop_total"]
        - PROFILE["DFT_energy"]
        - PROFILE["DFT_fock"]
    )

    print("\n================ PROFILING SUMMARY ================")
    print(f"DFT reference (SCF) time     : {PROFILE['DFT_reference']:.2f} s")
    print(f"DFT energy evaluations time : {PROFILE['DFT_energy']:.2f} s")
    print(f"DFT Fock build time         : {PROFILE['DFT_fock']:.2f} s")
    print(f"Hybrid embedding overhead  : {hybrid_overhead:.2f} s")
    print(f"Quantum VQE time           : {PROFILE['Quantum_VQE']:.2f} s")
    print("--------------------------------------------------")

    total = (
        PROFILE["DFT_reference"]
        + PROFILE["DFT_energy"]
        + PROFILE["DFT_fock"]
        + hybrid_overhead
        + PROFILE["Quantum_VQE"]
    )

    print(f"Total measured time        : {total:.2f} s")
    print("\nMemory snapshots (RSS, MB):")
    for k, v in MEMORY_SNAPSHOTS.items():
        print(f"{k:30s}: {v:8.2f} MB")
    print("==================================================")

# Run main when executed as a script
if __name__ == "__main__":
    _main()
