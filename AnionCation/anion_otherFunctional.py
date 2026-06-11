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
#from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.optimizers import L_BFGS_B
#from qiskit_algorithms.optimizers import COBYLA
from pyscf import gto, dft
from pyscf.dft import xcfun
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
# from qiskit_nature.second_q.circuit.library import UCC
# from qiskit.circuit.library import EvolvedOperatorAnsatz
#from qiskit_algorithms.optimizers import SPSA
#from qiskit_nature.second_q.circuit.library import HartreeFock, PUCCSD

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

    def solve(self, driver: PySCFDriver,mol_name: str = "Molecule", active_space_str: str = "CAS"):

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

            # c) Total AO density = inactive + active
            total_ao_density = inactive_ao_density + active_ao_density

            # # === 🔹 UKS / open-shell handling ===
            # if basis_trafo.coefficients.beta.is_empty():
            #     # Closed-shell / spin-restricted (alpha = beta)
            #     rho = np.asarray(total_ao_density.trace_spin()["+-"])
            # else:
            #     # Open-shell / spin-unrestricted
            #     rho = np.asarray([total_ao_density.alpha["+-"], total_ao_density.beta["+-"]])
            # === 🔹 ROKS / Open-Shell Handling ===
            if driver._calc.mol.spin == 0:
                # Closed-shell: Pass 2D total density
                rho = np.asarray(total_ao_density.trace_spin()["+-"])
            else:
                # ROKS Open-Shell: Force extraction of 3D (alpha, beta) density matrices
                rho = np.asarray([total_ao_density.alpha["+-"], total_ao_density.beta["+-"]])

            # d) Evaluate DFT energy at new density
            _tic("DFT_energy")
            e_tot = driver._calc.energy_tot(dm=rho)
            _toc("DFT_energy")

            # e) Fock operator (alpha/beta)
            _tic("DFT_fock")
            fock_a, fock_b = driver._expand_mo_object(driver._calc.get_fock(dm=rho), array_dimension=3)
            _toc("DFT_fock")

             # f) Update active-space references
            self.active_space.active_density = active_mo_density
            self.active_space.reference_inactive_energy = e_tot - e_nuc
            self.active_space.reference_inactive_fock = basis_trafo.transform_electronic_integrals(
                ElectronicIntegrals.from_raw_integrals(fock_a, h1_b=fock_b)
            )

            # h) reduce the full MO-basis problem to the active subsystem problem (this yields the smaller problem that the active-space solver will handle$
            as_problem = self.active_space.transform(problem)

            # i) solve the active-space problem (e.g. VQE via GroundStateEigensolver)`result` should contain electronic density for the active subspace and
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
            # nocc = driver._calc.mol.nelec[0]
            # homo_coeff = driver._calc.mo_coeff[:, nocc-1]
            # lumo_coeff = driver._calc.mo_coeff[:, nocc]
            nocc_alpha = driver._calc.mol.nelec[0] # Alpha count defines the ROKS HOMO (SOMO)
            
            # In ROKS, mo_coeff is a standard 2D array, so we can slice it directly
            homo_coeff = driver._calc.mo_coeff[:, nocc_alpha-1]
            lumo_coeff = driver._calc.mo_coeff[:, nocc_alpha]

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
    def damp_active_density(density_history, base_alpha=0.4, diis_start=10, diis_space=4):
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
    # Assuming your neutral system was 2e, 6o. The cation becomes 1e, 6o. Anion becomes 3e, 6o. Adjust the number of electrons and spin accordingly for your target charge state.
    active_num_electrons = 7
    num_alpha = 4
    num_beta = 3

    # Pass the exact spin distribution as a tuple
    num_particles = (num_alpha, num_beta)

    # VQE optimizer max iterations (passed to L-BFGS-B)
    #vqe_maxiter = 50

    # Geometry options (kept compact for documentation). Select the one to use.
    propy = (
    "C    -2.433983    0.537495   -0.037310; "
    "C    -1.091503    0.943452   -0.009332; "
    "C    -0.063338   -0.030201    0.030293; "
    "C    -0.430992   -1.378606    0.047467; "
    "C    -1.768625   -1.781752    0.018783; "
    "C    -2.770996   -0.814294   -0.028013; "
    "H    -3.215692    1.294324   -0.070640; "
    "H     0.344617   -2.137593    0.079969; "
    "H    -2.019323   -2.838597    0.030447; "
    "H    -3.817852   -1.106160   -0.051082; "
    "N    -0.768374    2.307310   -0.085737; "
    "H    -1.536207    2.923602    0.153538; "
    "H     0.074459    2.575879    0.407313; "
    "C     1.382078    0.433346    0.039028; "
    "H     1.532228    1.124075   -0.805385; "
    "H     1.565845    1.041604    0.942612; "
    "C     2.458262   -0.657433   -0.020535; "
    "H     2.307269   -1.271282   -0.918159; "
    "H     2.350252   -1.332509    0.838581; "
    "C     3.875996   -0.075634   -0.032726; "
    "H     4.070272    0.518311    0.869115; "
    "H     4.630306   -0.869263   -0.076736; "
    "H     4.029369    0.578772   -0.899641"
    ) # 2-propylaniline_optimized (B3LYP/6-31g*)

    isopropy = (
    "C    -2.106860    0.738622   -0.016537; "
    "C    -0.724496    0.948512   -0.145828; "
    "C     0.159366   -0.160044   -0.119024; "
    "C    -0.395719   -1.435217    0.038160; "
    "C    -1.770080   -1.641108    0.172695; "
    "C    -2.627225   -0.542097    0.145905; "
    "H    -2.774128    1.598393   -0.039605; "
    "H     0.263487   -2.297581    0.054022; "
    "H    -2.161430   -2.647084    0.294137; "
    "H    -3.701324   -0.676876    0.244520; "
    "N    -0.241278    2.264697   -0.238691; "
    "H     0.625771    2.368764   -0.751624; "
    "H    -0.933130    2.927741   -0.568459; "
    "C     1.666659    0.057925   -0.224081; "
    "H     1.842784    0.816616   -1.005103; "
    "C     2.455075   -1.185287   -0.663431; "
    "H     2.054568   -1.613941   -1.588774; "
    "H     3.504003   -0.919635   -0.837246; "
    "H     2.442540   -1.965998    0.106076; "
    "C     2.237783    0.616683    1.097987; "
    "H     3.305034    0.849240    0.994368; "
    "H     1.714149    1.526309    1.407294; "
    "H     2.128290   -0.124017    1.898918"
    ) # 2-isopropylaniline_optimized (B3LYP/6-31g*)

    # Friendly filename token used in saved plots
    geometry_scan = propy
    # ----------------------------------------------------------------
    # (B) PySCF driver setup
    # ----------------------------------------------------------------
    # --- VERSION-PROOF INITIALIZATION ---
    # 1. Create the PySCF Molecule object directly
    # mol = gto.M(
    #     atom=geometry_scan,
    #     basis="6-31g*",
    #     spin=1,
    #     charge=-1
    # )

    driver = PySCFDriver(
        atom=geometry_scan,  # Change this to the geometry you want to use (e.g., geometry_C6H6, geometry_C10H8, etc.)
        basis="6-31g*",
        method=MethodType.ROKS,
        xc_functional="lrc_wpbe",
        #xcf_library="xcfun",
        spin=1,
        charge=-1
    )

    # ----------------------------------------------------------------
    # (C) Active-space transformer and VQE setup
    # ----------------------------------------------------------------
    active_space = ActiveSpaceTransformer(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_electrons=num_particles,
    )

    # Qubit mapper used for the active-space qubit Hamiltonian (parity mapping +
    # tapering to exploit symmetries). This is the mapper passed to GroundStateEigensolver.
    mapper = TaperedQubitMapper(ParityMapper())

    # Build Hartree-Fock initial state and q-UCCSD ansatz sized for the active problem.
    #n = active_num_electrons
    #num_particles = (n // 2, n - (n // 2))

    # ---------- 🔹 CHANGE 1: Use L_BFGS_B with internal evaluation limits ----------
    from qiskit_algorithms.optimizers import L_BFGS_B
    optimizer = L_BFGS_B(maxiter=2000, maxfun=10000, ftol=1e-6)
    #optimizer = SPSA(maxiter=500)
# (keeps COBYLA option commented for reference)
# optimizer = COBYLA(maxiter=200, tol=1e-4)

# ---------- 🔹 CHANGE 2: Add a small random initial point to avoid flat gradient ----------
    #initial_point = 0.05 * (2 * np.random.rand(ansatz.num_parameters) - 1)

# Build Hartree-Fock initial state (HF) and UCCSD ansatz
    initial_state = HartreeFock(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_particles=(num_alpha, num_beta),
        qubit_mapper=mapper,
    )
    ansatz = UCCSD(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_particles=(num_alpha, num_beta),
        qubit_mapper=mapper,
        initial_state=initial_state,
        reps=1,
    )
    #initial_point = 0.05 * (2 * np.random.rand(ansatz.num_parameters) - 1)
    #initial_point = np.zeros(ansatz.num_parameters)
    np.random.seed(42)
    sigma = 0.001
    initial_point = sigma * np.random.randn(ansatz.num_parameters)

# ---------- 🔹 CHANGE 3: Use AerEstimator (compatible with your setup) ----------
    # from qiskit_aer.primitives import Estimator as AerEstimator
    #estimator = AerEstimator()
    from qiskit.primitives import Estimator
    estimator = Estimator()     # exact expectation values, no Aer limitations

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
        mol_name = "propylaniline_cation"  # Change this based on the geometry used (e.g., "H2O", "CO2", "Benzene", etc.)
        active_space_str = f"CAS_{active_num_electrons}e_{active_num_spatial_orbitals}o"

        result, embedding_energies = dft_solver.solve(driver, mol_name=mol_name, active_space_str=active_space_str)

        # ✅ Add this to print the final formatted result like in COBYLA
        print("\n=== DFT Embedding + VQE Results ===")
        print(result)

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