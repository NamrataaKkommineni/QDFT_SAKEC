# Developed By Namrataa Kkommineni under AICTE Industry Fellowship Program SH-2025
#Inspiration derived from Rossmanek 2020 quantum embedding paper https://arxiv.org/pdf/2009.01872

# ================================================================
#  IMPORTS & GLOBAL SETTINGS
# ================================================================
import numpy as np
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
from qiskit_nature.second_q.circuit.library import UCC
from qiskit.circuit.library import EvolvedOperatorAnsatz

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
        _tic("DFT_reference")
        driver.run_pyscf()
        _toc("DFT_reference")
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

            # === 🔹 UKS / open-shell handling ===
            if basis_trafo.coefficients.beta.is_empty():
                # Closed-shell / spin-restricted (alpha = beta)
                rho = np.asarray(total_ao_density.trace_spin()["+-"])
            else:
                # Open-shell / spin-unrestricted
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

            # h) reduce the full MO-basis problem to the active subsystem problem (this yields the smaller problem that the active-space solver will handle).
            as_problem = self.active_space.transform(problem)
            _toc("Embedding_loop_total")
            # i) solve the active-space problem (e.g. VQE via GroundStateEigensolver)result should contain electronic density for the active subspace and
            result = self.solver.solve(as_problem)

            # ---------- 🔹 ADDED: Print parameters for this iteration ----------
            if hasattr(result, 'raw_result') and hasattr(result.raw_result, 'optimal_point'):
                opt_params = result.raw_result.optimal_point
                print(f"  --> [Iteration {n_iter}] VQE Optimal Parameters:\n      {np.round(opt_params, 6)}")
            # -------------------------------------------------------------------

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
        # === damping helper (first-code style, alpha = 0.2) ===
    @staticmethod
    def damp_active_density(density_history, base_alpha=0.75, diis_start=9, diis_space=3):
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
    #omega = 5.40  # range-separation parameter used in PySCF's with_range_coulomb

    # Active-space specification (tune for your system)
    active_num_spatial_orbitals = 6
    active_num_electrons = 6
    # Open-shell HF initial state (alpha > beta)
    num_alpha = 3  # 2 unpaired electrons → alpha=2
    num_beta = 3
    num_particles = (num_alpha, num_beta)

    pyridine = (
    "N    0.000000    0.000000    1.421093; "
    "C    0.000000    0.000000   -1.385925; "
    "C    0.000000    1.142537    0.722644; "
    "C    0.000000   -1.142537    0.722644; "
    "C    0.000000    1.198737   -0.672743; "
    "C    0.000000   -1.198737   -0.672743; "
    "H    0.000000    0.000000   -2.472808; "
    "H    0.000000    2.060198    1.309358; "
    "H    0.000000   -2.060198    1.309358; "
    "H    0.000000    2.158029   -1.182151; "
    "H    0.000000   -2.158029   -1.182151"
    )  # Pyridine (Optimized B3LYP/6-31g*)
    porphyrin = (
    "N    7.153087    0.161552   -0.000000; "
    "N    4.884909   -2.115422    0.000000; "
    "N    5.194124    2.046507   -0.000000; "
    "N    2.991135   -0.164804    0.000000; "
    "C    7.860027   -1.016181   -0.000000; "
    "C    6.065932   -2.817299    0.000000; "
    "C    8.008693    1.242826   -0.000000; "
    "C    3.807699   -2.975804    0.000000; "
    "C    7.373126   -2.325197    0.000000; "
    "C    9.251204   -0.650671   -0.000000; "
    "C    5.706532   -4.209992    0.000000; "
    "C    9.341120    0.718650   -0.000000; "
    "C    4.337637   -4.305920    0.000000; "
    "C    7.567311    2.565667   -0.000000; "
    "C    2.482858   -2.540346    0.000000; "
    "C    6.216878    2.945524   -0.000000; "
    "C    2.096789   -1.191690    0.000000; "
    "C    4.050286    2.778746    0.000000; "
    "C    2.254154    0.975896    0.000000; "
    "C    2.736355    2.291600    0.000000; "
    "C    5.713392    4.314347   -0.000000; "
    "C    0.725408   -0.693932    0.000000; "
    "C    4.359299    4.210421   -0.000000; "
    "C    0.823665    0.660564    0.000000; "
    "H    8.142288   -3.091540    0.000000; "
    "H    6.144851    0.357836   -0.000000; "
    "H    4.684439   -1.108120    0.000000; "
    "H   10.063659   -1.365410   -0.000000; "
    "H    6.425164   -5.018987    0.000000; "
    "H   10.239587    1.321281   -0.000000; "
    "H    3.738609   -5.206810    0.000000; "
    "H    8.324723    3.343501   -0.000000; "
    "H    1.708281   -3.301088    0.000000; "
    "H    1.965477    3.059579    0.000000; "
    "H    6.319916    5.211773   -0.000000; "
    "H   -0.169642   -1.303930    0.000000; "
    "H    3.625989    5.007912   -0.000000; "
    "H    0.022875    1.390266    0.000000"
    )  # Porphyrin-Adjacent (Optimized B3LYP/6-31g*)

    # Friendly filename token used in saved plots
    geometry_name = "IITB-lrc"

    # ----------------------------------------------------------------
    # (B) PySCF driver setup
    # ----------------------------------------------------------------
    # The driver wraps PySCF DFT and produces integrals / occupations used by Qiskit Nature.
    # xc_functional uses a short-range LDA + a long-range HF component parameterized by omega.
    driver = PySCFDriver(
        atom=pyridine,
        basis="6-31g*",
        method=MethodType.RKS,
        xc_functional="lrc_wpbe",
        #xcf_library="xcfun",
    )

    # ----------------------------------------------------------------
    # (C) Active-space transformer and VQE setup
    # ----------------------------------------------------------------
    active_space = ActiveSpaceTransformer(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_electrons=active_num_electrons
    )

    # Qubit mapper used for the active-space qubit Hamiltonian (parity mapping +
    # tapering to exploit symmetries). This is the mapper passed to GroundStateEigensolver.
    mapper = TaperedQubitMapper(ParityMapper())

    # Build Hartree-Fock initial state and q-UCCSD ansatz sized for the active problem.
    #n = active_num_electrons
    #num_particles = (n // 2, n - (n // 2))
    # FOR OXYGEN (TRIPLET): Both active electrons are unpaired (alpha spin)
    # We bypass the default (n//2) logic to explicitly define 2 alpha, 0 beta
    #num_particles = (n, 0)

    # ---------- 🔹 CHANGE 1: Use L_BFGS_B with internal evaluation limits ----------
    from qiskit_algorithms.optimizers import L_BFGS_B
    optimizer = L_BFGS_B(maxiter=2000, maxfun=10000, ftol=1e-6)
    # (keeps COBYLA option commented for reference)
    #optimizer = COBYLA(maxiter=2000, tol=1e-5)

# ---------- 🔹 CHANGE 2: Add a small random initial point to avoid flat gradient ----------
    #initial_point = 0.05 * (2 * np.random.rand(ansatz.num_parameters) - 1)

# Build Hartree-Fock initial state (HF) and UCCSD ansatz
    initial_state = HartreeFock(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_particles=(num_alpha, num_beta),
        qubit_mapper=mapper,
    )
    # ansatz = UCCSD(
    #     num_spatial_orbitals=active_num_spatial_orbitals,
    #     num_particles=(num_alpha, num_beta),
    #     qubit_mapper=mapper,
    #     initial_state=initial_state,
    #     reps=1,
    # )
    # #initial_point = 0.05 * (2 * np.random.rand(ansatz.num_parameters) - 1)
    # #initial_point = np.zeros(ansatz.num_parameters)
    # np.random.seed(42)
    # sigma = 0.001
    # initial_point = sigma * np.random.randn(ansatz.num_parameters)

# ---------- 🔹 CHANGE 3: Use AerEstimator (compatible with your setup) ----------
    #from qiskit_aer.primitives import Estimator as AerEstimator
    #estimator = AerEstimator()
    from qiskit.primitives import Estimator
    estimator = Estimator()

# IITB pART 1
    first_list = []
    energy_list = []
    optimal_list = []
    pruned_excitation_list = []
    var_form = UCC(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=initial_state,
        excitations='d'
        )
    excitation_list = var_form._get_excitation_list()
    #print(excitation_list)
    fer_excitation_op = var_form.excitation_ops()  # getting the second_q operator for excitations in UCC
    excitation_list_pauli = list()
    for ex in fer_excitation_op:
        excitation_list_pauli.append(mapper.map(ex))
    print(f"\n[IITB Pruning] Total 'd' excitations BEFORE pruning: {len(excitation_list_pauli)}")
    print(excitation_list)
    _tic("Ansatz_generation")

    base_problem = driver.run()
    as_problem = active_space.transform(base_problem)
    second_q_op = as_problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(second_q_op)
    # Energy sorting of T2 excitations
    pruned_excitation_list_pauli = list()
    excitation_list_pruned=list()
    difference_E=list()
    for i in range(len(excitation_list_pauli)):

        var_form1 = EvolvedOperatorAnsatz(excitation_list_pauli[i], initial_state=initial_state)
        initial_job = estimator.run([var_form1], [qubit_op], parameter_values=[[0.0]])
        initial_energy = initial_job.result().values[0]
        vqe1 = VQE(estimator, var_form1, optimizer=optimizer, initial_point=[0.0])
        vqe_result = vqe1.compute_minimum_eigenvalue(qubit_op)
        E1 = np.real(vqe_result.eigenvalue)
        op_pt = vqe_result.optimal_point
        first_list.append(op_pt[0])
        #Pruned list sort karna hai, optimal list ko uss hisab se sort karna hai
        if abs(initial_energy - E1) > 1e-5:
            energy_list.append(E1)
            difference_E.append(abs(initial_energy-E1))
            optimal_list.append(op_pt[0])
            excitation_list_pruned.append(excitation_list[i])
            pruned_excitation_list.append(excitation_list[i])
            pruned_excitation_list_pauli.append(excitation_list_pauli[i])
        print('state and average value is', excitation_list[i],E1)

    difference_E1=difference_E.copy()
    difference_E1.sort(reverse=True)

    print('Sorted Differences Energy',difference_E1)
    dob_excitation_list = list()
    dob_energy_list = list()
    dob_params = list()
    final_dob_list = list()
    for iii in range(len(difference_E1)):
        eee = difference_E1[iii]
        for jjj in range(len(difference_E)):
            eee1 = difference_E[jjj]
            if eee1==eee:
                dob_excitation_list.append(excitation_list_pruned[jjj])
                dob_energy_list.append(energy_list[jjj])
                dob_params.append(optimal_list[jjj])
                final_dob_list.append(pruned_excitation_list_pauli[jjj])

                difference_E[jjj]=0.0

    print('Sorted Doubles List',dob_excitation_list)
    final_excitations_list = []
    final_parameters_list = []
    final_pauli_list = list()
    Sh_list= [(( 6 , 0 ),( 9 , 2 )),(( 6 , 1 ),( 9 , 2 )),(( 6 , 0 ),( 10 , 2 )),(( 6 , 1 ),( 10 , 2 )),(( 6 , 0 ),( 11 , 2 )),(( 6 , 1 ),( 11 , 2 )),(( 7 , 0 ),( 9 , 2 )),(( 7 , 1 ),( 9 , 2 )),(( 7 , 0 ),( 10 , 2 )),(( 7 , 1 ),( 10 , 2 )),(( 7 , 0 ),( 11 , 2 )),(( 7 , 1 ),( 11 , 2 )),(( 8 , 0 ),( 9 , 2 )),(( 8 , 1 ),( 9 , 2 )),(( 8 , 0 ),( 10 , 2 )),(( 8 , 1 ),( 10 , 2 )),(( 8 , 0 ),( 11 , 2 )),(( 8 , 1 ),( 11 , 2 )),(( 0 , 6 ),( 3 , 8 )),(( 0 , 7 ),( 3 , 8 )),(( 0 , 6 ),( 4 , 8 )),(( 0 , 7 ),( 4 , 8 )),(( 0 , 6 ),( 5 , 8 )),(( 0 , 7 ),( 5 , 8 )),(( 1 , 6 ),( 3 , 8 )),(( 1 , 7 ),( 3 , 8 )),(( 1 , 6 ),( 4 , 8 )),(( 1 , 7 ),( 4 , 8 )),(( 1 , 6 ),( 5 , 8 )),(( 1 , 7 ),( 5 , 8 )),(( 2 , 6 ),( 3 , 8 )),(( 2 , 7 ),( 3 , 8 )),(( 2 , 6 ),( 4 , 8 )),(( 2 , 7 ),( 4 , 8 )),(( 2 , 6 ),( 5 , 8 )),(( 2 , 7 ),( 5 , 8 ))]

    Sp_list= [(( 6 , 3 ),( 9 , 4 )),(( 6 , 3 ),( 9 , 5 )),(( 6 , 3 ),( 10 , 4 )),(( 6 , 3 ),( 10 , 5 )),(( 6 , 3 ),( 11 , 4 )),(( 6 , 3 ),( 11 , 5 )),(( 7 , 3 ),( 9 , 4 )),(( 7 , 3 ),( 9 , 5 )),(( 7 , 3 ),( 10 , 4 )),(( 7 , 3 ),( 10 , 5 )),(( 7 , 3 ),( 11 , 4 )),(( 7 , 3 ),( 11 , 5 )),(( 8 , 3 ),( 9 , 4 )),(( 8 , 3 ),( 9 , 5 )),(( 8 , 3 ),( 10 , 4 )),(( 8 , 3 ),( 10 , 5 )),(( 8 , 3 ),( 11 , 4 )),(( 8 , 3 ),( 11 , 5 )),(( 0 , 9 ),( 3 , 10 )),(( 0 , 9 ),( 3 , 11 )),(( 0 , 9 ),( 4 , 10 )),(( 0 , 9 ),( 4 , 11 )),(( 0 , 9 ),( 5 , 10 )),(( 0 , 9 ),( 5 , 11 )),(( 1 , 9 ),( 3 , 10 )),(( 1 , 9 ),( 3 , 11 )),(( 1 , 9 ),( 4 , 10 )),(( 1 , 9 ),( 4 , 11 )),(( 1 , 9 ),( 5 , 10 )),(( 1 , 9 ),( 5 , 11 )),(( 2 , 9 ),( 3 , 10 )),(( 2 , 9 ),( 3 , 11 )),(( 2 , 9 ),( 4 , 10 )),(( 2 , 9 ),( 4 , 11 )),(( 2 , 9 ),( 5 , 10 )),(( 2 , 9 ),( 5 , 11 ))]

    def S_list_h(num_spatial_orbitals, num_particles):
        ex = Sh_list
        return ex

    ucc_sh = UCC(num_spatial_orbitals=active_num_spatial_orbitals,num_particles= num_particles, excitations = S_list_h, qubit_mapper = mapper, initial_state = initial_state)
    sh_fer_ops = ucc_sh.excitation_ops()
    sh_pauli = list()
    for fer in sh_fer_ops:
        sh_pauli.append(mapper.map(fer))

    def S_list_p(num_spatial_orbitals, num_particles):
        ex = Sp_list
        return ex

    ucc_sp = UCC(active_num_spatial_orbitals, num_particles, excitations = S_list_p, qubit_mapper = mapper, initial_state = initial_state)
    sp_fer_ops = ucc_sp.excitation_ops()
    sp_pauli = list()
    for fer in sp_fer_ops:
        sp_pauli.append(mapper.map(fer))



    for index in range(len(dob_excitation_list)):
        double = dob_excitation_list[index]
        ref_E = dob_energy_list[index]
        i, j, a, b = double[0][0], double[0][1], double[1][0], double[1][1]
        final_excitations_list.append(double)
        final_parameters_list.append(dob_params[index])
        final_pauli_list.append(final_dob_list[index])
        dummy_pauli = list()
        t2_sh_list = []
        for kkk in range(len(Sh_list)):
            sh = Sh_list[kkk]
            p, q, c, d = sh[0][0], sh[0][1], sh[1][0], sh[1][1]
            if i == d or j == d:
                        #array_1, array_2 = np.array([i, j]), np.array([a, b])
                if (p != i) and (p != j) and (q != i) and (q != j) and (c != a) and (c != b):
                    print(sh)
                        #t2_sh_list.append(sh)
                    dummy_pauli = list()
                    dummy_pauli.append(final_dob_list[index])
                    dummy_pauli.append(sh_pauli[kkk])
                    ucc_custom_2 = EvolvedOperatorAnsatz(dummy_pauli, initial_state = initial_state)
                    init_par = [dob_params[index]]+[0.01]
                    vqe1 = VQE(estimator, ucc_custom_2, optimizer=optimizer, initial_point=init_par)
                    vqe_result = vqe1.compute_minimum_eigenvalue(qubit_op)
                    E1 = np.real(vqe_result.eigenvalue)
                    #print(E1)
                    op_pt = vqe_result.optimal_point

                    if abs(E1-ref_E) > 1e-5:
                        final_excitations_list.append(sh)
                        final_parameters_list.append(op_pt[-1])
                        t2_sh_list.append(sh)
                        final_pauli_list.append(sh_pauli[kkk])
        for kkk in range(len(Sp_list)):
            sp = Sp_list[kkk]
            p, q, c, d = sp[0][0], sp[0][1], sp[1][0], sp[1][1]
            if a == q or b == q:
                        #array_1, array_2 = np.array([i, j]), np.array([a, b])
                if (a != c) and (a != d) and (b != c) and (b != d) and (p != i) and (p != j):
                    print(sp)
                        #t2_sh_list.append(sp)
                    dummy_pauli = list()
                    dummy_pauli.append(final_dob_list[index])
                    dummy_pauli.append(sp_pauli[kkk])
                    ucc_custom_2 = EvolvedOperatorAnsatz(dummy_pauli, initial_state = initial_state)
                    init_par = [dob_params[index]]+[0.01]
                    vqe1 = VQE(estimator, ucc_custom_2, optimizer=optimizer, initial_point=init_par)
                    vqe_result = vqe1.compute_minimum_eigenvalue(qubit_op)
                    E1 = np.real(vqe_result.eigenvalue)
                    #print(E1)
                    op_pt = vqe_result.optimal_point

                    if abs(E1-ref_E) > 1e-5:
                        final_excitations_list.append(sp)
                        final_parameters_list.append(op_pt[-1])
                        t2_sh_list.append(sp)
                        final_pauli_list.append(sp_pauli[kkk])

        print('Doubles',double,'çorresponding S',t2_sh_list)

    #exit()
    var_form_s = UCC(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=initial_state,
        excitations='s'
        )
    excitation_list_s = var_form_s._get_excitation_list()
    fer_excitation_op_s = var_form_s.excitation_ops()
    # ---------- 🔹 FIX: Append to the new TRP lists ----------
    for ex in fer_excitation_op_s:
        final_pauli_list.append(mapper.map(ex))

    for ex in excitation_list_s:
        final_excitations_list.append(ex)
    print(excitation_list_s)
    ansatz = EvolvedOperatorAnsatz(final_pauli_list, initial_state=initial_state)
    padded_initial_point = final_parameters_list + [0.0] * len(fer_excitation_op_s)
    _toc("Ansatz_generation")
    print(final_excitations_list,'Length',len(final_excitations_list))
    print('Length of pauli',len(final_pauli_list))
    # ---------------------------------------------------------

#IITB
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
        initial_point=padded_initial_point,
    )

    # Wrap VQE into the Ground State Solver
    ground_state_solver = GroundStateEigensolver(mapper, vqe_solver)

    # Instantiate the DFT embedding solver
    dft_solver = DFTEmbeddingSolver(active_space, ground_state_solver)

    # ================================================================
    #  (D) Run embedding normally
    # ================================================================
    try:
        mol_name = "IITB_Pyridine_lrc"  # Change this based on the geometry used (e.g., "H2O", "CO2", "Benzene", etc.)
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
    #  FINAL PROFILING REPORT (COMPREHENSIVE DECOMPOSITION)
    # ============================================================
    hybrid_overhead = (
        PROFILE["Embedding_loop_total"]
        - PROFILE["DFT_energy"]
        - PROFILE["DFT_fock"]
    )

    print("\n================ PROFILING SUMMARY ================")
    print(f"DFT reference (SCF) time     : {PROFILE['DFT_reference']:.2f} s")
    print(f"Ansatz Pruning & Init Time   : {PROFILE['Ansatz_generation']:.2f} s") # Added
    print(f"DFT energy evaluations time : {PROFILE['DFT_energy']:.2f} s")
    print(f"DFT Fock build time         : {PROFILE['DFT_fock']:.2f} s")
    print(f"Hybrid embedding overhead  : {hybrid_overhead:.2f} s")
    print(f"Quantum VQE time           : {PROFILE['Quantum_VQE']:.2f} s")
    print("--------------------------------------------------")

    total = (
        PROFILE["DFT_reference"]
        + PROFILE["Ansatz_generation"]   # Added to total timeline
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
