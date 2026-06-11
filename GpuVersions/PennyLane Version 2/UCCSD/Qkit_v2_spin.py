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
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, BasisTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.second_q.properties import ElectronicDensity
from qiskit_nature.settings import settings
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_algorithms.gradients import ReverseEstimatorGradient
import time
import psutil
from collections import defaultdict

# Keep original (important) settings from your script
settings.tensor_unwrapping = False
settings.use_pauli_sum_op = False
settings.use_symmetry_reduced_integrals = True

PROCESS = psutil.Process()
PROFILE = defaultdict(float)
MEMORY_SNAPSHOTS = {}

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

    def solve(self, driver: PySCFDriver):

        # ---- Step 1: reference DFT run (classical DFT energy baseline) ----
        _tic("DFT_reference")
        driver.run_pyscf()
        _toc("DFT_reference")
        
        E_DFT_full = driver._calc.e_tot
        print("\n=== Classical DFT Reference Energy (PySCF) ===")
        print(f"Total DFT Energy: {E_DFT_full:.8f} Ha")
        print("==============================================\n")

        _tic("Embedding_loop_total")
        
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

        # Build total MO density (spin-adapted).
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

        inactive_ao_density = basis_trafo.invert().transform_electronic_integrals(
            total_mo_density
            - self.active_space.active_basis.invert().transform_electronic_integrals(
                active_density_history[-1]
            )
        )

        # ---- bookkeeping variables for iteration and outputs ----
        e_nuc = problem.hamiltonian.nuclear_repulsion_energy  
        e_tot = driver._calc.e_tot  
        e_next = float("NaN")
        e_prev = float("NaN")
        converged = False
        n_iter = 0
        embedding_energies = []

        # ---- Step 6: iterative embedding loop ----
        while n_iter < self.max_iter:
            n_iter += 1

            active_mo_density = (
                self.active_space.active_basis.invert().transform_electronic_integrals(
                    active_density_history[-1]
                )
            )

            active_ao_density = basis_trafo.invert().transform_electronic_integrals(
                active_mo_density
            )

            total_ao_density = inactive_ao_density + active_ao_density

            if basis_trafo.coefficients.beta.is_empty():
                rho = np.asarray(total_ao_density.trace_spin()["+-"])
            else:
                rho = np.asarray([total_ao_density.alpha["+-"], total_ao_density.beta["+-"]])

            _tic("DFT_energy")
            e_tot = driver._calc.energy_tot(dm=rho)
            _toc("DFT_energy")

            _tic("DFT_fock")
            fock_a, fock_b = driver._expand_mo_object(driver._calc.get_fock(dm=rho), array_dimension=3)
            _toc("DFT_fock")

            self.active_space.active_density = active_mo_density
            self.active_space.reference_inactive_energy = e_tot - e_nuc
            self.active_space.reference_inactive_fock = basis_trafo.transform_electronic_integrals(
                ElectronicIntegrals.from_raw_integrals(fock_a, h1_b=fock_b)
            )

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
            beta = 0.10110
            penalized_h_op = h_op + (beta * s2_op)

            # 4. Override the problem's Hamiltonian generation before passing to the solver
            as_problem.hamiltonian.second_q_op = lambda: penalized_h_op

            _tic("Quantum_VQE")
            result = self.solver.solve(as_problem)
            _toc("Quantum_VQE")

            new_damped_density = self.damp_active_density(
                active_density_history + [result.electronic_density]
            )
            active_density_history.append(new_damped_density)

            e_prev = e_next
            e_next = result.total_energies[0]
            embedding_energies.append(e_next)
            
            print(f"[Iteration {n_iter}] Embedding total energy = {e_next:.8f} Ha")
            if n_iter > 1:
                converged = np.abs(e_prev - e_next) < self.threshold
                if converged:
                    print(f"[Converged] change {np.abs(e_prev - e_next):.3e} < {self.threshold}")
                    break

        _toc("Embedding_loop_total")
        return result, embedding_energies

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

# ================================================================
#  MAIN: simulation setup, VQE initialization, run embedding
# ================================================================
def _main():
    active_num_spatial_orbitals = 6    
    active_num_electrons = 6
    num_alpha = 3
    num_beta = 3

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

    driver = PySCFDriver(
        atom=tetracene,
        basis="6-31g*",
        method=MethodType.RKS,
        xc_functional="B3LYP",
        xcf_library="xcfun",
        spin=0
    )

    active_space = ActiveSpaceTransformer(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_electrons=active_num_electrons
    )

    # MATCHING THE GPU SCRIPT: Pure JordanWigner
    mapper = JordanWignerMapper()

    optimizer = L_BFGS_B(maxiter=2000, maxfun=10000, ftol=1e-6)


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
        generalized=False,
        preserve_spin=True,
        reps=1,
    )

    # np.random.seed(42)
    # initial_point = 0.001 * np.random.randn(ansatz.num_parameters)
    initial_point = np.zeros(ansatz.num_parameters)

    from qiskit.primitives import Estimator
    estimator = Estimator()     
    adjoint_gradient = ReverseEstimatorGradient()

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

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        gradient=adjoint_gradient,
        initial_point=initial_point,
        callback=vqe_callback
    )

    ground_state_solver = GroundStateEigensolver(mapper, vqe)
    dft_solver = DFTEmbeddingSolver(active_space, ground_state_solver)

    try:
        result, embedding_energies = dft_solver.solve(driver)

        print("\n=== DFT Embedding + VQE Results ===")
        print(result)

    except Exception as e:
        print(f"[Warning] VQE encountered issue ({e}); continuing with last results.")
        result, embedding_energies = None, []


   # ============================================================
    #  FINAL PROFILING REPORT
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

