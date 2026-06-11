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
from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint
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
        driver.run_pyscf()
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
            _tic("DFT")
            e_tot = driver._calc.energy_tot(dm=rho)
            _toc("DFT")
            # e) Fock operator (alpha/beta)
            _tic("Embedding")
            fock_a, fock_b = driver._expand_mo_object(driver._calc.get_fock(dm=rho), array_dimension=3)



            # f) Update active-space references
            self.active_space.active_density = active_mo_density
            self.active_space.reference_inactive_energy = e_tot - e_nuc
            self.active_space.reference_inactive_fock = basis_trafo.transform_electronic_integrals(
                ElectronicIntegrals.from_raw_integrals(fock_a, h1_b=fock_b)
            )

            # h) reduce the full MO-basis problem to the active subsystem problem (this yields the smaller problem that the active-space solver will handle).
            as_problem = self.active_space.transform(problem)
            _toc("Embedding")
            # i) solve the active-space problem (e.g. VQE via GroundStateEigensolver)result should contain electronic density for the active subspace and
            _tic("Active Space")
            result = self.solver.solve(as_problem)
            _toc("Active Space")
            # j) add the newly-computed active density to history and apply damping/mixing
            new_damped_density = self.damp_active_density(
                active_density_history + [result.electronic_density]
            )
            active_density_history.append(new_damped_density)

            # k) convergence check: we track total energies produced by the active-space solver (result.total_energies[0]) and stop when energy change < threshold.
            e_prev = e_next
            e_next = result.total_energies[0]
            embedding_energies.append(e_next)
            print(f"[Iteration {n_iter}] Embedding total energy = {e_next:.8f} Ha")
            if n_iter > 1:
                converged = np.abs(e_prev - e_next) < self.threshold
                if converged:
                    print(f"[Converged] change {np.abs(e_prev - e_next):.3e} < {self.threshold}")
                    break

        # Return active-space result (last iteration) and the energy history
        return result, embedding_energies

    # === damping helper ===
        # === damping helper (first-code style, alpha = 0.2) ===
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
    #omega = 7.25  # range-separation parameter used in PySCF's with_range_coulomb

    # Active-space specification (tune for your system)
    active_num_spatial_orbitals = 6
    active_num_electrons = 6
    # Open-shell HF initial state (alpha > beta)
    num_alpha = 3  # 2 unpaired electrons → alpha=2
    num_beta = 3

    # VQE optimizer max iterations (passed to L-BFGS-B)
    #vqe_maxiter = 50

    # Geometry options (kept compact for documentation). Select the one to use.
    #geometry = "H 0 0 0; H 0 0 0.74"
    #geometry = "O 0.0000 0.0000 0.1197; H 0.0000 0.7616 -0.4786; H 0.0000 -0.7616 -0.4786"  # H2O
    # geometry = "O 0.0 0.0 0.0; O 0.0 0.0 1.21" # O2 (triplet, ROHF)
    # geometry1 = "C 0.0 0.0 0.0; O 0.0 0.0 1.1692; O 0.0 0.0 -1.1692"  # CO2
    # geometry2 = (
    #     "C    0.0000    1.3983    0.0000; "
    #     "C    1.2110    0.6992    0.0000; "
    #     "C    1.2110   -0.6992    0.0000; "
    #     "C    0.0000   -1.3983    0.0000; "
    #     "C   -1.2110   -0.6992    0.0000; "
    #     "C   -1.2110    0.6992    0.0000; "
    #     "H    0.0000    2.4848    0.0000; "
    #     "H    2.1519    1.2424    0.0000; "
    #     "H    2.1519   -1.2424    0.0000; "
    #     "H    0.0000   -2.4848    0.0000; "
    #     "H   -2.1519   -1.2424    0.0000; "
    #     "H   -2.1519    1.2424    0.0000"
    # )  # Benzene
    geometry = (
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

    # Friendly filename token used in saved plots
    geometry_name = "Pyridine-QDFT"

    # ----------------------------------------------------------------
    # (B) PySCF driver setup
    # ----------------------------------------------------------------
    # The driver wraps PySCF DFT and produces integrals / occupations used by Qiskit Nature.
    # xc_functional uses a short-range LDA + a long-range HF component parameterized by omega.
    driver = PySCFDriver(
        atom=geometry,
        basis="6-31g*",
        #method=MethodType.UKS,
        method=MethodType.RKS,      # <--- Change here
        xc_functional="lrc_wpbe",
        #xcf_library="xcfun",
        spin=0
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
    optimizer = L_BFGS_B(maxiter=500, maxfun=1000, ftol=1e-6)
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
    ansatz = UCCSD(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_particles=(num_alpha, num_beta),
        qubit_mapper=mapper,
        initial_state=initial_state,
        reps=1,
    )
    #initial_point = 0.05 * (2 * np.random.rand(ansatz.num_parameters) - 1)
    #initial_point = np.zeros(ansatz.num_parameters)
    #np.random.seed(42)
    #sigma = 0.001
    #initial_point = sigma * np.random.randn(ansatz.num_parameters)
    mp2_init = MP2InitialPoint()
    mp2_init.compute(ansatz=ansatz, problem=driver.run())
    initial_point = mp2_init.to_numpy_array()

# ---------- 🔹 CHANGE 3: Use AerEstimator (compatible with your setup) ----------
    from qiskit_aer.primitives import Estimator as AerEstimator
    #estimator = AerEstimator()
    from qiskit.primitives import Estimator
    estimator = Estimator() 
    # ---------- 🔹 VQE callback (original normal behavior) ----------
    vqe_energy_history = []

    def vqe_callback(eval_count, parameters, mean, std):
        """Simple callback to record VQE energy evaluations."""
        vqe_energy_history.append(mean)
        print(f"  [VQE callback] Eval {eval_count:3d}  Energy = {mean:.8f} Ha")

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
        result, embedding_energies = dft_solver.solve(driver)
    
        # ✅ Add this to print the final formatted result like in COBYLA
        print("\n=== DFT Embedding + VQE Results ===")
        print(result)

    except Exception as e:
        print(f"[Warning] VQE encountered issue ({e}); continuing with last results.")
        result, embedding_energies = None, []

    # 1. Calculate the 'Classical Embedding Overhead' (Total loop time minus VQE)
    # Your "Embedding" label already isolates the embedding logic from the active space solver.
    classical_embedding_overhead = PROFILE["Embedding"]

    # 2. Calculate the 'Total measured time' by adding the distinct phases
    total = (
        PROFILE["DFT"]             # Recurring energy evaluation
        + PROFILE["Embedding"]       # The embedding logic loop
        + PROFILE["Active Space"]    # The VQE / Solver time
    )

    print("\n================ PROFILING SUMMARY ================")
    print(f"DFT (Energy Evaluator) time   : {PROFILE['DFT']:.2f} s")
    print(f"Classical Embedding Overhead  : {classical_embedding_overhead:.2f} s")
    print(f"Active Space (Quantum) time   : {PROFILE['Active Space']:.2f} s")
    print("--------------------------------------------------")
    print(f"Total measured time           : {total:.2f} s")
    
    print("\nMemory snapshots (RSS, MB):")
    for k, v in MEMORY_SNAPSHOTS.items():
        print(f"{k:30s}: {v:8.2f} MB")
    print("==================================================")

# Run main when executed as a script
if __name__ == "__main__":
    _main()


