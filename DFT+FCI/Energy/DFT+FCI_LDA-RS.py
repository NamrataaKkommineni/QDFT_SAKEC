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

from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, BasisTransformer
from qiskit_nature.second_q.mappers import TaperedQubitMapper, ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.second_q.properties import ElectronicDensity
from qiskit_nature.settings import settings
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
        max_iter: int = 30,
        threshold: float = 1e-6,
    ) -> None:
      
        self.active_space = active_space
        self.solver = solver
        self.max_iter = max_iter
        self.threshold = threshold

    def solve(self, driver: PySCFDriver, omega: float):
       
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

        with driver._mol.with_range_coulomb(omega=omega):
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
            
            _tic("Embedding")
            # e) Fock operator (alpha/beta)
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
    def damp_active_density(density_history, base_alpha=0.4, diis_start=6, diis_space=4):
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
    omega = 5.25  # range-separation parameter used in PySCF's with_range_coulomb

    # Active-space specification (tune for your system)
    active_num_spatial_orbitals = 6
    active_num_electrons = 6
    # Open-shell HF initial state (alpha > beta)
    num_alpha = 3  # 2 unpaired electrons → alpha=2
    num_beta = 3

    # VQE optimizer max iterations (passed to L-BFGS-B)
    #vqe_maxiter = 50

    # Geometry options (kept compact for documentation). Select the one to use.
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
    
    porphyrin_adj = (
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
    ) # Porphyrin-Adjacent (Optimized B3LYP/6-31g*)

    porphyrin_opp = geometry = (		
        "N  13.516700  0.025342  0.000000;	"
        "N  9.543374  -3.840273  0.000000;	"
        "N  9.485069  3.830580  0.000000;	"
        "N  5.511621  -0.035290  0.000000;	"
        "C  15.004581  -2.099264  0.000000;	"
        "C  11.605306  -5.385172  0.000000;	"
        "C  14.972301  2.172526  0.000000;	"
        "C  7.505316  -5.416536  0.000000;	"
        "C  14.126956  -4.583370  0.000000;	"
        "C  17.577265  -1.240762  0.000000;	"
        "C  10.857054  -8.041785  0.000000;	"
        "C  17.557601  1.353170  0.000000;	"
        "C  8.293149  -8.060844  0.000000;	"
        "C  14.056966  4.643089  0.000000;	"
        "C  4.971834  -4.652802  0.000000;	"
        "C  11.523522  5.406696  0.000000;	"
        "C  4.056074  -2.182541  0.000000;	"
        "C  7.423374  5.375547  0.000000;	"
        "C  4.023944  2.089700  0.000000;	"
        "C  4.901553  4.573671  0.000000;	"
        "C  10.735471  8.051351  0.000000;	"
        "C  1.470414  -1.362896  0.000000;	"
        "C  8.171559  8.032179  0.000000;	"
        "C  1.451000  1.230824  0.000000;	"
        "H  15.568956  -6.044140  0.000000;	"
        "H  11.597907  0.011165  0.000000;	"
        "H  7.430415  -0.020602  0.000000;	"
        "H  19.205646  -2.477212  0.000000;	"
        "H  12.141719  -9.635427  0.000000;	"
        "H  19.166908  2.614328  0.000000;	"
        "H  7.032282  -9.673444  0.000000;	"
        "H  15.476687  6.125546  0.000000;	"
        "H  3.551907  -6.135052  0.000000;	"
        "H  3.459296  6.034213  0.000000;	"
        "H  11.996157  9.664072  0.000000;	"
        "H  -0.139041  -2.623880  0.000000;	"
        "H  6.887007  9.625923  0.000000;	"
        "H  -0.177306  2.467353  0.000000	"
    )	# Porphyrin-Opposite
    # ----------------------------------------------------------------
    # (B) PySCF driver setup
    # ----------------------------------------------------------------
    # The driver wraps PySCF DFT and produces integrals / occupations used by Qiskit Nature.
    # xc_functional uses a short-range LDA + a long-range HF component parameterized by omega.

    driver = PySCFDriver(
        atom=pyridine,
        basis="6-31g*",
        method=MethodType.RKS,
        xc_functional=f"ldaerf + lr_hf({omega})",
        xcf_library="xcfun",
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

    np.random.seed(42)
    
    from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver          
    
    exact_solver = NumPyMinimumEigensolver()

    ground_state_solver = GroundStateEigensolver(
        mapper,
        exact_solver
    )

    # Instantiate the DFT embedding solver
    dft_solver = DFTEmbeddingSolver(active_space, ground_state_solver)

    # ================================================================
    #  (D) Run embedding normally
    # ================================================================
    try:
        result, embedding_energies = dft_solver.solve(driver,omega)
    
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
