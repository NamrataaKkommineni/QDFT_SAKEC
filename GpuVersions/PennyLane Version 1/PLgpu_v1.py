# Developed By Namrataa Kkommineni under AICTE Industry Fellowship Program SH-2025
# Inspiration derived from Rossmanek 2020 quantum embedding paper https://arxiv.org/pdf/2009.01872

# ================================================================
#  IMPORTS & GLOBAL SETTINGS
# ================================================================
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp # PennyLane requires its own autograd-wrapped numpy
import pennylane.qchem as qchem # <-- Add this for native chemistry templates
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
import gpu4pyscf.dft as gpu_dft
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

            # a) expand active density
            active_mo_density = (
                self.active_space.active_basis.invert().transform_electronic_integrals(
                    active_density_history[-1]
                )
            )

            # b) transform the active MO density to AO basis
            active_ao_density = basis_trafo.invert().transform_electronic_integrals(
                active_mo_density
            )

            # c) Total AO density = inactive + active
            total_ao_density = inactive_ao_density + active_ao_density

            # === 🔹 UKS / open-shell handling ===
            if basis_trafo.coefficients.beta.is_empty():
                rho = np.asarray(total_ao_density.trace_spin()["+-"])
            else:
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

            # h) reduce the full MO-basis problem to the active subsystem problem
            as_problem = self.active_space.transform(problem)
            _toc("Embedding")

            # i) solve the active-space problem
            _tic("Active Space")
            result = self.solver.solve(as_problem)
            _toc("Active Space")

            # j) add the newly-computed active density to history and apply damping/mixing
            new_damped_density = self.damp_active_density(
                active_density_history + [result.electronic_density]
            )
            active_density_history.append(new_damped_density)

            # k) convergence check
            e_prev = e_next
            e_next = result.total_energies[0]
            embedding_energies.append(e_next)
            print(f"[Iteration {n_iter}] Embedding total energy = {e_next:.8f} Ha")
            if n_iter > 1:
                converged = np.abs(e_prev - e_next) < self.threshold
                if converged:
                    print(f"[Converged] change {np.abs(e_prev - e_next):.3e} < {self.threshold}")
                    break

        return result, embedding_energies

    # === damping helper ===
    @staticmethod
    def damp_active_density(density_history, base_alpha=0.4, diis_start=6, diis_space=4):
        if len(density_history) < 2:
            return density_history[-1]

        prev_density = density_history[-2]
        new_density = density_history[-1]

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
                pass

        alpha = max(0.05, base_alpha / np.sqrt(len(density_history)))
        print(f"[Damping] Linear mix with alpha={alpha:.3f}")

        try:
            return (1.0 - alpha) * prev_density + alpha * new_density
        except Exception:
            return new_density

class GPUPySCFDriver(PySCFDriver):
    """Intercepts Qiskit's PySCF call to run the initial SCF on the GPU."""
    def run_pyscf(self):
        from pyscf import gto

        print(">> Building PySCF molecule...")
        atom_str = getattr(self, "atom", getattr(self, "_atom", None))
        basis_str = getattr(self, "basis", getattr(self, "_basis", None))
        chg = getattr(self, "charge", getattr(self, "_charge", 0))
        spn = getattr(self, "spin", getattr(self, "_spin", 0))

        self._mol = gto.M(atom=atom_str, basis=basis_str, charge=chg, spin=spn)
        self._mol.build()

        print(f">> Intercepting Qiskit Nature: Running gpu4pyscf on GPU...")

        # -----------------------------------------------------------
        # Using RKS for closed-shell Water (H2O)
        # -----------------------------------------------------------
        gpu_calc = gpu_dft.RKS(self._mol)

        gpu_calc.xc = self.xc_functional
        gpu_calc.grids.level = 3
        gpu_calc.run()

        print(">> Transferring GPU matrices back to CPU for Qiskit Mapper...\n")

        # This prevents Qiskit's ActiveSpaceTransformer from crashing on CuPy arrays
        self._calc = gpu_calc.to_cpu()

# ================================================================
#  MAIN: simulation setup, VQE initialization, run embedding
# ================================================================
def _main():
    # ----------------------------------------------------------------
    # (A) Simulation parameters & geometries
    # ----------------------------------------------------------------
    active_num_spatial_orbitals = 6    #-----> 2 for H2
    active_num_electrons = 2
    num_alpha = 1
    num_beta = 1

    geometry1 = (
    "H    0.000000    0.000000   -0.371394; "
    "H    0.000000    0.000000    0.371394"
    ) #H2_optimized
    geometry2 = (
        "O    0.000000    0.000000    0.118852; "
        "H    0.000000    0.762815   -0.478289; "
        "H    0.000000   -0.762815   -0.478289"
    ) #H2O_optimized
    geometry3 = (
        "O    0.000000    0.000000   -0.606726; "
        "O    0.000000    0.000000    0.606726"
    ) #O2_optimized

    geometry_C16H10 = (
    "C  0.713438  -0.000167  -0.000001;"
    "C  -0.713475  -0.000222  0.000001;"
    "C  1.429193  1.236362  -0.000004;"
    "C  -1.429241  1.236304  0.000001;"
    "C  1.429801  -1.236386  -0.000001;"
    "C  -1.429779  -1.236480  0.000004;"
    "C  0.680887  2.464304  -0.000004;"
    "C  -0.680975  2.464270  -0.000002;"
    "C  0.680950  -2.463967  0.000002;"
    "C  -0.680911  -2.464051  0.000004;"
    "C  2.833349  1.210851  -0.000006;"
    "C  -2.833394  1.210770  0.000004;"
    "C  2.833998  -1.211012  -0.000004;"
    "C  -2.833976  -1.211170  0.000006;"
    "C  3.524777  0.000101  -0.000006;"
    "C  -3.524721  -0.000038  0.000006;"
    "H  1.230624  3.402564  -0.000006;"
    "H  -1.230749  3.402510  -0.000002;"
    "H  1.230574  -3.402280  0.000002;"
    "H  -1.230475  -3.402401  0.000006;"
    "H  3.379803  2.150919  -0.000009;"
    "H  -3.379891  2.150815  0.000003;"
    "H  3.380699  -2.150952  -0.000003;"
    "H  -3.380666  -2.151118  0.000009;"
    "H  4.611457  0.000364  -0.000008;"
    "H  -4.611399  0.000194  0.000008"
    ) #pyrene

    # ----------------------------------------------------------------
    # (B) PySCF driver setup
    # ----------------------------------------------------------------
    driver = GPUPySCFDriver(          # <--- CHANGE THIS LINE
        atom=geometry_C16H10,
        basis="6-31g*",
        method=MethodType.RKS,
        xc_functional="PBE",
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

    mapper = TaperedQubitMapper(ParityMapper())

    from qiskit_algorithms.optimizers import L_BFGS_B
    #optimizer = L_BFGS_B(maxiter=500, maxfun=1000, ftol=1e-6)

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

    np.random.seed(42)
    sigma = 0.001
    initial_point = sigma * np.random.randn(ansatz.num_parameters)

    # from qiskit.primitives import Estimator
    # estimator = Estimator()
    from qiskit_aer.primitives import Estimator as AerEstimator
    estimator = AerEstimator(run_options={"device": "GPU"})

    # ---------- 🔹 VQE callback with quantum timing & convergence tracking ----------
    _vqe_last_time = {"t": None}

    vqe_history = {"evals": [], "energies": [], "step_energies": []}
    evals_per_iteration = ansatz.num_parameters + 1
    vqe_ftol = 1e-6

    def vqe_callback(eval_count, parameters, mean, std):
        now = time.perf_counter()
        if _vqe_last_time["t"] is not None:
            PROFILE["Quantum_VQE"] += now - _vqe_last_time["t"]
        _vqe_last_time["t"] = now

        if eval_count == 1:
            vqe_history["evals"].clear()
            vqe_history["energies"].clear()
            vqe_history["step_energies"].clear()
            print("  --- Starting VQE Optimization Step ---")

        vqe_history["evals"].append(eval_count)
        vqe_history["energies"].append(mean)

        if (eval_count - 1) % evals_per_iteration == 0:
            iteration = (eval_count - 1) // evals_per_iteration

            if iteration == 0:
                print(f"  [VQE] Iteration: {iteration:4d} | Energy: {mean:12.8f} Ha | ΔE: N/A")
            else:
                prev_mean = vqe_history["step_energies"][-1]
                delta_e = abs(mean - prev_mean)
                print(f"  [VQE] Iteration: {iteration:4d} | Energy: {mean:12.8f} Ha | ΔE: {delta_e:10.8f}")

                if delta_e < vqe_ftol:
                    print(f"  [VQE Converged] ΔE ({delta_e:.3e}) is below threshold ({vqe_ftol})!")

            vqe_history["step_energies"].append(mean)

    # Build the VQE solver
    # =========================================================================
    #  PARALLEL BATCHED VQE SOLVER (Error-Free SciPy Bypass)
    # =========================================================================
    class PennyLaneVQESolver:
        def __init__(self, ansatz, mapper, initial_point=None, callback=None):
            self.ansatz = ansatz
            self.mapper = mapper
            self.callback = callback

            if initial_point is not None:
                self.initial_point = pnp.array(initial_point, requires_grad=True)
            else:
                self.initial_point = pnp.zeros(ansatz.num_parameters, requires_grad=True)

            self.optimal_params = None

        def solve(self, problem):
            print("\n>> [PennyLane] Initiating Quantum-Classical Bridge...")

            qiskit_hamiltonian = problem.hamiltonian.second_q_op()
            qubit_op = self.mapper.map(qiskit_hamiltonian)

            # 1A. Translate Hamiltonian
            pl_hamiltonian = qml.from_qiskit_op(qubit_op)
            num_qubits = len(qubit_op.paulis[0])

            # We delete the symbolic translation here!
            print(f"  [*] Bridged {len(qubit_op)} Pauli terms on {num_qubits} qubits.")

            # 2. Hardware Device
            dev = qml.device("lightning.gpu", wires=num_qubits)

            # 3. The Binding Wrapper & Native QNode
            @qml.qnode(dev, diff_method="adjoint")
            def _pl_cost(numerical_ansatz_template):
                # Execute the fully numeric template
                numerical_ansatz_template()
                return qml.expval(pl_hamiltonian)

            def cost_fn(params):
                # 1. Let Qiskit handle the symbolic-to-numeric binding natively FIRST
                bound_circuit = self.ansatz.assign_parameters(params)

                # 2. Decompose and translate the PURELY NUMERICAL circuit
                decomposed_bound = bound_circuit.decompose(reps=3)
                numerical_ansatz_template = qml.from_qiskit(decomposed_bound)

                # 3. Execute the PennyLane QNode
                return _pl_cost(numerical_ansatz_template)

            # 4. SciPy Optimizer
            self.eval_count = 1
            x0 = self.initial_point.numpy()

            def scipy_cost(x):
                val = cost_fn(x)
                # Fire your custom callback for the log files
                if self.callback is not None:
                    self.callback(self.eval_count, x, float(val), 0.0)
                self.eval_count += 1
                return float(val)

            import scipy.optimize
            print("  [*] Launching Accelerated L-BFGS-B Optimization...")

            res = scipy.optimize.minimize(
                scipy_cost,
                x0,
                method="L-BFGS-B",
                # We remove jac=scipy_grad to let SciPy drive the accelerated cost_fn
                options={"maxiter": 500, "maxfun": 1000, "ftol": 1e-6}
            )

            self.optimal_params = res.x
            print(f"  [*] VQE Complete! Final Energy: {res.fun:.8f} Ha")

            # 5. Output
            from qiskit_algorithms.minimum_eigensolvers import VQE
            from qiskit_algorithms.optimizers import L_BFGS_B
            from qiskit.primitives import Estimator as ReferenceEstimator

            dummy_vqe = VQE(
                ansatz=self.ansatz,
                optimizer=L_BFGS_B(maxiter=0),
                estimator=ReferenceEstimator(),
                initial_point=self.optimal_params
            )
            return GroundStateEigensolver(self.mapper, dummy_vqe).solve(problem)

    # Wrap VQE into the PennyLane Solver
    ground_state_solver = PennyLaneVQESolver(
        ansatz,
        mapper,
        initial_point=initial_point,
        callback=vqe_callback
    )

    # Instantiate the DFT embedding solver
    dft_solver = DFTEmbeddingSolver(active_space, ground_state_solver)

    # ================================================================
    #  (D) Run embedding normally
    # ================================================================
    try:
        result, embedding_energies = dft_solver.solve(driver)

        print("\n=== DFT Embedding + VQE Results ===")
        print(result)

    except Exception as e:
        print(f"[Warning] VQE encountered issue ({e}); continuing with last results.")
        result, embedding_energies = None, []

    classical_embedding_overhead = PROFILE["Embedding"]

    total = (
        PROFILE["DFT"]
        + PROFILE["Embedding"]
        + PROFILE["Active Space"]
    )

    print("\n================ PROFILING SUMMARY ================")
    print(f"DFT (Energy Evaluator) time   : {PROFILE['DFT']:.2f} s")
    print(f"Classical Embedding Overhead  : {classical_embedding_overhead:.2f} s")
    print(f"Active Space (Quantum) time   : {PROFILE['Active Space']:.2f} s")
    if PROFILE["Quantum_VQE"] > 0:
        print(f"  --> Pure VQE Callback Time  : {PROFILE['Quantum_VQE']:.2f} s")
    print("--------------------------------------------------")
    print(f"Total measured time           : {total:.2f} s")

    print("\nMemory snapshots (RSS, MB):")
    for k, v in MEMORY_SNAPSHOTS.items():
        print(f"{k:30s}: {v:8.2f} MB")
    print("==================================================")

# Run main when executed as a script
if __name__ == "__main__":
    _main()