import time
import psutil
from collections import defaultdict

# ================================================================
#  PROFILING GLOBAL STATE
# ================================================================
PROCESS = psutil.Process()
PROFILE = defaultdict(float)
MEMORY_SNAPSHOTS = {}
_vqe_last_time = {"t": None}
_vqe_iter_start = {"t": None}   # <-- CHANGE 1

def _mem_mb():
    return PROCESS.memory_info().rss / 1024**2

def _tic(label):
    PROFILE[f"{label}_start"] = time.perf_counter()
    MEMORY_SNAPSHOTS[f"{label}_start_MB"] = _mem_mb()

def _toc(label):
    PROFILE[label] += time.perf_counter() - PROFILE[f"{label}_start"]
    MEMORY_SNAPSHOTS[f"{label}_end_MB"] = _mem_mb()


_tic("Library_Imports")

# ================================================================
#  IMPORTS & GLOBAL SETTINGS
# ================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from qiskit.primitives import Estimator, EstimatorResult

settings.tensor_unwrapping = False
settings.use_pauli_sum_op = False
settings.use_symmetry_reduced_integrals = True

_toc("Library_Imports")


# ================================================================
#  TimingEstimator
# ================================================================
class TimingEstimator(Estimator):
    """
    Drop-in replacement for qiskit.primitives.Estimator that timestamps
    every .run() invocation so we can split energy vs gradient time
    per VQE iteration in the callback.
    """
    def __init__(self):
        super().__init__()
        self.call_log = []

    def run(self, circuits, observables, parameter_values=None, **run_options):
        global _vqe_last_time, _vqe_iter_start   # <-- CHANGE 2a

        if _vqe_last_time["t"] is None:
            _vqe_last_time["t"] = time.perf_counter()

        # <-- CHANGE 2b: stamp iteration start on first estimator call of each iteration
        if _vqe_iter_start["t"] is None:
            _vqe_iter_start["t"] = time.perf_counter()

        t0 = time.perf_counter()
        result = super().run(circuits, observables, parameter_values, **run_options)
        t1 = time.perf_counter()

        n_circuits = len(circuits) if hasattr(circuits, "__len__") else 1
        self.call_log.append({
            "n_circuits": n_circuits,
            "duration_s": t1 - t0,
            "type": "unknown",
        })
        return result


# ================================================================
#  CLASS: DFTEmbeddingSolver
# ================================================================
class DFTEmbeddingSolver:

    def __init__(
        self,
        active_space: ActiveSpaceTransformer,
        solver: GroundStateEigensolver,
        *,
        max_iter: int = 6,
        threshold: float = 1e-6,
    ) -> None:
        self.active_space = active_space
        self.solver = solver
        self.max_iter = max_iter
        self.threshold = threshold

    def solve(self, driver: PySCFDriver, omega: float,vqe_iter_log: list = None):
        _tic("DFT_reference")
        driver.run_pyscf()
        _toc("DFT_reference")

        E_DFT_full = driver._calc.e_tot
        print("\n=== Classical DFT Reference Energy (PySCF) ===")
        print(f"Total DFT Energy: {E_DFT_full:.8f} Ha")
        print("==============================================\n")

        _tic("Problem_Setup_and_Integrals")
        (mo_coeff, mo_coeff_b) = driver._expand_mo_object(driver._calc.mo_coeff, array_dimension=3)
        basis_trafo = BasisTransformer(
            ElectronicBasis.AO,
            ElectronicBasis.MO,
            ElectronicIntegrals.from_raw_integrals(mo_coeff, h1_b=mo_coeff_b),
        )

        with driver._mol.with_range_coulomb(omega=omega):
            problem = driver.to_problem(basis=ElectronicBasis.MO, include_dipole=False)
            total_mo_density = ElectronicDensity.from_orbital_occupation(
                problem.orbital_occupations, problem.orbital_occupations_b, include_rdm2=False
            )
            problem.properties.electronic_density = total_mo_density

        self.active_space.prepare_active_space(
            problem.num_particles,
            problem.num_spatial_orbitals,
            occupation_alpha=problem.orbital_occupations,
            occupation_beta=problem.orbital_occupations_b,
        )

        active_density_history = [self.active_space.active_basis.transform_electronic_integrals(total_mo_density)]
        inactive_ao_density = basis_trafo.invert().transform_electronic_integrals(
            total_mo_density - self.active_space.active_basis.invert().transform_electronic_integrals(active_density_history[-1])
        )
        _toc("Problem_Setup_and_Integrals")

        e_nuc = problem.hamiltonian.nuclear_repulsion_energy
        e_tot = driver._calc.e_tot
        e_next = float("NaN")
        e_prev = float("NaN")
        converged = False
        n_iter = 0
        embedding_energies = []

        while n_iter < self.max_iter:
            n_iter += 1
            _tic("Embedding_loop_total")

            active_mo_density = self.active_space.active_basis.invert().transform_electronic_integrals(active_density_history[-1])
            active_ao_density = basis_trafo.invert().transform_electronic_integrals(active_mo_density)
            total_ao_density = inactive_ao_density + active_ao_density

            if basis_trafo.coefficients.beta.is_empty():
                rho = np.asarray(total_ao_density.trace_spin()["+-"])
            else:
                rho = np.asarray([total_ao_density.alpha["+-"], total_ao_density.beta["+-"]])

            _tic("DFT_energy")
            e_tot = driver._calc.energy_tot(dm=rho)
            _toc("DFT_energy")

            _tic("DFT_fock")
            (fock_a, fock_b) = driver._expand_mo_object(driver._calc.get_fock(dm=rho), array_dimension=3)
            _toc("DFT_fock")

            self.active_space.active_density = active_mo_density
            self.active_space.reference_inactive_energy = e_tot - e_nuc
            self.active_space.reference_inactive_fock = basis_trafo.transform_electronic_integrals(
                ElectronicIntegrals.from_raw_integrals(fock_a, h1_b=fock_b)
            )

            as_problem = self.active_space.transform(problem)
            _toc("Embedding_loop_total")

            global _vqe_last_time, _vqe_iter_start
            _vqe_last_time["t"] = None
            _vqe_iter_start["t"]= None

            _tic("VQE_initialization")
            prev_len = len(vqe_iter_log) if vqe_iter_log is not None else 0
            _tic("Solver_total")
            result = self.solver.solve(as_problem)
            _toc("Solver_total")
            if vqe_iter_log is not None:
                rows_this_iter = vqe_iter_log[prev_len:]
                sum_energy = sum(r["t_energy_s"] for r in rows_this_iter)
                sum_grad   = sum(r["t_gradient_s"] for r in rows_this_iter)
                sum_optim  = sum(r["t_optimizer_s"] for r in rows_this_iter)
                print(f"\n[Embedding Iter {n_iter}] VQE Column Sums:")
                print(f"  EnergyEval : {sum_energy*1000:.1f} ms")
                print(f"  GradEval   : {sum_grad*1000:.1f} ms")
                print(f"  Optimizer  : {sum_optim*1000:.1f} ms")
            active_density_history.append(self.damp_active_density(active_density_history + [result.electronic_density]))

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

    @staticmethod
    def damp_active_density(density_history):
        if len(density_history) < 2:
            return density_history[-1]
        prev_density = density_history[-2]
        new_density = density_history[-1]

        alpha_init = 0.2
        alpha_min = 0.05
        alpha = max(alpha_min, alpha_init / np.sqrt(len(density_history)))

        try:
            damped_density = (1.0 - alpha) * prev_density + alpha * new_density
        except Exception as e:
            print(f"[Warning] Damping failed ({e}); using new density directly.")
            damped_density = new_density

        return damped_density


# ================================================================
#  MAIN
# ================================================================
def _main():
    omega = 5.0
    active_num_spatial_orbitals = 6
    active_num_electrons = 2

    geometry  = "O 0.0000 0.0000 0.1197; H 0.0000 0.7616 -0.4786; H 0.0000 -0.7616 -0.4786"
    geometry1 = "C 0.0 0.0 0.0; O 0.0 0.0 1.1692; O 0.0 0.0 -1.1692"

    geometry3 = (
        "N    0.0000    0.0000    1.4209; C    0.0000    0.0000   -1.3855; "
        "C    0.0000    1.1421    0.7220; C    0.0000   -1.1421    0.7220; "
        "C    0.0000    1.1986   -0.6730; C    0.0000   -1.1986   -0.6730; "
        "H    0.0000    0.0000   -2.4724; H    0.0000    2.0598    1.3085; "
        "H    0.0000   -2.0598    1.3085; H    0.0000    2.1578   -1.1826; "
        "H    0.0000   -2.1578   -1.1826"
    )

    propane = (
        "C   0.000000   0.000000   0.586866; C  -0.000000   1.277427  -0.259609; "
        "C   0.000000  -1.277427  -0.259609; H   0.877797   0.000000   1.247354; "
        "H  -0.877797  -0.000000   1.247354; H  -0.000000   2.176115   0.368129; "
        "H   0.000000  -2.176115   0.368129; H   0.884702   1.322572  -0.907060; "
        "H  -0.884702   1.322572  -0.907060; H  -0.884702  -1.322572  -0.907060; "
        "H   0.884702  -1.322572  -0.907060"
    )

    pentane = (
        "C   0.000000   0.000000   0.316190; C  -0.000000   1.284088  -0.523531; "
        "C   0.000000  -1.284088  -0.523531; C  -0.000000   2.561094   0.323584; "
        "C   0.000000  -2.561094   0.323584; H   0.878693   0.000000   0.978795; "
        "H  -0.878693  -0.000000   0.978795; H   0.878229   1.283604  -1.184936; "
        "H  -0.878229   1.283604  -1.184936; H  -0.878229  -1.283604  -1.184936; "
        "H   0.878229  -1.283604  -1.184936; H  -0.000000   3.459308  -0.304697; "
        "H   0.000000  -3.459308  -0.304697; H  -0.884699   2.607266   0.970763; "
        "H   0.884699   2.607266   0.970763; H   0.884699  -2.607266   0.970763; "
        "H  -0.884699  -2.607266   0.970763"
    )
    geometry_C14H10 = (
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
    )
    geometry_scan = geometry_C14H10

    _tic("Driver_init")
    driver = PySCFDriver(
        atom=geometry_scan,
        basis="6-31g*",
        method=MethodType.RKS,
        xc_functional=f"ldaerf + lr_hf({omega})",
        xcf_library="xcfun",
    )
    _toc("Driver_init")

    active_space = ActiveSpaceTransformer(
        num_spatial_orbitals=active_num_spatial_orbitals,
        num_electrons=active_num_electrons,
    )

    mapper = TaperedQubitMapper(ParityMapper())

    n = active_num_electrons
    num_particles = (n // 2, n - (n // 2))

    optimizer = L_BFGS_B(maxiter=500, maxfun=1000, ftol=1e-6)

    _tic("Ansatz_build")
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
        reps=1,
    )
    _toc("Ansatz_build")

    np.random.seed(42)
    sigma = 0.001
    initial_point = sigma * np.random.randn(ansatz.num_parameters)

    estimator = TimingEstimator()

    vqe_iter_log = []
    vqe_energy_history = []
    _last_call_index = {"i": 0}

    # <-- CHANGE 3: full replacement of vqe_callback
    def vqe_callback(eval_count, parameters, mean, std):
        global _vqe_last_time, _vqe_iter_start
        callback_start = time.perf_counter()
        if eval_count == 1:
            _toc("VQE_initialization")   # >>> CHANGE A: stop init timer
        # independent wall-clock total: from first estimator call to now
        t_iter_wallclock = (
            callback_start - _vqe_iter_start["t"]
            if _vqe_iter_start["t"] is not None
            else 0.0
        )

        if _vqe_last_time["t"] is not None:
            total_since_last = callback_start - _vqe_last_time["t"]
            PROFILE["Quantum_VQE"] += total_since_last
        else:
            total_since_last = 0.0

        calls_this_iter = estimator.call_log[_last_call_index["i"]:]
        _last_call_index["i"] = len(estimator.call_log)

        energy_calls   = calls_this_iter[:1]
        gradient_calls = calls_this_iter[1:]

        t_energy   = sum(c["duration_s"] for c in energy_calls)
        t_gradient = sum(c["duration_s"] for c in gradient_calls)

        for c in energy_calls:   c["type"] = "energy"
        for c in gradient_calls: c["type"] = "gradient"

        t_optimizer = max(0.0, total_since_last - t_energy - t_gradient)

        cb_t0 = time.perf_counter()
        vqe_energy_history.append(mean)
        t_callback = time.perf_counter() - cb_t0

        PROFILE["Quantum_VQE"] += t_callback

        # include callback body in wall-clock total
        t_iter_wallclock += t_callback

        iter_entry = {
            "eval_count":    eval_count,
            "energy":        mean,
            "t_energy_s":    t_energy,
            "t_gradient_s":  t_gradient,
            "t_optimizer_s": t_optimizer,
            "t_callback_s":  t_callback,
            "t_total_s":     t_iter_wallclock,  # wall-clock, not additive
        }

        vqe_iter_log.append(iter_entry)

        print(
            f"  [VQE iter {eval_count:3d}] "
            f"E={mean:.8f} Ha | "
            f"energy={t_energy*1000:6.1f}ms | "
            f"grad={t_gradient*1000:6.1f}ms | "
            f"optim={t_optimizer*1000:6.1f}ms"
        )

        _vqe_last_time["t"] = time.perf_counter()
        _vqe_iter_start["t"] = None  # reset for next iteration

    vqe_solver = VQE(
        ansatz=ansatz,
        optimizer=optimizer,
        estimator=estimator,
        callback=vqe_callback,
        initial_point=initial_point,
    )

    ground_state_solver = GroundStateEigensolver(mapper, vqe_solver)
    dft_solver = DFTEmbeddingSolver(active_space, ground_state_solver)

    try:
        result, embedding_energies = dft_solver.solve(driver, omega,vqe_iter_log=vqe_iter_log)
        print("\n=== DFT Embedding + VQE Results ===")
        print(result)
    except Exception as e:
        print(f"[Warning] VQE encountered issue ({e}); continuing with last results.")
        result, embedding_energies = None, []

    print("\n" + "=" * 90)
    print("VQE PER-ITERATION BREAKDOWN")
    print("=" * 90)
    header = (
        f"{'Iter':>4} | {'Energy (Ha)':>14} | "
        f"{'EnergyEval':>11} | {'GradEval':>10} | "
        f"{'Optimizer':>10} | {'Callback':>9} | {'IterTotal':>10}"
    )
    print(header)
    print("-" * 90)

    sum_energy = sum_grad = sum_optim = sum_cb = sum_total = 0.0

    for row in vqe_iter_log:
        print(
            f"{row['eval_count']:>4d} | "
            f"{row['energy']:>14.8f} | "
            f"{row['t_energy_s']*1000:>10.1f}ms | "
            f"{row['t_gradient_s']*1000:>9.1f}ms | "
            f"{row['t_optimizer_s']*1000:>9.1f}ms | "
            f"{row['t_callback_s']*1000:>8.1f}ms | "
            f"{row['t_total_s']*1000:>9.1f}ms"
        )
        sum_energy  += row["t_energy_s"]
        sum_grad    += row["t_gradient_s"]
        sum_optim   += row["t_optimizer_s"]
        sum_cb      += row["t_callback_s"]
        sum_total   += row["t_total_s"]

    print("-" * 90)
    print(
        f"{'SUM':>4} | {'':>14} | "
        f"{sum_energy*1000:>10.1f}ms | "
        f"{sum_grad*1000:>9.1f}ms | "
        f"{sum_optim*1000:>9.1f}ms | "
        f"{sum_cb*1000:>8.1f}ms | "
        f"{sum_total*1000:>9.1f}ms"
    )
    print("=" * 90)

    hybrid_overhead = (
        PROFILE["Embedding_loop_total"]
        - PROFILE["DFT_energy"]
        - PROFILE["DFT_fock"]
    )

    print("\n================ PROFILING SUMMARY ================")
    print(f"Library Imports              : {PROFILE['Library_Imports']:.2f} s")
    print(f"Driver Initialization        : {PROFILE['Driver_init']:.2f} s")
    print(f"Problem Setup & Integrals    : {PROFILE['Problem_Setup_and_Integrals']:.2f} s")

    print("--------------------------------------------------")
    print(f"Ansatz build                 : {PROFILE['Ansatz_build']:.2f} s")
    print(f"VQE Initialization (map+taper+prep)    : {PROFILE['VQE_initialization']:.2f} s")

    print(f"Solver total (Pre/Post-processing + Quantum VQE) : {PROFILE['Solver_total']:.2f} s")
    classical_qiskit_overhead = max(0.0, PROFILE["Solver_total"] - PROFILE["Quantum_VQE"])
    print(f"Classical Qiskit Overhead    : {classical_qiskit_overhead:.2f} s")
    print("--------------------------------------------------")

    print("--------------------------------------------------")
    print(f"Total Energy Eval time(Circuit Simulation Time)       : {sum_energy:.2f} s")
    print(f"Total Gradient Eval time     : {sum_grad:.2f} s")
    print(f"Total Optimizer time         : {sum_optim:.2f} s")
    print(f"Total Callback time          : {sum_cb:.2f} s")
    print(f"Total VQE Iteration time     : {sum_total:.2f} s")
    print("--------------------------------------------------")


    print("--------------------------------------------------")
    print(f"DFT reference (SCF) time     : {PROFILE['DFT_reference']:.2f} s")
    print(f"DFT energy evaluations time  : {PROFILE['DFT_energy']:.2f} s")
    print(f"DFT Fock build time          : {PROFILE['DFT_fock']:.2f} s")
    print(f"Hybrid embedding overhead    : {hybrid_overhead:.2f} s")
    print(f"Quantum VQE Execution        : {PROFILE['Quantum_VQE']:.2f} s")

    core_algorithmic_time = (
        PROFILE["DFT_reference"]
        + PROFILE["DFT_energy"]
        + PROFILE["DFT_fock"]
        + hybrid_overhead
        + PROFILE["Quantum_VQE"]
    )
    print(f"Total measured (QDFT only)   : {core_algorithmic_time:.2f} s")
    print("==================================================")

    print("\nMemory snapshots (RSS, MB):")
    for k, v in MEMORY_SNAPSHOTS.items():
        print(f"{k:30s}: {v:8.2f} MB")
    print("==================================================")

if __name__ == "__main__":
    _main()
