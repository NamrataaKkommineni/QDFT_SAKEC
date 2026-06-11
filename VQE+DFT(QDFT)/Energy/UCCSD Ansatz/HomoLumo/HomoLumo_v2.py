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

    def solve(self, driver: PySCFDriver, omega: float,mol_name: str = "Molecule", active_space_str: str = "CAS"):

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

            # h) reduce the full MO-basis problem to the active subsystem problem (this yields the smaller problem that the active-space solver will handle).
            as_problem = self.active_space.transform(problem)

            # i) solve the active-space problem (e.g. VQE via GroundStateEigensolver)`result` should contain electronic density for the active subspace and
            _toc("Embedding_loop_total")
            result = self.solver.solve(as_problem)

            # j) add the newly-computed active density to history and apply damping/mixing
            active_density_history.append(
                self.damp_active_density(
                    active_density_history + [result.electronic_density]
                )
            )

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

        # ================================================================
        #  >>> ADDED: HOMO-LUMO Analysis Patch
        # ================================================================
        try:
            print("\n=== HOMO–LUMO Analysis ===")
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

            # Dynamic filenames create karna
            homo_filename = f"{mol_name}_HOMO_{active_space_str}.cube"
            lumo_filename = f"{mol_name}_LUMO_{active_space_str}.cube"

            print("\nAttempting to write HOMO/LUMO cube files...")
            nocc = driver._calc.mol.nelec[0]
            cubegen.orbital(driver._calc.mol, homo_filename, driver._calc.mo_coeff[:, nocc-1])
            cubegen.orbital(driver._calc.mol, lumo_filename, driver._calc.mo_coeff[:, nocc])
            print(f"[Saved] {homo_filename}, {lumo_filename}")
        except Exception as e:
            print(f"[Cube] Could not write cube files: {e}")

        # Return active-space result (last iteration) and the energy history
        return result, embedding_energies

    # === damping helper ===
    @staticmethod
    def damp_active_density(density_history):
        """
        Simple linear damping (mixing) of active-space densities.
        Uses a fixed alpha = 0.2 → keeps 80% new density, 20% previous.
        """

        # If there's no previous density, just return the newest one
        if len(density_history) < 2:
            return density_history[-1]

        prev_density = density_history[-2]
        new_density = density_history[-1]

        # Damping factor (fraction of previous density retained) #Adaptive damping in place
        alpha_init = 0.2
        alpha_min = 0.05
        alpha = max(alpha_min, alpha_init / np.sqrt(len(density_history)))

        #alpha = 0.2  # smaller → faster updates; larger → smoother convergence

        try:
            # Linear mixing: ρ_damped = (1 - α)*prev + α*new
            damped_density = (1.0 - alpha) * prev_density + alpha * new_density
        except Exception as e:
            print(f"[Warning] Damping failed ({e}); using new density directly.")
            damped_density = new_density

        return damped_density


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
    omega = 5.0  # range-separation parameter used in PySCF's with_range_coulomb

    # Active-space specification (tune for your system)
    active_num_spatial_orbitals = 6
    active_num_electrons = 8

    # VQE optimizer max iterations (passed to L-BFGS-B)
    #vqe_maxiter = 50

    # Geometry options (kept compact for documentation). Select the one to use.
    geometry = "O 0.0000 0.0000 0.1197; H 0.0000 0.7616 -0.4786; H 0.0000 -0.7616 -0.4786"  # H2O
    geometry1 = "C 0.0 0.0 0.0; O 0.0 0.0 1.1692; O 0.0 0.0 -1.1692"  # CO2
    '''
    geometry2 = (
        "C    0.0000    1.3983    0.0000; "
        "C    1.2110    0.6992    0.0000; "
        "C    1.2110   -0.6992    0.0000; "
        "C    0.0000   -1.3983    0.0000; "
        "C   -1.2110   -0.6992    0.0000; "
        "C   -1.2110    0.6992    0.0000; "
        "H    0.0000    2.4848    0.0000; "
        "H    2.1519    1.2424    0.0000; "
        "H    2.1519   -1.2424    0.0000; "
        "H    0.0000   -2.4848    0.0000; "
        "H   -2.1519   -1.2424    0.0000; "
        "H   -2.1519    1.2424    0.0000"
    )  # Benzene
    '''
    geometry3 = (
        "N    0.0000    0.0000    1.4209; "
        "C    0.0000    0.0000   -1.3855; "
        "C    0.0000    1.1421    0.7220; "
        "C    0.0000   -1.1421    0.7220; "
        "C    0.0000    1.1986   -0.6730; "
        "C    0.0000   -1.1986   -0.6730; "
        "H    0.0000    0.0000   -2.4724; "
        "H    0.0000    2.0598    1.3085; "
        "H    0.0000   -2.0598    1.3085; "
        "H    0.0000    2.1578   -1.1826; "
        "H    0.0000    -2.1578   -1.1826"
    )  # Pyridine
    propane = (
    "C   0.000000   0.000000   0.586866; "
    "C  -0.000000   1.277427  -0.259609; "
    "C   0.000000  -1.277427  -0.259609; "
    "H   0.877797   0.000000   1.247354; "
    "H  -0.877797  -0.000000   1.247354; "
    "H  -0.000000   2.176115   0.368129; "
    "H   0.000000  -2.176115   0.368129; "
    "H   0.884702   1.322572  -0.907060; "
    "H  -0.884702   1.322572  -0.907060; "
    "H  -0.884702  -1.322572  -0.907060; "
    "H   0.884702  -1.322572  -0.907060"
    )  # Propane

    pentane = (
    "C   0.000000   0.000000   0.316190; "
    "C  -0.000000   1.284088  -0.523531; "
    "C   0.000000  -1.284088  -0.523531; "
    "C  -0.000000   2.561094   0.323584; "
    "C   0.000000  -2.561094   0.323584; "
    "H   0.878693   0.000000   0.978795; "
    "H  -0.878693  -0.000000   0.978795; "
    "H   0.878229   1.283604  -1.184936; "
    "H  -0.878229   1.283604  -1.184936; "
    "H  -0.878229  -1.283604  -1.184936; "
    "H   0.878229  -1.283604  -1.184936; "
    "H  -0.000000   3.459308  -0.304697; "
    "H   0.000000  -3.459308  -0.304697; "
    "H  -0.884699   2.607266   0.970763; "
    "H   0.884699   2.607266   0.970763; "
    "H   0.884699  -2.607266   0.970763; "
    "H  -0.884699  -2.607266   0.970763"
    )

    heptane = (
    "C   0.000000   0.000000   0.494887; "
    "C  -0.000000   1.283886  -0.345225; "
    "C   0.000000  -1.283886  -0.345225; "
    "C  -0.000000   2.567996   0.494821; "
    "C   0.000000  -2.567996   0.494821; "
    "C  -0.000000   3.844938  -0.352381; "
    "C   0.000000  -3.844938  -0.352381; "
    "H  -0.878564  -0.000000   1.157356; "
    "H   0.878564   0.000000   1.157356; "
    "H  -0.878545   1.285024  -1.007836; "
    "H   0.878545   1.285024  -1.007836; "
    "H   0.878545  -1.285024  -1.007836; "
    "H  -0.878545  -1.285024  -1.007836; "
    "H   0.878117   2.567474   1.156358; "
    "H  -0.878117   2.567474   1.156358; "
    "H  -0.878117  -2.567474   1.156358; "
    "H   0.878117  -2.567474   1.156358; "
    "H  -0.000000   4.743228   0.275817; "
    "H   0.884682   3.891077  -0.999631; "
    "H  -0.884682   3.891077  -0.999631; "
    "H   0.000000  -4.743228   0.275817; "
    "H  -0.884682  -3.891077  -0.999631; "
    "H   0.884682  -3.891077  -0.999631"
    )

    nonane = (
    "C   0.000000   0.000000   0.361928; "
    "C  -0.000000   1.283829  -0.478659; "
    "C   0.000000  -1.283829  -0.478659; "
    "C  -0.000000   2.567719   0.361516; "
    "C   0.000000  -2.567719   0.361516; "
    "C  -0.000000   3.851848  -0.478565; "
    "C   0.000000  -3.851848  -0.478565; "
    "C  -0.000000   5.128779   0.368731; "
    "C   0.000000  -5.128779   0.368731; "
    "H   0.878531   0.000000   1.024319; "
    "H  -0.878531  -0.000000   1.024319; "
    "H   0.878565   1.283794  -1.141071; "
    "H  -0.878565   1.283794  -1.141071; "
    "H  -0.878565  -1.283794  -1.141071; "
    "H   0.878565  -1.283794  -1.141071; "
    "H   0.878575   2.568865   1.024085; "
    "H  -0.878575   2.568865   1.024085; "
    "H  -0.878575  -2.568865   1.024085; "
    "H   0.878575  -2.568865   1.024085; "
    "H  -0.878153   3.851350  -1.140039; "
    "H   0.878153   3.851350  -1.140039; "
    "H   0.878153  -3.851350  -1.140039; "
    "H  -0.878153  -3.851350  -1.140039; "
    "H  -0.000000   6.027110  -0.259415; "
    "H  -0.884687   5.174845   1.015980; "
    "H   0.884687   5.174845   1.015980; "
    "H   0.000000  -6.027110  -0.259415; "
    "H   0.884687  -5.174845   1.015980; "
    "H  -0.884687  -5.174845   1.015980  "
    )

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
        xc_functional=f"ldaerf + lr_hf({omega})",
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
    optimizer = L_BFGS_B(maxiter=50, maxfun=100, ftol=1e-6)
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
        reps=1,
    )
    #initial_point = 0.05 * (2 * np.random.rand(ansatz.num_parameters) - 1)
    #initial_point = np.zeros(ansatz.num_parameters)
    np.random.seed(42)
    sigma = 0.001
    initial_point = sigma * np.random.randn(ansatz.num_parameters)

# ---------- 🔹 CHANGE 3: Use AerEstimator (compatible with your setup) ----------
    from qiskit_aer.primitives import Estimator as AerEstimator
    #estimator = AerEstimator()
    from qiskit.primitives import Estimator
    estimator = Estimator()     # exact expectation values, no Aer limitations
    # ---------- 🔹 VQE callback (original normal behavior) ----------

    # ---------- 🔹 VQE callback with quantum timing ----------
    _vqe_last_time = {"t": None}
    vqe_energy_history = []

    def vqe_callback(eval_count, parameters, mean, std):
        """
        VQE callback that records energy history and accumulates
        wall-clock quantum optimization time.
        """
        now = time.perf_counter()

        if _vqe_last_time["t"] is not None:
            PROFILE["Quantum_VQE"] += now - _vqe_last_time["t"]

        _vqe_last_time["t"] = now

        vqe_energy_history.append(mean)
        print(f"  [VQE] Eval {eval_count:3d}  Energy = {mean:.8f} Ha")

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
        mol_name = "Anthracene"  # Change this based on the geometry used (e.g., "H2O", "CO2", "Benzene", etc.)
        active_space_str = f"CAS_{active_num_electrons}e_{active_num_spatial_orbitals}o"

        result, embedding_energies = dft_solver.solve(driver, omega, mol_name=mol_name, active_space_str=active_space_str)

        # ✅ Add this to print the final formatted result like in COBYLA
        print("\n=== DFT Embedding + VQE Results ===")
        print(result)

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