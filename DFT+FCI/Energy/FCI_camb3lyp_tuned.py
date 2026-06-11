# Developed By Namrataa Kkommineni under AICTE Industry Fellowship Program SH-2025
#Inspiration derived from Rossmanek 2020 quantum embedding paper https://arxiv.org/pdf/2009.01872

# ================================================================
#  IMPORTS & GLOBAL SETTINGS
# ================================================================
import numpy as np
import scipy.linalg
import matplotlib
# Use non-interactive backend for headless environments (CI / servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyscf import gto, dft
from pyscf.dft import xcfun
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
        max_iter: int = 100,
        threshold: float = 1e-6,
    ) -> None:

        self.active_space = active_space
        self.solver = solver
        self.max_iter = max_iter
        self.threshold = threshold

    def solve(self, driver: PySCFDriver,mol_name: str = "Molecule", active_space_str: str = "CAS"):
        # ---- Step 1: reference DFT run (classical DFT energy baseline) ----
        _tic("DFT_reference")
        if not hasattr(driver, "_calc"):
            driver.run_pyscf()
        else:
            print("\n>>> CONFIRMED: Using tuned functional parameters in solver loop.")
            print(f">>> Custom XC String: {driver._calc.xc}")
            driver._calc.kernel()
        #driver.run_pyscf()
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
    #omega = 5  # range-separation parameter used in PySCF's with_range_coulomb

    # Active-space specification (tune for your system)
    active_num_spatial_orbitals = 6
    active_num_electrons = 6
    # Open-shell HF initial state (alpha > beta)
    num_alpha = 3  # 2 unpaired electrons → alpha=2
    num_beta = 3

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

    triphenylene =(
   "C  3.124238  -0.429150  0.000000;"
   "C  3.857750  0.841751  0.000000;"
   "C  3.835371  -1.660274  0.000000;"
   "C  5.279498  0.841956  0.000000;"
   "C  5.302739  -1.660063  0.000000;"
   "C  6.013448  -0.428691  0.000000;"
   "C  1.711302  -0.468092  0.000000;"
   "C  3.184522  2.084621  0.000000;"
   "C  3.095695  -2.864747  0.000000;"
   "C  5.952395  2.084999  0.000000;"
   "C  6.042844  -2.864278  0.000000;"
   "C  7.426400  -0.467123  0.000000;"
   "C  1.011304  -1.661670  0.000000;"
   "C  3.867838  3.287808  0.000000;"
   "C  1.712068  -2.874784  0.000000;"
   "C  5.268795  3.288004  0.000000;"
   "C  7.426483  -2.873794  0.000000;"
   "C  8.126820  -1.660441  0.000000;"
   "H  1.144977  0.455533  0.000000;"
   "H  2.101471  2.112857  0.000000;"
   "H  3.612840  -3.816783  0.000000;"
   "H  7.035443  2.113506  0.000000;"
   "H  5.526079  -3.816520  0.000000;"
   "H  7.992418  0.456692  0.000000;"
   "H  -0.075145  -1.652780  0.000000;"
   "H  3.316605  4.224075  0.000000;"
   "H  1.176973  -3.820373  0.000000;"
   "H  5.819751  4.224434  0.000000;"
   "H  7.961906  -3.819196  0.000000;"
   "H  9.213265  -1.651184  0.000000"
    )

    # Friendly filename token used in saved plots
    geometry_name = geometry_C6H6
    # ----------------------------------------------------------------
    # (B) PySCF driver setup
    # ----------------------------------------------------------------
    # The driver wraps PySCF DFT and produces integrals / occupations used by Qiskit Nature.
    # xc_functional uses a short-range LDA + a long-range HF component parameterized by omega.
    # --- VERSION-PROOF INITIALIZATION ---
    # 1. Create the PySCF Molecule object directly
    mol = gto.M(
        atom=geometry_name,
        basis="6-31g*",
        spin=0,
        charge=0
    )
    # 2. Setup the RKS calculator with XCFUN backend
    mf = dft.RKS(mol)
    mf._numint.libxc = xcfun
    # 3. Define Parameters
    #Default
    # mu = 0.33
    # alpha = 0.19
    # beta = 0.46
    #Tuned t5
    mu = 5.0      # Range separation (ω)
    alpha = 0.19  # Short-range HF fraction
    beta = 0.07  # Long-range addition (Total LR = alpha + beta = 1.0)
    # #Tuned t6
    # mu = 0.15      # Range separation (ω)
    # alpha = 0.10  # Short-range HF fraction
    # beta = 0.00   # Long-range addition (Total LR = alpha + beta = 1.0)

    # --- Universal Weight Calculator ---
    sr_hf_weight = alpha
    lr_hf_weight = alpha + beta
    lr_b88_weight = 1.0 - lr_hf_weight
    sr_b88_weight = beta
    mf.omega = mu
    # The Universal Custom String
    mf.xc = f'{sr_hf_weight}*SR_HF({mu}) + {lr_hf_weight}*LR_HF({mu}) + {lr_b88_weight}*B88 + {sr_b88_weight}*BECKESRX, 0.81*LYPC + 0.19*VWN5C'

    # 4. Initialize the Driver and attach our custom calculator
    driver = PySCFDriver(
        atom=geometry_name,
        basis="6-31g*",
        method=MethodType.RKS,
        xc_functional=mf.xc,
    )
    driver._calc = mf
    driver._mol = mol

    # driver = PySCFDriver(
    #     atom=geometry_C16H10,
    #     basis="6-31g*",
    #     method=MethodType.RKS,
    #     xc_functional="camb3lyp",
    #     xcf_library="xcfun",
    #     spin=0
    # )

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
        mol_name = "FCI_Benzene_set10"  # Change this based on the geometry used (e.g., "H2O", "CO2", "Benzene", etc.)
        active_space_str = f"CAS_{active_num_electrons}e_{active_num_spatial_orbitals}o"

        result, embedding_energies = dft_solver.solve(driver, mol_name=mol_name, active_space_str=active_space_str)

        # ✅ Add this to print the final formatted result like in COBYLA
        print("\n=== DFT Embedding + VQE Results ===")
        print(result)

        # ================================================================
        #  >>> ADDED: Quantum Eigenstate Participation Ratio (FCI)
        # ================================================================
        try:
            print("\n=== Quantum Eigenstate PR Analysis (FCI) ===")
            from qiskit.quantum_info import Statevector

            # 1. Extract the exact ground state from the raw solver result.
            # Using a fallback to safely handle different Qiskit updates.
            if hasattr(result.raw_result, 'eigenstates'):
                exact_eigenstate = result.raw_result.eigenstates[0]
            else:
                exact_eigenstate = result.raw_result.eigenstate

            # 2. Convert it to a Qiskit Statevector object
            state = Statevector(exact_eigenstate)

            # 3. Get the probabilities (|c_gamma|^2) for every basis state
            probabilities = state.probabilities()

            # 4. Calculate Eigenstate PR: 1 / sum( (|c_gamma|^2)^2 )
            ipr = np.sum(probabilities**2)
            pr_eigenstate = 1.0 / ipr

            print(f"[Exact Eigenstate PR] {pr_eigenstate:.4f} (Effective configurations participating)")

            # Optional: Print the top contributing electron configurations
            print("Top contributing configurations (>1% probability):")
            top_indices = np.argsort(probabilities)[::-1]
            num_qubits = state.num_qubits

            for idx in top_indices:
                if probabilities[idx] > 0.01:
                    bin_str = format(idx, f"0{num_qubits}b")
                    print(f"  |{bin_str}> : Probability = {probabilities[idx]*100:.2f}%")

        except Exception as e:
            print(f"[FCI PR Analysis] Failed: {e}")

    except Exception as e:
        print(f"[Warning] VQE encountered issue ({e}); continuing with last results.")
        result, embedding_energies = None, []


    # 1. Calculate the 'Classical Embedding Overhead' (Total loop time minus VQE)
    # Your "Embedding" label already isolates the embedding logic from the active space solver.
    classical_embedding_overhead = PROFILE["Embedding"]

    # Calculate the 'Total measured time' by adding all distinct phases
    total = (
        PROFILE.get("DFT_reference", 0.0)
        + PROFILE["DFT"]
        + PROFILE["Embedding"]
        + PROFILE["Active Space"]
    )

    print("\n================ PROFILING SUMMARY ================")
    print(f"DFT Reference time            : {PROFILE.get('DFT_reference', 0.0):.2f} s")
    print(f"DFT (Energy Evaluator) time   : {PROFILE['DFT']:.2f} s")
    print(f"Classical Embedding Overhead  : {PROFILE['Embedding']:.2f} s")
    print(f"Active Space (Exact/FCI) time : {PROFILE['Active Space']:.2f} s")
    print("--------------------------------------------------")
    print(f"Total measured time           : {total:.2f} s")

    print("\nMemory snapshots (RSS, MB):")
    for k, v in MEMORY_SNAPSHOTS.items():
        print(f"{k:30s}: {v:8.2f} MB")
    print("==================================================")

# Run main when executed as a script
if __name__ == "__main__":
    _main()
