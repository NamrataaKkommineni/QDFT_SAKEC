# Essential Imports
import time
import numpy as np
import warnings

#Qiskit Imports
from qiskit import transpile
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.mappers import ParityMapper, TaperedQubitMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer
from qiskit_nature.second_q.circuit.library import HartreeFock 
from qiskit.circuit.library import efficient_su2
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit_algorithms.utils import algorithm_globals

# ==============================================================================
# STEP 1: GLOBAL CONFIGURATION & SEEDING
# ==============================================================================
warnings.filterwarnings("ignore")
algorithm_globals.random_seed = 320

# ==============================================================================
# STEP 2: CONVERGENCE CALLBACK UTILITY
# ==============================================================================
class SpsaConvergenceCallback:
    def __init__(self, target_value, threshold):
        self.target_value = target_value
        self.threshold = threshold
        self.prev_fx = None
        self.energies = []
        
    def __call__(self, nfev, x, fx, stepsize, accepted):
        self.energies.append(fx)
        print(f"Iteration {nfev//3} Energy {fx:.6f}")
        if self.prev_fx is not None:
            if abs(fx - self.prev_fx) < self.threshold:
                print(f"Iteration {nfev//3} - Energy {fx:.6f}, Delta(E) {abs(fx - self.prev_fx):.6f}")
                raise StopIteration('Both value and threshold convergence reached')
        self.prev_fx = fx

# ==============================================================================
# STEP 3: DEFINE THE CORE VQE WORKFLOW FUNCTION
# ==============================================================================
def get_ground_state_energy(mol_str, threshold=1e-6, active_electrons=2, active_orbitals=6, spin_multiplicity=1, charge=0):
    
    # ==========================================================================
    # STEP 4: MOLECULAR DATA GENERATION (PYSCF DRIVER)
    # ==========================================================================
    driver = PySCFDriver(atom=mol_str, 
                         basis='6-31g*', 
                         spin=spin_multiplicity-1, 
                         charge=charge, 
                         method=MethodType.ROKS
    )
    problem = driver.run()
    print("-" * 10 + " DFT Reference Energy is: " + str(problem.reference_energy) + " Ha " + "-" * 10)
    
    # ==========================================================================
    # STEP 5: HAMILTONIAN SIMPLIFICATION (FREEZE CORE & ACTIVE SPACE)
    # ==========================================================================
    fc_transformer = FreezeCoreTransformer(freeze_core=True)
    problem_fc = fc_transformer.transform(problem)
    as_transformer = ActiveSpaceTransformer(num_electrons=active_electrons, num_spatial_orbitals=active_orbitals)
    problem_reduced = as_transformer.transform(problem_fc)
    
    # ==========================================================================
    # STEP 6: SIMPLIFIED QUBIT MAPPING (TaperedQubitMapper)
    # ==========================================================================
    mapper = TaperedQubitMapper(ParityMapper())
    
    # ==========================================================================
    # STEP 7: GENERATE QUBIT HAMILTONIAN
    # ==========================================================================
    hamiltonian = mapper.map(problem_reduced.hamiltonian.second_q_op())
    
    num_orbitals = problem_reduced.num_orbitals
    num_particles = problem_reduced.num_particles
    
    # ==========================================================================
    # STEP 8: PRE-COMPUTE HARTREE-FOCK INITIAL STATE
    # ==========================================================================
    hf_state = HartreeFock(
        num_spatial_orbitals=num_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper 
    )
    
    # ==========================================================================
    # STEP 9: ANSATZ DEFINITION (EFFICIENT SU2)
    # ==========================================================================
    ansatz = efficient_su2(
        num_qubits=hamiltonian.num_qubits, 
        reps=1, 
        entanglement='linear', 
        initial_state=hf_state
    ).decompose()
    
    # ==========================================================================
    # STEP 10: BACKEND SELECTION & PRIMITIVE INITIALIZATION
    # ==========================================================================
    service = QiskitRuntimeService()
    real_backend = service.least_busy(operational=True, simulator=False)
    
    # Real Hardware
    # estimator = Estimator(mode=real_backend)
    
    # Local
    estimator = AerEstimatorV2.from_backend(real_backend)
    
    # Options
    # 1. No of Shots
    estimator.options.default_shots = 2048
    
    # 2. Resilience Level 
    # resilience_level = 0: None
    # resilience_level = 1: T-REX
    # resilience_level = 2: T-REX + ZNE
    estimator.options.resilience_level = 0
    
    # ==========================================================================
    # STEP 11: TRANSPILE CIRCUIT FOR BACKEND ARCHITECTURE
    # ==========================================================================
    ansatz = transpile(ansatz, backend=real_backend)
    
    # ==========================================================================
    # STEP 12: CALCULATE CONSTANT ENERGY SHIFT & SETUP CALLBACK
    # ==========================================================================
    energy_shift = sum(problem_reduced.hamiltonian.constants.values())
    callback_handler = SpsaConvergenceCallback(target_value=problem.reference_energy, threshold=threshold)
    
    # ==========================================================================
    # STEP 13: OPTIMIZER & INITIAL PARAMETER CONFIGURATION
    # ==========================================================================
    initial_point = np.zeros(ansatz.num_parameters)
    optimizer = SPSA(maxiter=100, callback=callback_handler)
    
    # ==========================================================================
    # STEP 14: VQE ALGORITHM EXECUTION
    # ==========================================================================
    vqe = VQE(estimator, ansatz, optimizer) 
    
    try:
        result = vqe.compute_minimum_eigenvalue(hamiltonian, initial_point=initial_point)
        final_energy = result.eigenvalue + energy_shift
    except StopIteration:
        final_energy = result.prev_fx + energy_shift

    return final_energy

# ==============================================================================
# STEP 15: MAIN EXECUTION & PERFORMANCE MEASUREMENT
# ==============================================================================
start_time = time.time()
print("Setting up H2 Calculation")
final_energy = get_ground_state_energy(
    'H 0 0 0; H 0 0 0.74',
    active_electrons=2, 
    active_orbitals=2
)

time_taken = time.time() - start_time
print("Final Energy Retrieved: ", final_energy)
print("Time Taken: ", time_taken)