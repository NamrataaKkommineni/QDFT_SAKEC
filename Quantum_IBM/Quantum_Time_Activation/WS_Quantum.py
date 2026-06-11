# ================================================================
# FULL ANSATZ & EXECUTION SCRIPT (Qiskit 1.0+ & Runtime 0.34+)
# Optimized for IBM Open Plan & ISA Compliance
# ================================================================
import numpy as np
import random
from qiskit_algorithms.utils import algorithm_globals
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

# V2 Primitives and Transpilation
from qiskit.primitives import StatevectorEstimator as LocalEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

# --- 1. SEEDING ---
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
algorithm_globals.random_seed = seed_value

try:
    optimized_parameters = np.load("optimal_vqe_params.npy")
except FileNotFoundError:
    print("Error: 'optimal_vqe_params.npy' not found.")
    exit()

# --- 2. MOLECULAR SETUP ---
geometry = "O 0.0 0.0 0.0; O 0.0 0.0 1.21"
driver = PySCFDriver(atom=geometry, basis="6-31g*", spin=2, method=MethodType.RKS)
base_problem = driver.run() 
driver._calc.e_tot = 0 

active_space = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=6)
as_problem = active_space.transform(base_problem)
mapper = as_problem.get_tapered_mapper(ParityMapper(num_particles=as_problem.num_particles))
initial_state = HartreeFock(
    num_spatial_orbitals=as_problem.num_spatial_orbitals,
    num_particles=as_problem.num_particles,
    qubit_mapper=mapper
)

# --- 3. ANSATZ BUILDING (Pruned UCC) ---
var_form = UCC(
    num_spatial_orbitals=as_problem.num_spatial_orbitals,
    num_particles=as_problem.num_particles,
    qubit_mapper=mapper,
    initial_state=initial_state,
    excitations='d'
)
fer_excitation_op = var_form.excitation_ops()
excitation_list_pauli = [mapper.map(ex) for ex in fer_excitation_op if mapper.map(ex) is not None]

driver._calc.kernel()
as_problem = active_space.transform(driver.to_problem())
qubit_op = mapper.map(as_problem.hamiltonian.second_q_op())

local_est = LocalEstimator()
optimizer = COBYLA(maxiter=100)
pruned_excitation_list_pauli = []

for op in excitation_list_pauli:
    var_form1 = EvolvedOperatorAnsatz(op, initial_state=initial_state)
    initial_job = local_est.run([(var_form1, qubit_op, [0.0])])
    initial_energy = initial_job.result()[0].data.evs
    
    vqe1 = VQE(local_est, var_form1, optimizer=optimizer, initial_point=[0.0])
    vqe_result = vqe1.compute_minimum_eigenvalue(qubit_op)
    
    if abs(vqe_result.optimal_value - initial_energy) > 1e-5:
        pruned_excitation_list_pauli.append(op)

var_form_s = UCC(
    num_spatial_orbitals=as_problem.num_spatial_orbitals,
    num_particles=as_problem.num_particles,
    qubit_mapper=mapper,
    initial_state=initial_state,
    excitations='s'
)
pruned_excitation_list_pauli.extend([mapper.map(ex) for ex in var_form_s.excitation_ops() if mapper.map(ex) is not None])

# The "Abstract" Ansatz
ansatz = EvolvedOperatorAnsatz(pruned_excitation_list_pauli, initial_state=initial_state)

# --- 4. HARDWARE TRANSPILATION & EXECUTION ---
print("\n>>> Connecting to IBM Quantum...")
service = QiskitRuntimeService(channel="ibm_quantum_platform") 
backend = service.least_busy(operational=True, simulator=False)

# NEW: Create a PassManager to transform the circuit for the specific backend
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

print(f"Transpiling circuit for {backend.name}...")
isa_ansatz = pm.run(ansatz)
# NEW: The observable must also be transformed to match the transpiled circuit's layout
isa_qubit_op = qubit_op.apply_layout(isa_ansatz.layout)

energy_shift = as_problem.nuclear_repulsion_energy + as_problem.hamiltonian.constants.get('ActiveSpaceTransformer', 0.0)

# Initialize Estimator in Job Mode
estimator = Estimator(mode=backend)
estimator.options.resilience_level = 0 # Basic error mitigation recommended for ISA circuits
estimator.options.default_shots = 1000

print(f"Submitting ISA-compliant job to: {backend.name}")
job = estimator.run([(isa_ansatz, isa_qubit_op, optimized_parameters.tolist())])

print(f"Job ID: {job.job_id()}")
print("Waiting for results...")

result = job.result()[0].data.evs
total_energy = float(result) + energy_shift

print("\n" + "="*35)
print("      FINAL QUANTUM RESULTS")
print("="*35)
print(f"Total Energy (Ha):   {total_energy:.6f}")
print(f"Backend Name:        {backend.name}")
print("="*35)