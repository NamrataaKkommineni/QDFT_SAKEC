# ================================================================
# SCRIPT 1: CLASSICAL VQE (Seeded, Qiskit 1.0, Callbacks & Prints)
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
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.primitives import Estimator

# --- SEEDING FOR REPRODUCIBILITY ---
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
algorithm_globals.random_seed = seed_value

AS = False

geometry = "O 0.0 0.0 0.0; O 0.0 0.0 1.21"
driver = PySCFDriver(atom=geometry, 
                     basis="6-31g*", 
                     spin=2, 
                     method=MethodType.ROKS, 
                     xc_functional="LDA",
                     xcf_library="xcfun"
                    )

base_problem = driver.run() 
driver._calc.e_tot = 0 
active_space = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=6)

as_problem = active_space.transform(base_problem)
active_num_spatial_orbitals = as_problem.num_spatial_orbitals
num_particles = as_problem.num_particles

base_mapper = ParityMapper(num_particles=num_particles)
mapper = as_problem.get_tapered_mapper(base_mapper)

initial_state = HartreeFock(active_num_spatial_orbitals, num_particles, mapper)
optimizer = COBYLA(maxiter=100) 

estimator = Estimator() 

# --- IITB SNIPPET ---
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

# 1. Filter Double Excitations
excitation_list_raw = var_form._get_excitation_list()
fer_excitation_op = var_form.excitation_ops()
excitation_list_pauli = list()
excitation_list = list() 

for i, ex in enumerate(fer_excitation_op):
    mapped_op = mapper.map(ex)
    if mapped_op is not None:
        excitation_list_pauli.append(mapped_op)
        excitation_list.append(excitation_list_raw[i])

if driver._calc.e_tot == 0:
    driver._calc.kernel()

base_problem = driver.to_problem()
as_problem = active_space.transform(base_problem)
second_q_op = as_problem.hamiltonian.second_q_op()
qubit_op = mapper.map(second_q_op)

pruned_excitation_list_pauli = list()
for i in range(len(excitation_list_pauli)):
    var_form1 = EvolvedOperatorAnsatz(excitation_list_pauli[i], initial_state=initial_state)
    initial_job = estimator.run([var_form1], [qubit_op], parameter_values=[[0.0]])
    initial_energy = initial_job.result().values[0]
    vqe1 = VQE(estimator, var_form1, optimizer=optimizer, initial_point=[0.0])
    vqe_result = vqe1.compute_minimum_eigenvalue(qubit_op)
    E1 = np.real(vqe_result.eigenvalue)
    op_pt = vqe_result.optimal_point
    first_list.append(op_pt[0])
    
    if abs(initial_energy - E1) > 1e-5:
        energy_list.append(E1)
        optimal_list.append(op_pt[0])
        pruned_excitation_list.append(excitation_list[i])
        pruned_excitation_list_pauli.append(excitation_list_pauli[i])
        
    # RESTORED PRINT
    print('state and average value is', excitation_list[i], E1)

if optimal_list:
    paired_lists = zip(optimal_list, pruned_excitation_list_pauli)
    sorted_pairs = sorted(paired_lists, key=lambda pair: abs(pair[0]), reverse=True)
    optimal_list = [pair[0] for pair in sorted_pairs]
    pruned_excitation_list_pauli = [pair[1] for pair in sorted_pairs]
    
    # RESTORED SORTED PRINT
    print("\n=== Sorted Double Excitations (Descending by Magnitude) ===")
    for i in range(len(optimal_list)):
        print(f"Param: {optimal_list[i]:.8f} | Operator: {pruned_excitation_list_pauli[i]}")

var_form_s = UCC(
    num_spatial_orbitals=active_num_spatial_orbitals,
    num_particles=num_particles,
    qubit_mapper=mapper,
    initial_state=initial_state,
    excitations='s'
)
excitation_list_s = var_form_s._get_excitation_list()
fer_excitation_op_s = var_form_s.excitation_ops()

# 2. Filter Single Excitations
valid_singles_count = 0
for ex in fer_excitation_op_s:
    mapped_op = mapper.map(ex)
    if mapped_op is not None:
        pruned_excitation_list_pauli.append(mapped_op)
        valid_singles_count += 1

# RESTORED PRINT
print(excitation_list_s)

ansatz = EvolvedOperatorAnsatz(pruned_excitation_list_pauli, initial_state=initial_state)
padded_initial_point = optimal_list + [0.0] * valid_singles_count
# --- END IITB SNIPPET ---


# --- VQE CALLBACK (PER ITERATION LOGIC) ---
evals_per_iteration = ansatz.num_parameters + 1
vqe_history = {"evals": [], "energies": [], "step_energies": []}

def vqe_callback(eval_count, parameters, mean, std):
    if eval_count == 1:
        print(f"\n  [VQE] Starting Optimization...")
        
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
            
        vqe_history["step_energies"].append(mean)


# --- ACTUAL CLASSICAL VQE ---
lbfgsb_optimizer = L_BFGS_B(maxiter=99)
classical_vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=lbfgsb_optimizer,
    initial_point=padded_initial_point,
    callback=vqe_callback
)

print("\n>>> Running Classical VQE...")
classical_result = classical_vqe.compute_minimum_eigenvalue(qubit_op)
optimized_parameters = classical_result.optimal_point

# --- ADD SHIFT BACK IN ---
energy_shift = as_problem.nuclear_repulsion_energy + as_problem.hamiltonian.constants.get('ActiveSpaceTransformer', 0.0)
total_energy = classical_result.eigenvalue.real + energy_shift

print(f"\nClassical VQE Energy (Active Space): {classical_result.eigenvalue.real:.6f} Ha")
print(f"Energy Shift (Nuclear + Frozen Core): {energy_shift:.6f} Ha")
print(f"Total Ground State Energy: {total_energy:.6f} Ha")
print(f"Optimized Parameters: {optimized_parameters}")

# Save parameters to a file to pass to the Quantum Script
np.save("optimal_vqe_params.npy", optimized_parameters)
print("Saved optimized parameters to 'optimal_vqe_params.npy'")