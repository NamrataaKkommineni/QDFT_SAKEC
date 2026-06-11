# ================================================================
# FULL ANSATZ & RESOURCE MEASUREMENT SCRIPT (WITH PLOTTING)
# ================================================================
import numpy as np
import random
import matplotlib.pyplot as plt
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
from qiskit.providers.fake_provider import GenericBackendV2

# --- 1. SEEDING ---
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
algorithm_globals.random_seed = seed_value

# Mock parameters for measurement
optimized_parameters = np.array([0.0]) 

# --- 2. MOLECULAR SETUP ---
geometry = (
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
 )
H2 = (
    "H    0.000000    0.000000   -0.371394; "
    "H    0.000000    0.000000    0.371394"
)
Co2 = (
    "O   0.000000   0.000000  -1.169590;"
    "C   0.000000   0.000000   0.000000;"
    "O   0.000000   0.000000   1.169590"
)
benzene = (
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
)
pyridene = (
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
)

naphthalene = (
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
)
h2o = (
    "O    0.000000    0.000000    0.118852; "
    "H    0.000000    0.762815   -0.478289; "
    "H    0.000000   -0.762815   -0.478289"
) #H2O_optimized
LiH = (
    "Li  0.000000  0.000000  -0.013277; "
    "H  0.000000  0.000000  1.608016"
)
driver = PySCFDriver(atom=LiH, basis="6-31g*", spin=0, method=MethodType.ROKS)
base_problem = driver.run()     
driver._calc.e_tot = 0 

active_space = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2)
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

# --- 4. DEBUGGING / MEASURING CONSTANTS ---
print("\n" + "="*45)
print("     RESOURCE MEASUREMENT (DEBUG MODE)")
print("="*45)

grouped_ops = qubit_op.group_commuting()
nc = len(grouped_ops)

for i, group in enumerate(grouped_ops):
    print(f"Group {i+1} Pauli Strings:{group.paulis}")

fake_backend = GenericBackendV2(num_qubits=127)
pm = generate_preset_pass_manager(optimization_level=3, backend=fake_backend)
isa_ansatz = pm.run(ansatz)

depth = isa_ansatz.depth()
ops = isa_ansatz.count_ops()
multi_qubit_gates = ops.get('cx', 0) + ops.get('ecr', 0) + ops.get('cz', 0)

print(f"Number of Parameters:            {ansatz.num_parameters}")
print(f"Number of Grouped Circuits (Nc): {nc}")
print(f"ISA Circuit Depth (D):           {depth}")
print(f"Multi-qubit Gates:               {multi_qubit_gates}")
print(f"Total Shots (at 4000/circuit):   {nc * 4000:,}")
print("-" * 45)

# Plot ISA (Instruction Set Architecture) Ansatz (Hardware-ready)
print("Generating ISA Ansatz Plot (Sample snippet)...")
# Drawing only a portion if it's too deep for clarity
# isa_ansatz.draw('mpl', idle_wires=False, fold=50)
# plt.title(f"ISA Transpiled Ansatz (Depth: {depth})")
# plt.show()