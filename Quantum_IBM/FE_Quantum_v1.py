from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np

def compute_gs_energy(molecule_str, spin_multiplicity=1, charge=0, num_elec=2, num_orb=2):
    driver = PySCFDriver(atom=molecule_str, basis='sto3g', 
                         spin=spin_multiplicity-1, charge=charge, method=MethodType.ROHF)
    problem_raw = driver.run()
    
    transformer = ActiveSpaceTransformer(num_electrons=num_elec, num_spatial_orbitals=num_orb)
    problem = transformer.transform(problem_raw)
    
    parity_mapper = ParityMapper(num_particles=problem.num_particles)
    mapper = problem.get_tapered_mapper(parity_mapper)
    
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())
    
    hf_state = HartreeFock(num_spatial_orbitals=num_orb, 
                           num_particles=problem.num_particles, 
                           qubit_mapper=mapper)
    
    ansatz = EfficientSU2(num_qubits=hamiltonian.num_qubits, 
                          entanglement='linear', reps=1, initial_state=hf_state)
    
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False)
    
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    ansatz_isa = pm.run(ansatz)
    hamiltonian_isa = hamiltonian.apply_layout(ansatz_isa.layout)
    
    estimator = Estimator(mode=backend)
    estimator.options.default_precision = 1e-3
    estimator.options.resilience_level = 0
    
    eval_count = 0
    def objective(params):
        nonlocal eval_count
        job = estimator.run([(ansatz_isa, hamiltonian_isa, params)])
        energy = float(job.result()[0].data.evs)
        eval_count += 1
        if eval_count % 10 == 0:
            print(f"  -> Evaluation {eval_count}: Energy = {energy:.6f}")
        return energy
        
    initial_point = np.zeros(ansatz_isa.num_parameters)
    optimizer = SPSA(maxiter=100, tol=1e-3)
    opt_result = optimizer.minimize(objective, initial_point)
    
    total_internal_shift = sum(problem.hamiltonian.constants.values())
    total_energy = opt_result.fun + total_internal_shift

    print(f"\n--- Result for {molecule_str.split()[0]} ---")
    print(f"VQE Minimized Energy: {opt_result.fun:.6f} Ha")
    print(f"Final Total Ground State Energy: {total_energy:.6f} Ha")

    return total_energy

print("Starting H2 (2e, 2o)...")
E_H2 = compute_gs_energy('H 0 0 0; H 0 0 0.74', num_elec=2, num_orb=2)
print("H2 Energy:", E_H2)

# print("\nStarting O2 (6e, 6o)...")
# E_O2 = compute_gs_energy('O 0 0 0; O 0 0 1.21', spin_multiplicity=3, num_elec=6, num_orb=6)
# print("O2 Energy:", E_O2)

# print("\nStarting H2O (6e, 6o)...")
# E_H2O = compute_gs_energy('O 0 0 0; H 0 0.757 0.586; H 0 0.757 -0.586', num_elec=6, num_orb=6)
# print("H2O Energy:", E_H2O)

# formation_energy = E_H2O - E_H2 - 0.5 * E_O2
# print(f"\nFinal Formation Energy: {formation_energy:.4f} Ha")