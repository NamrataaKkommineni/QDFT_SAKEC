import numpy as np
from qiskit_nature.second_q.circuit.library import UCC
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_algorithms.minimum_eigensolvers import VQE

def build_iitb_sd_ansatz(
    num_spatial_orbitals,
    num_particles,
    qubit_mapper,
    initial_state,
    qubit_op,
    estimator,
    optimizer,
    energy_threshold=1e-5
):
    print("\n[IITB Pruning] Starting SD Ansatz Generation...")
    first_list = []
    energy_list = []
    optimal_list = []
    pruned_excitation_list = []
    var_form = UCC(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=qubit_mapper,
        initial_state=initial_state,
        excitations='d'
    )
    excitation_list = var_form._get_excitation_list()
    fer_excitation_op = var_form.excitation_ops()
    excitation_list_pauli = list()
    for ex in fer_excitation_op:
        excitation_list_pauli.append(qubit_mapper.map(ex))
    print(f"[IITB Pruning] Total 'd' excitations BEFORE pruning: {len(excitation_list_pauli)}")
    print(excitation_list)
    pruned_excitation_list_pauli, excitation_list_pruned, difference_E = [], [], []

    for i in range(len(excitation_list_pauli)):
        var_form1 = EvolvedOperatorAnsatz(excitation_list_pauli[i], initial_state=initial_state)
        initial_job = estimator.run([(var_form1, qubit_op, [0.0])])
        initial_energy = initial_job.result()[0].data.evs
        vqe1 = VQE(estimator, var_form1, optimizer=optimizer, initial_point=[0.0])
        vqe_result = vqe1.compute_minimum_eigenvalue(qubit_op)
        E1 = np.real(vqe_result.eigenvalue)
        op_pt = vqe_result.optimal_point
        first_list.append(op_pt[0])
        if abs(initial_energy - E1) > energy_threshold:
            energy_list.append(E1)
            difference_E.append(abs(initial_energy - E1))
            optimal_list.append(op_pt[0])
            excitation_list_pruned.append(excitation_list[i])
            pruned_excitation_list.append(excitation_list[i])
            pruned_excitation_list_pauli.append(excitation_list_pauli[i])
        print('state and average value is', excitation_list[i],E1)

    difference_E1 = difference_E.copy()
    difference_E1.sort(reverse=True)
    
    print('Sorted Differences Energy',difference_E1)
    dob_excitation_list = list()
    dob_energy_list = list()
    dob_params = list()
    final_dob_list = list()
    for iii in range(len(difference_E1)):
        eee = difference_E1[iii]
        for jjj in range(len(difference_E)):
            eee1 = difference_E[jjj]
            if eee1==eee:
                dob_excitation_list.append(excitation_list_pruned[jjj])
                dob_energy_list.append(energy_list[jjj])
                dob_params.append(optimal_list[jjj])
                final_dob_list.append(pruned_excitation_list_pauli[jjj])

                difference_E[jjj]=0.0

    print('Sorted Doubles List',dob_excitation_list)

    var_form_s = UCC(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=qubit_mapper,
        initial_state=initial_state,
        excitations='s'
    )
    excitation_list_s = var_form_s._get_excitation_list()
    fer_excitation_op_s = var_form_s.excitation_ops()
    for ex in fer_excitation_op_s:
        final_dob_list.append(qubit_mapper.map(ex))
    print(excitation_list_s)
    ansatz = EvolvedOperatorAnsatz(final_dob_list, initial_state=initial_state)
    padded_initial_point = dob_params + [0.0] * len(fer_excitation_op_s)

    total_excitations_count = len(dob_excitation_list) + len(excitation_list_s)
    print(f"\n[Ansatz Composition] Pruned Double Excitations: {len(dob_excitation_list)}")
    print(f"[Ansatz Composition] Single Excitations        : {len(excitation_list_s)}")
    print(f"[Ansatz Composition] Total Operators Sized     : {total_excitations_count}")
    print(f"Length of Pauli list acting on Ansatz          : {len(final_dob_list)}")

    return ansatz, padded_initial_point

def build_iitb_triples_ansatz(
    num_spatial_orbitals,
    num_particles,
    qubit_mapper,
    initial_state,
    qubit_op,
    estimator,
    optimizer,
    energy_threshold=1e-5
):
    print("\n[IITB Pruning] Starting Triples Ansatz Generation...")
    first_list = []
    energy_list = []
    optimal_list = []
    pruned_excitation_list = []
    var_form = UCC(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=qubit_mapper,
        initial_state=initial_state,
        excitations='d'
    )
    excitation_list = var_form._get_excitation_list()
    fer_excitation_op = var_form.excitation_ops()
    excitation_list_pauli = list()
    for ex in fer_excitation_op:
        excitation_list_pauli.append(qubit_mapper.map(ex))
    print(f"\n[IITB Pruning] Total 'd' excitations BEFORE pruning: {len(excitation_list_pauli)}")
    print(excitation_list)

    pruned_excitation_list_pauli = list()
    excitation_list_pruned=list()
    difference_E=list()
    for i in range(len(excitation_list_pauli)):
        var_form1 = EvolvedOperatorAnsatz(excitation_list_pauli[i], initial_state=initial_state)
        initial_job = estimator.run([(var_form1, qubit_op, [0.0])])
        initial_energy = initial_job.result()[0].data.evs
        vqe1 = VQE(estimator, var_form1, optimizer=optimizer, initial_point=[0.0])
        vqe_result = vqe1.compute_minimum_eigenvalue(qubit_op)
        E1 = np.real(vqe_result.eigenvalue)
        op_pt = vqe_result.optimal_point
        first_list.append(op_pt[0])
        if abs(initial_energy - E1) > energy_threshold:
            energy_list.append(E1)
            difference_E.append(abs(initial_energy - E1))
            optimal_list.append(op_pt[0])
            excitation_list_pruned.append(excitation_list[i])
            pruned_excitation_list.append(excitation_list[i])
            pruned_excitation_list_pauli.append(excitation_list_pauli[i])
        print('state and average value is', excitation_list[i],E1)

    difference_E1 = difference_E.copy()
    difference_E1.sort(reverse=True)
    
    print('Sorted Differences Energy',difference_E1)
    dob_excitation_list = list()
    dob_energy_list = list()
    dob_params = list()
    final_dob_list = list()
    for iii in range(len(difference_E1)):
        eee = difference_E1[iii]
        for jjj in range(len(difference_E)):
            eee1 = difference_E[jjj]
            if eee1==eee:
                dob_excitation_list.append(excitation_list_pruned[jjj])
                dob_energy_list.append(energy_list[jjj])
                dob_params.append(optimal_list[jjj])
                final_dob_list.append(pruned_excitation_list_pauli[jjj])
                difference_E[jjj]=0.0
    print('Sorted Doubles List',dob_excitation_list)
    final_excitations_list = []
    final_parameters_list = []
    final_pauli_list = list()
    Sh_list= [(( 6 , 0 ),( 9 , 2 )),(( 6 , 1 ),( 9 , 2 )),(( 6 , 0 ),( 10 , 2 )),(( 6 , 1 ),( 10 , 2 )),(( 6 , 0 ),( 11 , 2 )),(( 6 , 1 ),( 11 , 2 )),(( 7 , 0 ),( 9 , 2 )),(( 7 , 1 ),( 9 , 2 )),(( 7 , 0 ),( 10 , 2 )),(( 7 , 1 ),( 10 , 2 )),(( 7 , 0 ),( 11 , 2 )),(( 7 , 1 ),( 11 , 2 )),(( 8 , 0 ),( 9 , 2 )),(( 8 , 1 ),( 9 , 2 )),(( 8 , 0 ),( 10 , 2 )),(( 8 , 1 ),( 10 , 2 )),(( 8 , 0 ),( 11 , 2 )),(( 8 , 1 ),( 11 , 2 )),(( 0 , 6 ),( 3 , 8 )),(( 0 , 7 ),( 3 , 8 )),(( 0 , 6 ),( 4 , 8 )),(( 0 , 7 ),( 4 , 8 )),(( 0 , 6 ),( 5 , 8 )),(( 0 , 7 ),( 5 , 8 )),(( 1 , 6 ),( 3 , 8 )),(( 1 , 7 ),( 3 , 8 )),(( 1 , 6 ),( 4 , 8 )),(( 1 , 7 ),( 4 , 8 )),(( 1 , 6 ),( 5 , 8 )),(( 1 , 7 ),( 5 , 8 )),(( 2 , 6 ),( 3 , 8 )),(( 2 , 7 ),( 3 , 8 )),(( 2 , 6 ),( 4 , 8 )),(( 2 , 7 ),( 4 , 8 )),(( 2 , 6 ),( 5 , 8 )),(( 2 , 7 ),( 5 , 8 ))]

    Sp_list= [(( 6 , 3 ),( 9 , 4 )),(( 6 , 3 ),( 9 , 5 )),(( 6 , 3 ),( 10 , 4 )),(( 6 , 3 ),( 10 , 5 )),(( 6 , 3 ),( 11 , 4 )),(( 6 , 3 ),( 11 , 5 )),(( 7 , 3 ),( 9 , 4 )),(( 7 , 3 ),( 9 , 5 )),(( 7 , 3 ),( 10 , 4 )),(( 7 , 3 ),( 10 , 5 )),(( 7 , 3 ),( 11 , 4 )),(( 7 , 3 ),( 11 , 5 )),(( 8 , 3 ),( 9 , 4 )),(( 8 , 3 ),( 9 , 5 )),(( 8 , 3 ),( 10 , 4 )),(( 8 , 3 ),( 10 , 5 )),(( 8 , 3 ),( 11 , 4 )),(( 8 , 3 ),( 11 , 5 )),(( 0 , 9 ),( 3 , 10 )),(( 0 , 9 ),( 3 , 11 )),(( 0 , 9 ),( 4 , 10 )),(( 0 , 9 ),( 4 , 11 )),(( 0 , 9 ),( 5 , 10 )),(( 0 , 9 ),( 5 , 11 )),(( 1 , 9 ),( 3 , 10 )),(( 1 , 9 ),( 3 , 11 )),(( 1 , 9 ),( 4 , 10 )),(( 1 , 9 ),( 4 , 11 )),(( 1 , 9 ),( 5 , 10 )),(( 1 , 9 ),( 5 , 11 )),(( 2 , 9 ),( 3 , 10 )),(( 2 , 9 ),( 3 , 11 )),(( 2 , 9 ),( 4 , 10 )),(( 2 , 9 ),( 4 , 11 )),(( 2 , 9 ),( 5 , 10 )),(( 2 , 9 ),( 5 , 11 ))]

    def S_list_h(num_spatial_orbitals, num_particles):
        ex = Sh_list
        return ex
    ucc_sh = UCC(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, excitations=S_list_h, qubit_mapper=qubit_mapper, initial_state=initial_state)
    sh_fer_ops = ucc_sh.excitation_ops()
    sh_pauli = list()
    for fer in sh_fer_ops:
        sh_pauli.append(qubit_mapper.map(fer))
    
    def S_list_p(num_spatial_orbitals, num_particles):
        ex = Sp_list
        return ex
    ucc_sp = UCC(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles, excitations=S_list_p, qubit_mapper=qubit_mapper, initial_state=initial_state)
    sp_fer_ops = ucc_sp.excitation_ops()
    sp_pauli = list()
    for fer in sp_fer_ops:
        sp_pauli.append(qubit_mapper.map(fer))

    for index in range(len(dob_excitation_list)):
        double = dob_excitation_list[index]
        ref_E = dob_energy_list[index]
        i, j, a, b = double[0][0], double[0][1], double[1][0], double[1][1]
        final_excitations_list.append(double)
        final_parameters_list.append(dob_params[index])
        final_pauli_list.append(final_dob_list[index])
        dummy_pauli = list()
        t2_sh_list = []
        for kkk in range(len(Sh_list)):
            sh = Sh_list[kkk]
            p, q, c, d = sh[0][0], sh[0][1], sh[1][0], sh[1][1]
            if i == d or j == d:
                        #array_1, array_2 = np.array([i, j]), np.array([a, b])
                if (p != i) and (p != j) and (q != i) and (q != j) and (c != a) and (c != b):
                    print(sh)
                        #t2_sh_list.append(sh)
                    dummy_pauli = list()
                    dummy_pauli.append(final_dob_list[index])
                    dummy_pauli.append(sh_pauli[kkk])
                    ucc_custom_2 = EvolvedOperatorAnsatz(dummy_pauli, initial_state = initial_state)
                    init_par = [dob_params[index]]+[0.01]
                    vqe1 = VQE(estimator, ucc_custom_2, optimizer=optimizer, initial_point=init_par)
                    vqe_result = vqe1.compute_minimum_eigenvalue(qubit_op)
                    E1 = np.real(vqe_result.eigenvalue)
                    #print(E1)
                    op_pt = vqe_result.optimal_point

                    if abs(E1-ref_E) > 1e-5:
                        final_excitations_list.append(sh)
                        final_parameters_list.append(op_pt[-1])
                        t2_sh_list.append(sh)
                        final_pauli_list.append(sh_pauli[kkk])
        for kkk in range(len(Sp_list)):
            sp = Sp_list[kkk]
            p, q, c, d = sp[0][0], sp[0][1], sp[1][0], sp[1][1]
            if a == q or b == q:
                        #array_1, array_2 = np.array([i, j]), np.array([a, b])
                if (a != c) and (a != d) and (b != c) and (b != d) and (p != i) and (p != j):
                    print(sp)
                        #t2_sh_list.append(sp)
                    dummy_pauli = list()
                    dummy_pauli.append(final_dob_list[index])
                    dummy_pauli.append(sp_pauli[kkk])
                    ucc_custom_2 = EvolvedOperatorAnsatz(dummy_pauli, initial_state = initial_state)
                    init_par = [dob_params[index]]+[0.01]
                    vqe1 = VQE(estimator, ucc_custom_2, optimizer=optimizer, initial_point=init_par)
                    vqe_result = vqe1.compute_minimum_eigenvalue(qubit_op)
                    E1 = np.real(vqe_result.eigenvalue)
                    #print(E1)
                    op_pt = vqe_result.optimal_point

                    if abs(E1-ref_E) > 1e-5:
                        final_excitations_list.append(sp)
                        final_parameters_list.append(op_pt[-1])
                        t2_sh_list.append(sp)
                        final_pauli_list.append(sp_pauli[kkk])

        print('Doubles',double,'çorresponding S',t2_sh_list)

    var_form_s = UCC(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=qubit_mapper,
        initial_state=initial_state,
        excitations='s'
    )
    excitation_list_s = var_form_s._get_excitation_list()
    fer_excitation_op_s = var_form_s.excitation_ops()
    
    for ex in fer_excitation_op_s:
        final_pauli_list.append(qubit_mapper.map(ex))
    for ex in excitation_list_s:
        final_excitations_list.append(ex)
    print(excitation_list_s)
    ansatz = EvolvedOperatorAnsatz(final_pauli_list, initial_state=initial_state)
    padded_initial_point = final_parameters_list + [0.0] * len(fer_excitation_op_s)
    print(final_excitations_list,'Length',len(final_excitations_list))
    print('Length of pauli',len(final_pauli_list))

    return ansatz, padded_initial_point