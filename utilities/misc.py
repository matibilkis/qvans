import numpy as np
import cirq

def dict_to_json(dictionary):
    d="{"
    for k,v in dictionary.items():
        if isinstance(k,str):
            d+='\"{}\":\"{}\",'.format(k,v)
        else:
            d+='\"{}\":{},'.format(k,v)
    d=d[:-1]
    d+="}" #kill the comma
    return "\'"+d+ "\'"


def scheduler_selector_temperature(energy, lowest_energy_found, when_on=10):
    relative_energy = np.abs((energy - lowest_energy_found)/lowest_energy_found)
    if relative_energy < 1e-1:
       return 1#
    else:
       return when_on

def scheduler_parameter_perturbation_wall(its_without_improvig, max_randomness=.8, min_randomness=.1, decrease_to=20):
    slope = (max_randomness-min_randomness)/decrease_to
    return np.min([min_randomness + slope*its_without_improvig, max_randomness])

def give_kronecker_of_list(lista):
    #lista=[auto_handler.zero_proj(vqe_handler.qubits[k]).matrix() for k in range(3)]
    m=[]
    for ind,k in enumerate(lista):
        if ind == 0:
            m.append(k)
        else:
            m.append(np.kron(m[-1],k))
    return m[-1]


def give_kr_prod(matrices):
    #matrices list of 2 (or more in principle) matrices
    while len(matrices) != 1:
        sm, smf=[],[]
        for ind in range(len(matrices)):
            sm.append(matrices[ind])
            if len(sm) == 2:
                smf.append(np.kron(*sm))
                sm=[]
        matrices = smf
    return matrices[0]

# def compute_ground_energy(self):
#     ground_energy = np.min(np.linalg.eigvals(sum(self.observable).matrix()))
#     return ground_energy
# def compute_ground_energy_1(obse,qubits):
#     """
#     TO do. Implement this for, say, 6 qubits (therer's a problem in give_kr_prod...)
#     """
#     ind_to_2 = {"0":np.eye(2), "1":cirq.unitary(cirq.X), "2":cirq.unitary(cirq.Y), "3":cirq.unitary(cirq.Z)}
#     ham = np.zeros((2**len(qubits),2**len(qubits))).astype(np.complex128)
#     for kham in obse:
#         item= kham.dense(qubits)
#         string = item.pauli_mask
#         matrices = [ind_to_2[str(int(ok))] for ok in string]
#         ham += give_kr_prod(matrices)*item.coefficient
#     return np.sort(np.real(np.linalg.eigvals(ham)))

# def compute_ground_energy(obse,qubits):
#     """
#     TO do. Implement this for, say, 6 qubits (therer's a problem in give_kr_prod...)
#     """
#     if -np.log2(len(qubits)).is_integer() is True:
#         ind_to_2 = {"0":np.eye(2), "1":cirq.unitary(cirq.X), "2":cirq.unitary(cirq.Y), "3":cirq.unitary(cirq.Z)}
#         ham = np.zeros((2**len(qubits),2**len(qubits))).astype(np.complex128)
#         for kham in obse:
#             item= kham.dense(qubits)
#             string = item.pauli_mask
#             matrices = [ind_to_2[str(int(ok))] for ok in string]
#             ham += give_kr_prod(matrices)*item.coefficient
#         return np.sort(np.real(np.linalg.eigvals(ham)))
#     else:
#         return [-np.inf]
#

    #
    #     if  < 0.05:
    #         return 0
    #     else:
    #         return
    # else:
    #     return
