import tensorflow_quantum as tfq
import cirq
import numpy as np

def give_observable(model,problem_config):
    #### CONDENSED MATTER HAMILTONIANS ####
    if problem_config["problem"].upper() in ["XXZ","TFIM"]:
        model.problem_nature = "cm"
        for field in ["g","J"]:
            if field not in problem_config.keys():
                raise ValueError("You have not specified the fields correctly. Check out your problem_config back again. Current dict: {}".format(problem_config))
        if problem_config["problem"].upper()=="TFIM":
            #H = -J \sum_i^{n} X_i X_{i+1} - g \sum_i^{n} Z_i
            observable = [-float(problem_config["g"])*cirq.Z.on(q) for q in model.qubits]
            for q in range(len(model.qubits)):
                observable.append(-float(problem_config["J"])*cirq.X.on(model.qubits[q])*cirq.X.on(model.qubits[(q+1)%len(model.qubits)]))
            return observable
        elif problem_config["problem"].upper()=="XXZ":
            #H = \sum_i^{n} X_i X_{i+1} + Y_i Y_{i+1} + J Z_i Z_{i+1} + g \sum_i^{n} \sigma_i^{z}
            observable = [float(problem_config["g"])*cirq.Z.on(q) for q in model.qubits]
            for q in range(len(model.qubits)):
                observable.append(cirq.X.on(model.qubits[q])*cirq.X.on(model.qubits[(q+1)%len(model.qubits)]))
                observable.append(cirq.Y.on(model.qubits[q])*cirq.Y.on(model.qubits[(q+1)%len(model.qubits)]))
                observable.append(float(problem_config["J"])*cirq.Z.on(model.qubits[q])*cirq.Z.on(model.qubits[(q+1)%len(model.qubits)]))
            return observable

    elif problem_config["problem"].upper() in ["H2","H4","LiH"]:
        model.problem_nature = "chemical"
        oo = ChemicalObservable()
        for key,defvalue in zip(["geometry","multiplicity", "charge", "basis"], [None,1,0,"sto-3g"]):
            if key not in list(problem_config.keys()):
                raise ValueError("{} not specified in problem_config. Dictionary obtained: {}".format(key, problem_config))
        observable, model.lower_bound_energy =oo.give_observable(model.qubits, problem_config["geometry"], problem_config["multiplicity"], problem_config["charge"], problem_config["basis"],return_lower_bound=model.return_lower_bound)
        return observable

    elif problem_config["problem"].upper() in ["QADC","DISCRIMINATION"]:
        return [cirq.Z.on(q) for q in model.qubits]

    else:
        raise AttributeError("check out your hamiltonians")


def compute_lower_bound_cost(model,lower_bound, compute_lower_bound):
    if (lower_bound == -np.inf) or lower_bound is None:
        if compute_lower_bound is not True:
            return -np.inf
        else:
            return np.real(np.min(np.linalg.eigvals(sum(model.observable).matrix())))
    else:
        return lower_bound
