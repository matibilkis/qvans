import numpy as np
from utilities.compiling import *
from utilities.vqe import *
from utilities.discrimination import *
from utilities.misc import shift_symbols_down

class GateKiller:
    def __init__(self,
                translator,
                mode,
                **kwargs):
        """
        To do: extend this to the other problems..
        """
        self.translator = translator
        self.max_relative_increment = kwargs.get("max_relative_increment", 0.05)

        if mode.upper() == "VQE":
            hamiltonian = kwargs.get("hamiltonian")
            params = kwargs.get("params")
            self.observable = give_observable_vqe(self.translator,hamiltonian, params)
            self.loss = EnergyLoss()
            self.model_class = QNN_VQE

        elif mode.upper() == "DISCRIMINATION":

            params = kwargs.get("params")
            number_hyp = kwargs.get("number_hyp",2)
            self.observable = [cirq.Z.on(q) for q in self.translator.qubits]
            self.loss = PerrLoss(discard_qubits=self.translator.env_qubits, number_hyp = number_hyp)
            self.model_class = QNN_DISCRIMINATION
            self.discrimination_params = params

        elif mode.upper() == "COMPILING":
            self.observable = give_observable_compiling(self.translator)
            self.loss = CompilingLoss(d = self.translator.n_qubits)
            self.model_class = QNN_Compiling

    def give_cost_external_model(self, batched_circuit, model):
        return self.loss(*[model(batched_circuit)]*2) ###useful for unitary killer


    def remove_irrelevant_gates(self,initial_cost, circuit_db):
        first_cost = initial_cost
        number_of_gates = len(self.translator.get_trainable_params_value(circuit_db))
        for murder_attempt in range(number_of_gates):
            circuit_db, new_cost, killed = self.kill_one_unitary(first_cost, circuit_db)
            print("kill 1qbit gate, try {}/{}. Increased by: {}%".format(murder_attempt, number_of_gates, (initial_cost-new_cost)/np.abs(initial_cost)))
            if killed is False:
                break
        return circuit_db, new_cost

    def kill_one_unitary(self, initial_cost, circuit_db):

        blocks = list(set(circuit_db["block_id"]))
        for b in self.translator.untouchable_blocks:
            blocks.remove(b)

        candidates = []
        for b in blocks:
            block_db = circuit_db[circuit_db["block_id"] == b]
            block_db_trainable = block_db[block_db["trainable"] == True]
            block_db_trainable = block_db_trainable[~block_db_trainable["symbol"].isna()]
            block_db_trainable = block_db_trainable[~block_db_trainable["symbol"].isna()]
            candidates += list(block_db_trainable.index)

        killed_costs = []

        for index_candidate in candidates:
            killed_circuit_db = circuit_db.copy()
            killed_circuit_db = killed_circuit_db.drop(labels=[index_candidate])
            killed_circuit_db = shift_symbols_down(self.translator, index_candidate+1, killed_circuit_db)

            if self.model_class.__name__ == 'QNN_DISCRIMINATION':
                killed_batch_circuits, survival_symbols, survival_params_value = prepare_optimization_discrimination(self.translator, killed_circuit_db, self.discrimination_params )
                unitary_killer_model = self.model_class(survival_symbols, self.observable)
                #try:
                unitary_killer_model(killed_batch_circuits)
                unitary_killer_model.trainable_variables[0].assign(tf.convert_to_tensor(survival_params_value.astype(np.float32)))
                killed_costs.append(self.give_cost_external_model(killed_batch_circuits, model=unitary_killer_model).numpy())
                # except Exception: ###it happened in the past that the circuit was too short. TO do, solve this in a more elegant way.
                #     killed_costs.append(np.inf)
                #     print("problem in unitary killer, index_candidate {}".format(index_candidate))
            else:
                raise NotImplementedError("...")


        relative_increments = (np.array(killed_costs)-initial_cost)/np.abs(initial_cost)
        if np.min(relative_increments) < self.max_relative_increment:
            index_to_kill = np.argmin(relative_increments)
            new_cost = killed_costs[index_to_kill]
            killed_circuit_db = circuit_db.copy()
            killed_circuit_db = killed_circuit_db.drop(labels=[index_to_kill])
            killed_circuit_db = shift_symbols_down(self.translator, index_to_kill+1, killed_circuit_db)
            killed_circuit_db = killed_circuit_db.sort_index().reset_index(drop=True)
            return killed_circuit_db, new_cost, True
        else:
            return circuit_db, initial_cost, False
