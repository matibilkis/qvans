from utilities.circuit_basics import Basic
import numpy as np
from copy import deepcopy

class IdInserter(Basic):
    def __init__(self, n_qubits=3,initialization="epsilon", epsilon=0.1,
                selector_temperature=10,focus_on_blocks=None):
        """
        epsilon: perturbation strength
        initialization: how parameters at ientity compilation are perturbated on single-qubit unitary.
                        Options = ["PosNeg", "epsilon"]
        focus_on_blocks: list or None; list should only include blocks that are not FIXED.
        """
        super(IdInserter, self).__init__(n_qubits=n_qubits)
        self.epsilon = epsilon
        self.init_params = initialization
        self.selector_temperature=selector_temperature
        self.focus_on_blocks = focus_on_blocks

    def place_identities(self, circuit_db, rate_iids_per_step=1):
        ngates = np.random.exponential(scale=rate_iids_per_step)
        ngates = int(ngates+1)

        Mcircuit_db = self.place_almost_identity(circuit_db)
        for ll in range(ngates-1):
            Mcircuit_db = self.place_almost_identity(Mcircuit_db)
        return Mcircuit_db

    def place_almost_identity(self, circuit_db):
        block_to_insert, insertion_index = self.choose_block(circuit_db)
        Icircuit_db = self.inserter(circuit_db, block_to_insert, insertion_index)
        return Icircuit_db

    def choose_block(self, circuit_db):
        """
        randomly choices an identity resolution and index to place it at indexed_circuit.
        """
        ngates = np.array(list(self.gate_counter_on_qubits(circuit_db["ind"]).values()))
        ngates_CNOT = ngates[:,1]
        qubits_not_CNOT = np.where(ngates_CNOT == 0)[0] ### target qubits are chosen later
        which_prob = lambda x: [.5,.5] if len(x)==0 else [.3,.7]

        #### CHOOSE BLOCK ####
        which_block = np.random.choice([0,1], p=which_prob(qubits_not_CNOT))
        #### CHOOSE qubits####
        qubits = self.choose_target_bodies(ngates=ngates,gate_type=["one-qubit","two-qubit"][which_block])
        #### CHOOSE INSERTION INDEX ####
        insertion_index = self.where_to_insert(circuit_db)
        ### retrieve block of gates ##
        if which_block==0:
            block_of_gates = self.resolution_1qubit(qubits)
        else:
            block_of_gates = self.resolution_2cnots(*qubits)

        return block_of_gates, insertion_index

    def choose_target_bodies(self, ngates, gate_type="one-qubit"):
        """
        gate_type: "one-qubit" or "two-qubit"
        ngates: gate_counter_on_qubits

        Note that selector_temperature could be annealed as energy decreases.. (at beta = 0 we get uniform sampling)
        function that selects qubit according to how many gates are acting on each one in the circuit
        """
        if gate_type == "one-qubit":
            gc=ngates[:,0]+1 #### gives the gate population for each qubit
            probs=np.exp(self.selector_temperature*(1-gc/np.sum(gc)))/np.sum(np.exp(self.selector_temperature*(1-gc/np.sum(gc))))
            return np.random.choice(range(self.n_qubits),1,p=probs)[0]
        else:
            gc=ngates[:,1]+1 #### gives the gate population for each qubit
            probs=np.exp(self.selector_temperature*(1-gc/np.sum(gc)))/np.sum(np.exp(self.selector_temperature*(1-gc/np.sum(gc))))
            qubits = np.random.choice(range(self.n_qubits),2,p=probs,replace=False)
            return qubits

    def where_to_insert(self, circuit_db):
        """
        There should be an order in blocks which are non-trainable
        """
        c1 = circuit_db[circuit_db["trainable"]==True]
        blocks = c1["block_id"].unique()
        if self.focus_on_blocks is not None:
            blocks_priority = np.unique(self.focus_on_blocks + blocks)
            probs = list(0.8*np.ones(len(self.focus_on_blocks))/len(self.focus_on_blocks) ) + \
            list(0.2* np.ones(len(blocks_priority[len(self.focus_on_blocks):]))/len(blocks_priority[len(self.focus_on_blocks):])
            )
            which_block = np.random.choice(blocks_priority, 1, p=probs)[0]
        else:
            which_block = np.random.choice(blocks, 1)[0]

        c2 = c1[c1["block_id"] == which_block]
        insertion_index = np.squeeze(np.random.choice(c2.index, 1)) ##there might be a problem here later...

        return insertion_index



    def resolution_2cnots(self, q1, q2):
        """
        sequence of integers describing a CNOT, then unitary (compiled close to identity, rz rx rz) and the same CNOT
        q1: control qubit
        q2: target qubit
        """
        if q1==q2:
            raise Error("SAME QUBIT!")
        rzq1 = self.number_of_cnots + q1
        rzq2 = self.number_of_cnots +  q2
        rxq1 = self.number_of_cnots + self.n_qubits + q1
        rxq2 = self.number_of_cnots + self.n_qubits + q2
        cnot = self.cnots_index[str([q1,q2])] #q1 control q2 target
        return [cnot, rzq1, rxq1, rzq1, rxq2, rzq2, rxq2, cnot]


    def resolution_1qubit(self, q):
        """
        retrieves rz rx rz on qubit q
        """
        rzq1 = self.number_of_cnots +  q
        rxq1 = self.number_of_cnots + self.n_qubits + q
        return [rzq1, rxq1, rzq1]


    def inserter(self, circuit_dbb, block_of_gates, insertion_index):
        """
        """
        circuit_db = deepcopy(circuit_dbb)

        indice = insertion_index
        block_id = circuit_db.iloc[insertion_index]["block_id"]

        for new_gate in block_of_gates:
            if new_gate<self.number_of_cnots:
                circuit_db.loc[int(indice)+0.1] = self.give_gate_template(new_gate, block_id=block_id)
                circuit_db = circuit_db.sort_index().reset_index(drop=True)
            else:
                ### check symbol number ###
                symbol_found=False
                for k in range(0, indice+1)[::-1]:
                    if type(circuit_db.loc[k]["symbol"]) == str:
                        number_symbol = int(circuit_db.loc[k]["symbol"].replace("th_","")) +1
                        symbol_found=True
                        break
                if not symbol_found:
                    number_symbol = 0

                circuit_db.loc[int(indice)+0.1] = self.give_gate_template(new_gate, param_value=np.random.random()*self.epsilon, symbol="th_"+str(number_symbol), block_id=block_id)
                circuit_db = circuit_db.sort_index().reset_index(drop=True)

                for k in range(indice+2, circuit_db.shape[0]):
                    if circuit_db.loc[k]["ind"] < self.number_of_cnots or type(circuit_db.loc[k]["symbol"]) != str:
                        pass
                    else:
                        old_value = circuit_db.iloc[k]["symbol"]
                        number_symbol = int(old_value.replace("th_","")) +1
                        new_value = "th_{}".format(number_symbol)
                        circuit_db.loc[k] = circuit_db.loc[k].replace(to_replace=old_value,value=new_value)
            indice+=1
        return circuit_db
