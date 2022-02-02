from utilities.circuit_basics import Basic
import numpy as np
import cirq
import sympy
import tensorflow_quantum as tfq
import tensorflow as tf
import pandas as pd

class UnitaryMurder(Basic):
    def __init__(self, minimizer,testing=False, accept_wall=1e2):
        """
        Scans a circuit, evaluates mean value of observable and retrieves a shorter circuit if the energy is not increased too much.

        Takes as input vqe_handler object inheriting its observable

        The expected value of hamiltonian is computed via vqe_handler.give_energy(indexed_circuit, resolver). Importantly, for noisy channels that are decomposed as sum of unitary transf, the accuracy of this expectation will depend on the q_batch_size (how many circuits we are considering, each suffering from each unitary transf with the corresponding probability)

        """
        super(UnitaryMurder, self).__init__(n_qubits=minimizer.n_qubits, testing=testing)
        self.single_qubit_unitaries = {"rx":cirq.rx, "rz":cirq.rz}
        self.observable = minimizer.observable
        self.accept_wall=accept_wall
        self.minimizer = minimizer

    def unitary_slaughter(self, circuit_db, displaying=False):

        original_batch = self.prepare_channel_discrimination_circuits(circuit_db)
        perr = self.minimizer.give_energy(original_batch)

        max_its = len(circuit_db[circuit_db["trainable"]!=False])
        reduced = True
        count=0

        while reduced is True and count < max_its:
            if count==0:
                self.initial_perr = perr
            circuit_db, perr, reduced = self.kill_one_unitary_in_blocks(circuit_db)
            count+=1
            if (displaying and reduced) is True :
                print("I killed {} unitaries, Pf - Pi: {}".format(count, perr-self.initial_perr))
        return circuit_db, perr, reduced, count



    def kill_one_unitary_in_blocks(self,circuit_db):
        """
        this function scans
        """
        trainable_blocks_dict = self.give_trainable_blocks(circuit_db)
        simplified_db_in_blocks = []

        original_batch = self.prepare_channel_discrimination_circuits(circuit_db)
        perr = self.minimizer.give_energy(original_batch)

        for block_id, trainable in trainable_blocks_dict.items():
            if trainable == False:
                continue
            else:
                circuit_db_block_symboled = circuit_db[circuit_db.block_id==block_id]
                circuit_db_block = self.give_circuit_from_db(circuit_db_block_symboled.drop("symbol",axis=1))[-1]

                indexed_circuit = list(circuit_db_block.ind)
                symbol_to_value = self.give_symbol_to_value(circuit_db_block)
                index_to_symbols = self.give_index_to_symbols(circuit_db_block)
                first_symbol_number, first_ind = self.give_proper_indices(circuit_db_block_symboled)

                Kcircuit_db, Kperr, bool_killed = self.kill_one_unitary(indexed_circuit, symbol_to_value, index_to_symbols, block_id, circuit_db, perr)

                if bool_killed is True: #if there's no bool_killed means that the whole circuit is fixed-structure!
                    circuit_db, perr, bool_killed = Kcircuit_db, Kperr, bool_killed
                    break

        return circuit_db, perr, bool_killed


    def kill_one_unitary(self, indexed_circuit, symbol_to_value, index_to_symbols, block_id, original_circuit_db, original_perr):
        """
        This method kills one unitary, looping on the circuit and, if finding a parametrized gate, computes the
        energy of a circuit without it.

        If energy is at least %99, then returns the shorter circuit.
        """

        ###### STEP 1: COMPUTE ORIGINAL ENERGY ####
        #reconstructed_circuit_db = self.give_circuit_db(block_id, original_circuit_db,  indexed_circuit, symbol_to_value, index_to_symbols)

        first_ind = original_circuit_db[original_circuit_db["block_id"] == block_id].index[0]
        first_symbol_number = int(original_circuit_db[original_circuit_db["block_id"] == block_id]["symbol"].dropna().iloc[0].replace("th_",""))

        circuit_proposals=[]
        circuit_proposals_energies=[]

        self.indexed_circuit = indexed_circuit
        self.symbol_to_value = symbol_to_value

        ###### STEP 2: Loop over original circuit. #####
        for index_victim, victim in enumerate(indexed_circuit):
            #this first index will be the one that - potentially - will be deleted
            if victim < self.number_of_cnots:
                pass
            else:
                info_gate = [index_victim, victim]
                valid, proposal_indexed_circuit, proposal_symbols_to_values, prop_cirq_circuit = self.create_proposal_without_gate(info_gate)

                if valid:

                    resymbols = self.reindex_symbol(list(proposal_symbols_to_values.keys()), first_symbol_number)
                    symbols, params = self.fill_values_with_nan(proposal_indexed_circuit, resymbols, list(proposal_symbols_to_values.values()))

                    new_circuit_db = {"ind": proposal_indexed_circuit, "symbol":symbols, "param_value":params,
                                      "trainable":[True]*len(proposal_indexed_circuit), "block_id":[block_id]*len(proposal_indexed_circuit)}
                    new_circuit_db = pd.DataFrame(new_circuit_db)
                    new_circuit_db.index = pd.Index(range(first_ind, first_ind+len(new_circuit_db)))

                    pd_complement = original_circuit_db[~(original_circuit_db["block_id"] == block_id)]
                    pdd = pd.concat([pd_complement,new_circuit_db])
                    pdd = pdd.sort_index().reset_index(drop=True)

                    proposed_batch = self.prepare_channel_discrimination_circuits(pdd)
                    proposal_perr = self.minimizer.give_energy(proposed_batch)

                    if self.accepting_criteria(proposal_perr):
                        circuit_proposals.append(pdd)
                        circuit_proposals_energies.append(proposal_perr)

        ### STEP 3: keep the one of lowest energy (if there're many)
        if len(circuit_proposals)>0:
            favourite = np.argmin(circuit_proposals_energies)
            killed_pd, killed_perr = circuit_proposals[favourite], circuit_proposals_energies[favourite]
            return killed_pd, killed_perr, True
        else:
            return original_circuit_db, original_perr, False


    def accepting_criteria(self, e_new):
        """
        if decreases energy, we accept it;
        otherwise exponentially decreasing probability of acceptance (the 100 is yet another a bit handcrafted)
        """

        error =  e_new-self.initial_perr#
        if e_new <= self.initial_perr:
            return True
        else:
            return np.random.random() < np.exp(-error*self.accept_wall) ### eventually allow for veeery tiny increment in the error probability


    def create_proposal_without_gate(self, info_gate):
        """
        Create a circuit without the gate corresponding to info_gate.
        Also, if the new circuit has no gates enough, returns bool value (valid)
        so to not consider this into tfq.expectation_layer.
        """

        index_victim, victim = info_gate

        proposal_circuit=[]
        proposal_symbols_to_values = {}
        prop_cirq_circuit=cirq.Circuit()

        ko=0 #index_of_smymbols_added_to_circuit
        for ind_survivors, gate_survivors in enumerate(self.indexed_circuit):
            if gate_survivors < self.number_of_cnots:
                proposal_circuit.append(gate_survivors)
                control, target = self.indexed_cnots[str(gate_survivors)]
                prop_cirq_circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            else:
                if ind_survivors != index_victim:
                    proposal_circuit.append(gate_survivors)
                    qubit = self.qubits[(gate_survivors-self.number_of_cnots)%self.n_qubits]
                    new_param = "th_"+str(len(proposal_symbols_to_values.keys()))
                    if 0 <= gate_survivors-self.number_of_cnots < self.n_qubits:
                        prop_cirq_circuit.append(cirq.rz(sympy.Symbol(new_param)).on(qubit))
                    else:
                        prop_cirq_circuit.append(cirq.rx(sympy.Symbol(new_param)).on(qubit))
                    #### add value to resolver ####
                    proposal_symbols_to_values[new_param] = self.symbol_to_value["th_"+str(ko)]
                ko+=1

        connections, _ = self.scan_qubits(proposal_circuit)
        valid=True
        #now check if we have killed all the gates in a given qubit. If so, will return valid=False
        for q, path in connections.items():
            if len(path) == 0:
                valid = False
            else:
                if ("rx" not in path) and ("rz" not in path):
                    valid = False

        return valid, proposal_circuit, proposal_symbols_to_values, prop_cirq_circuit


    def scan_qubits(self, indexed_circuit):
        """
        this function scans the circuit as described by {self.indexed_circuit}
        and returns a dictionary with the gates acting on each qubit and the order of appearence on the original circuit.

        It's the same than Simplifier method.
        """
        connections={str(q):[] for q in range(self.n_qubits)} #this saves the gates at each qubit. It does not respects the order.
        places_gates = {str(q):[] for q in range(self.n_qubits)} #this saves, for each gate on each qubit, the position in the original indexed_circuit
        flagged = [False]*len(indexed_circuit) #used to check if you have seen a cnot already, so not to append it twice to the qubit's dictionary

        for nn,idq in enumerate(indexed_circuit): #sweep over all gates in original circuit's list
            for q in range(self.n_qubits): #sweep over all qubits
                if idq<self.number_of_cnots: #if the gate it's a CNOT or not
                    control, target = self.indexed_cnots[str(idq)] #give control and target qubit
                    if q in [control, target] and not flagged[nn]: #if the qubit we are looking at is affected by this CNOT, and we haven't add this CNOT to the dictionary yet
                        connections[str(control)].append(idq)
                        connections[str(target)].append(idq)
                        places_gates[str(control)].append(nn)
                        places_gates[str(target)].append(nn)
                        flagged[nn] = True #so you don't add the other
                else:
                    if (idq-self.number_of_cnots)%self.n_qubits == q: #check if the unitary is applied to the qubit we are looking at
                        if 0 <= idq - self.number_of_cnots< self.n_qubits:
                            connections[str(q)].append("rz")
                        elif self.n_qubits <= idq-self.number_of_cnots <  2*self.n_qubits:
                            connections[str(q)].append("rx")
                        places_gates[str(q)].append(nn)
                    flagged[nn] = True #to check that all gates have been flagged
        ####quick test
        for k in flagged:
            if k is False:
                raise Error("not all flags in flagged are True!")
        return connections, places_gates
