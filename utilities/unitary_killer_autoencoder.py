from utilities.circuit_basics import Basic
import numpy as np
import cirq
import sympy
import tensorflow_quantum as tfq
import tensorflow as tf
from utilities.qmodels import QNN, EnergyLoss
import copy
class UnitaryMurder(Basic):
    def __init__(self, au_handler, many_indexed_circuits, many_symbols_to_values,
        noise_config={},testing=False,accept_wall=1e5):
        """
        Scans a circuit, evaluates mean value of observable and retrieves a shorter circuit if the energy is not increased too much.

        Takes as input vqe_handler object inheriting its observable and noise attributes.

        The expected value of hamiltonian is computed via vqe_handler.give_energy(indexed_circuit, resolver). Importantly, for noisy channels that are decomposed as sum of unitary transf, the accuracy of this expectation will depend on the q_batch_size (how many circuits we are considering, each suffering from each unitary transf with the corresponding probability)

        """
        super(UnitaryMurder, self).__init__(n_qubits=au_handler.n_qubits, testing=testing, noise_config=noise_config)
        self.single_qubit_unitaries = {"rx":cirq.rx, "rz":cirq.rz}
        self.observable = au_handler.observable
        self.initial_energy = -np.inf #this is to compare with iniital_energy, at each round
        self.accept_wall=accept_wall
        self.qbatch = self.give_batch_of_circuits(many_indexed_circuits, many_symbols_to_values)#mixed states


    def give_batch_of_circuits(self, listas, resolvers):
        """
        this guy gives the mixed state i form of batched circuits (wegiths are added later)
        """
        qbatch=[]
        for pure_indexed, resolver in zip(listas, resolvers):
            circuit=self.give_circuit(pure_indexed)[0]
            preparation_circuit=cirq.resolve_parameters(circuit, resolver)
            qbatch.append(preparation_circuit)
        return qbatch


    def give_energy(self, indexed_circuit, symbols_to_values):
        """
        CHANGE NAMES
        """
        au_circuit,symbols  = self.give_circuit(indexed_circuit)[0:2]
        qbatch=[]
        # qq = copy.deepcopy(self.qbatch)
        qq=copy.deepcopy(self.qbatch)
        for qc in qq:
            qc.append(au_circuit)
            qbatch.append(qc)

        model = QNN(symbols=symbols, observable=self.observable, batch_sizes=len(qbatch))
        tfqcircuit=tfq.convert_to_tensor(qbatch)
        model(tfqcircuit)
        model.compile(loss=EnergyLoss(mode_var="autoencoder"))
        model.trainable_variables[0].assign(tf.convert_to_tensor(np.array(list(symbols_to_values.values())).astype(np.float32)))
        pred = model(tfqcircuit)
        loss =model.compiled_loss(pred,pred)
        return np.squeeze(loss) #(1/self.nb)*np.squeeze(model.compiled_loss(pred,pred))/len(self.qbatch) #los estados de la mezcla, sent with equal priors.


    def unitary_slaughter(self, indexed_circuit, symbol_to_value, index_to_symbols,reference_energy):
        max_its = len(indexed_circuit)
        reduced = True
        count=0
        while reduced is True and count < max_its:
            if count==0:
                self.initial_energy = np.squeeze(reference_energy)
            indexed_circuit, symbol_to_value, index_to_symbols, energy, reduced = self.kill_one_unitary(indexed_circuit, symbol_to_value, index_to_symbols)
            count+=1
            print("I killed {} unitaries, Ef - Ei: {}".format(count, energy-self.initial_energy))
        return indexed_circuit, symbol_to_value, index_to_symbols, energy, reduced

    def kill_one_unitary(self, indexed_circuit, symbol_to_value, index_to_symbols):
        """
        This method kills one unitary, looping on the circuit and, if finding a parametrized gate, computes the
        energy of a circuit without it.

        If energy is at least %99, then returns the shorter circuit.
        """

        ###### STEP 1: COMPUTE ORIGINAL ENERGY ####
        original_energy = self.give_energy(indexed_circuit, symbol_to_value)

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
                valid, proposal_circuit, proposal_symbols_to_values, prop_cirq_circuit = self.create_proposal_without_gate(info_gate) #notice prop_cirq_circuit beomes useless if noise.-
                if valid:
                    proposal_energy = self.give_energy(proposal_circuit, proposal_symbols_to_values)

                    if self.accepting_criteria(proposal_energy):
                        circuit_proposals.append([proposal_circuit, proposal_symbols_to_values,proposal_energy])
                        circuit_proposals_energies.append(proposal_energy)

        ### STEP 3: keep the one of lowest energy (if there're many)
        if len(circuit_proposals)>0:
            favourite = np.where(np.array(circuit_proposals_energies) == np.min(circuit_proposals_energies))[0][0]
            short_circuit, symbol_to_value, energy = circuit_proposals[favourite]
            _,_, index_to_symbols = self.give_circuit(short_circuit)
            return short_circuit, symbol_to_value, index_to_symbols, circuit_proposals_energies[favourite], True
        else:
            return indexed_circuit, symbol_to_value, index_to_symbols, original_energy, False


    def accepting_criteria(self, e_new):
        """
        if decreases energy, we accept it;
        otherwise exponentially decreasing probability of acceptance (the 100 is yet another a bit handcrafted)
        """
        #return  < 0.01
        e_old = self.initial_energy
        relative_error = (e_new-e_old)/np.abs(e_old)
        if e_new <= e_old:
            return True
        else:
            return np.random.random() < np.exp(-np.abs(relative_error)*self.accept_wall)


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
