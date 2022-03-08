import pandas as pd
import numpy as np
import cirq
from utilities.templates import gate_template
import sympy

class CirqTranslater:
    def __init__(self, n_qubits, **kwargs):
        """
        class that translates database to cirq circuits
        """
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)

        ### blocks that are fixed-structure (i.e. channels, state_preparation, etc.)
        untouchable_blocks = kwargs.get("untouchable_blocks",[])
        self.untouchable_blocks = untouchable_blocks


        #### keep a register on which integers corresponds to which CNOTS, target or control.
        self.indexed_cnots = {}
        self.cnots_index = {}
        count = 0
        for control in range(self.n_qubits):
            for target in range(self.n_qubits):
                if control != target:
                    self.indexed_cnots[str(count)] = [control, target]
                    self.cnots_index[str([control,target])] = count
                    count += 1
        self.number_of_cnots = len(self.indexed_cnots)
        self.cgates = {0:cirq.rz, 1: cirq.rx, 2:cirq.ry}


    def append_to_circuit(self, gate_id, circuit, circuit_db, **kwargs):
        """
        adds gate_id instructions to current circuit. Returns new circuit (cirq object) and new circuit_db (pd.DataFrame)

        gate_id: dictionary containing gate info to append
        circuit: Cirq.Circuit object
        circuit_db: pandas DataFrame (all circuit info is appended to here)
        """
        unresolved = kwargs.get("unresolved",False)
        ind = gate_id["ind"]
        gate_index = len(list(circuit_db.keys()))
        circuit_db[gate_index] = gate_id #this is the new item to add

        ## the symbols are all elements we added but the very last one (added on the very previous line)
        symbols = []
        for j in [k["symbol"] for k in circuit_db.values()][:-1]:
            if j != None:
                symbols.append(j)
        ## custom gate
        if ind == -1: ## warning, no support on TFQ for the moment...
            circuit_db[gate_index]["symbol"] = None
            u=gate_id["param_value"]   ##param_value will be the unitary (np.array)
            q=gate_id["qubits"] #list
            qubits = circuit_db[gate_index]["qubits"]
            uu = cirq.MatrixGate(u)
            circuit.append(uu.on(*[self.qubits[qq] for qq in qubits]))
            return circuit, circuit_db
        #### add CNOT
        elif 0 <= ind < self.number_of_cnots:
            control, target = self.indexed_cnots[str(ind)]
            circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            circuit_db[gate_index]["symbol"] = None
            return circuit, circuit_db

        ### add HADDAMARD
        elif self.number_of_cnots + 3*self.n_qubits <= ind < self.number_of_cnots + 4*self.n_qubits:
            qubit  = (ind-self.number_of_cnots)%self.n_qubits
            circuit.append(cirq.H.on(self.qubits[qubit]))
            circuit_db[gate_index]["symbol"] = None
            return circuit, circuit_db

        #### add rotation
        elif self.number_of_cnots <= ind < self.number_of_cnots + 3*self.n_qubits:
            gate_type_index = (ind - self.number_of_cnots)//self.n_qubits
            qubit  = (ind-self.number_of_cnots)%self.n_qubits
            gate = self.cgates[gate_type_index]

            symbol_name = gate_id["symbol"]
            param_value = gate_id["param_value"]

            if symbol_name is None:
                symbol_name = "th_"+str(len(symbols))
                circuit_db[gate_index]["symbol"] = symbol_name

            else:
                if symbol_name in symbols:
                    print("warning, repeated symbol while constructing the circuit, see circuut_\n  symbol_name {}\n symbols {}\ncircuit_db {} \n\n\n".format(symbol_name, symbols, circuit_db))
            if (param_value is None) or (unresolved is True):
                if gate_id["trainable"] == True: ##only leave unresolved those gates that will be trianed
                    param_value = sympy.Symbol(symbol_name)
            circuit.append(gate(param_value).on(self.qubits[qubit]))
            return circuit, circuit_db

        else:
            raise AttributeError("Wrong index!", ind)

    def give_circuit(self, dd,**kwargs):
        """
        retrieves circuit from circuit_db. It is assumed that the order in which the symbols are labeled corresponds to order in which their gates are applied in the circuit.
        If unresolved is False, the circuit is retrieved with the values of rotations (not by default, since we feed this to a TFQ model)
        """
        unresolved = kwargs.get("unresolved",True)
        list_of_gate_ids = [gate_template(**dict(dd.iloc[k])) for k in range(len(dd))]
        circuit, circuit_db = [],{}
        for k in list_of_gate_ids:
            circuit , circuit_db = self.append_to_circuit(k,circuit, circuit_db, unresolved=unresolved)
        circuit = cirq.Circuit(circuit)
        circuit_db = pd.DataFrame.from_dict(circuit_db,orient="index")
        #### we make sure that the symbols appearing correspond to the ordering in which we add the gate to the circuit
        return circuit, circuit_db


    def get_trainable_symbols(self, circuit_db):
        trainable_symbols = circuit_db[circuit_db["trainable"] == True]["symbol"]
        return list(trainable_symbols[circuit_db["symbol"].notnull()])

    def get_trainable_params_value(self,circuit_db):
        index_trainable_params = circuit_db[circuit_db["trainable"] == True]["symbol"].dropna().index
        return circuit_db["param_value"][index_trainable_params]


    def give_trainable_parameters(self, circuit_db):
        indices =  circuit_db[circuit_db["trainable"] == True]["ind"]
        trainable_coefficients = indices[(indices < self.number_of_cnots+ 3*self.n_qubits) & (indices >= self.number_of_cnots)]
        return len(trainable_coefficients)

    def give_trainable_cnots(self, circuit_db):
        indices =  circuit_db[circuit_db["trainable"] == True]["ind"]
        cnots = indices[(indices < self.number_of_cnots)]
        return len(cnots)

    def update_circuit_db_param_values(self, circuit_db,symbol_to_value):
        """
        circuit_db (unoptimized) pd.DataFrame
        symbol_to_value: resolver, dict
        """
        trianables = circuit_db[circuit_db["trainable"] == True]
        trainable_symbols = trianables[~trianables["symbol"].isna()]
        circuit_db["param_value"].update({ind:val for ind, val in zip(trainable_symbols.index, symbol_to_value.values())})
        return circuit_db

    def give_resolver(self, circuit_db):
        trianables = circuit_db[circuit_db["trainable"] == True]
        trainable_symbols = trianables[~trianables["symbol"].isna()]
        return dict(trainable_symbols[["symbol","param_value"]].values)
