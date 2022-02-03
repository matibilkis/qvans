import pandas as pd
import numpy as np
import cirq
from utilities.templates import gate_template
import sympy

class CirqTranslater:
    def __init__(self, n_qubits):
        """
        class that translates database to cirq circuits
        """
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)

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


    def append_to_circuit(self, gate_id, circuit, circuit_db):
        """
        adds gate_id instructions to current circuit. Returns new circuit (cirq object) and new circuit_db (pd.DataFrame)

        gate_id: dictionary containing gate info to append
        circuit: Cirq.Circuit object
        circuit_db: pandas DataFrame (all circuit info is appended to here)
        """
        ind = gate_id["ind"]
        gate_index = len(list(circuit_db.keys()))
        circuit_db[gate_index] = gate_id #this is the new item to add
        symbols = []
        for j in [k["symbol"] for k in circuit_db.values()]:
            if j != None:
                symbols.append(j)
        #### add CNOT
        if ind < self.number_of_cnots:
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

            if param_value is None:
                param_value = sympy.Symbol(symbol_name)
            circuit.append(gate(param_value).on(self.qubits[qubit]))
            return circuit, circuit_db

        else:
            raise AttributeError("Wrong index!")

    def give_circuit(self, dd):
        list_of_gate_ids = [gate_template(**dict(dd.iloc[k])) for k in range(len(dd))]
        circuit, circuit_db = [],{}
        for k in list_of_gate_ids:
            circuit , circuit_db = self.append_to_circuit(k,circuit, circuit_db)
        circuit = cirq.Circuit(circuit)
        circuit_db = pd.DataFrame.from_dict(circuit_db,orient="index")
        return circuit, circuit_db


    def get_symbols(self, circuit_db):
        trainable_symbols = circuit_db[circuit_db["trainable"] == True]["symbol"]
        return list(trainable_symbols[circuit_db["symbol"].notnull()])


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
        symbol_db = circuit_db.loc[circuit_db["symbol"].isin(list(symbol_to_value))]["symbol"]
        circuit_db["param_value"].update({ind:val for ind,val in zip(symbol_db.index, symbol_to_value.values())})
        return circuit_db