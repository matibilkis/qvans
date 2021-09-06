import cirq
import numpy as np
import sympy
import pickle
import os
from datetime import datetime
from functools import wraps
import errno
import os
import signal


class Basic:
    def __init__(self, n_qubits=3, testing=False, noise_config={}):
        """
        n_qubits: number of qubits on your ansatz

        testing: this is inherited by other classes to ease the debugging.

        noise_config: implemented in batches.
            if self.noise_config = {} ---> no noise
            else: passed thorugh the Basic, to inherit the circuit_with_noise
                if should be in the form of {"channel":"depolarizing", "channel_params":array, "q_batch_size":int}

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

        self.testing=testing
        if not isinstance(noise_config, dict):
            print("noise_config should be passed as dict, in a form of, see docs")
        elif noise_config == {} :
            self.q_batch_size = 1
            self.noise=False
        else:
            self.define_channel_things(noise_config)

    def define_channel_things(self, noise_config):
        """
        options: "depolarizing" (symetric depolarizing channel) {"channel":"depolarizing", "channel_params":[p], "q_batch_size":10**3}
        """
        self.noise = True
        self.channel = noise_config["channel"]
        try:
            self.channel_params = eval(noise_config["channel_params"])
            self.q_batch_size = eval(noise_config["q_batch_size"])
        except Exception or Error: #if you are not passing from parser args
            self.channel_params = noise_config["channel_params"]
            self.q_batch_size = noise_config["q_batch_size"]
        if self.channel == "depolarizing":
            self.channel_unitaries = [cirq.I, cirq.X, cirq.Y, cirq.Z]
            self.number_noisy_unitaries = len(self.channel_unitaries)
            p = self.channel_params[0]
            self.channel_params = [1-p, p/3, p/3, p/3]
            #self.number_noisy_unitaries, self.channel_unitaries, self.channel_params
            return

    def append_to_circuit(self, ind, circuit, params, index_to_symbols):
        """
        ind: integer describing the gate to append to circuit
        circuit: cirq object that describes the quantum circuit
        params: a list containing the symbols appearing in the circuit so far
        index_to_sybols: tells which symbol corresponds to i^{th} item in the circuit (useful for simplifier)
        """
        #### add CNOT
        if ind < self.number_of_cnots:
            control, target = self.indexed_cnots[str(ind)]
            circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            if isinstance(index_to_symbols,dict):
                index_to_symbols[len(list(index_to_symbols.keys()))] = []
                return circuit, params, index_to_symbols
            else:
                return circuit, params

        #### add rz #####
        elif 0 <= ind - self.number_of_cnots  < self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            for par, gate in zip(range(1),[cirq.rz]):
                new_param = "th_"+str(len(params))
                params.append(new_param)
                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_param
                return circuit, params, index_to_symbols

        #### add rx #####
        elif self.n_qubits <= ind - self.number_of_cnots  < 2*self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            for par, gate in zip(range(1),[cirq.rx]):
                new_param = "th_"+str(len(params))
                params.append(new_param)
                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_param
            return circuit, params, index_to_symbols

        #### add ry #####
        elif 2*self.n_qubits <= ind - self.number_of_cnots  < 3*self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            for par, gate in zip(range(1),[cirq.ry]):
                new_param = "th_"+str(len(params))
                params.append(new_param)
                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_param
            return circuit, params, index_to_symbols

    def append_to_circuit_with_noise(self, ind, circuit, params, index_to_symbols):
        """
        ind: integer describing the gate to append to circuit
        circuit: cirq object that describes the quantum circuit
        params: a list containing the symbols appearing in the circuit so far
        index_to_sybols: tells which symbol corresponds to i^{th} item in the circuit (useful for simplifier)

        Note. This is the noisy version, in randomly choose unitaries before each gate is placed (if a CNOT is placed we add gate on both
        control and target). self.channel_unitaries is a list with the unitary transf into which the channel is decomposed, of length self.number_noisy_unitaries.
        """

        #### add CNOT
        if ind < self.number_of_cnots:
            control, target = self.indexed_cnots[str(ind)]
            #adding noise
            ngates = np.random.choice(self.channel_unitaries, 2,p=self.channel_params)
            for ng,nq in zip(ngates,[control,target]):
                circuit.append(ng.on(self.qubits[nq]))

            circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            if isinstance(index_to_symbols,dict):
                index_to_symbols[len(list(index_to_symbols.keys()))] = []
                return circuit, params, index_to_symbols
            else:
                return circuit, params

        #### add rz #####
        elif 0 <= ind - self.number_of_cnots  < self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            for par, gate in zip(range(1),[cirq.rz]):
                new_param = "th_"+str(len(params))
                params.append(new_param)

                #adding noise
                ng = np.random.choice(self.channel_unitaries,1, p=self.channel_params)[0]
                circuit.append(ng.on(qubit))

                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_param
                return circuit, params, index_to_symbols

        #### add rx #####
        elif self.n_qubits <= ind - self.number_of_cnots  < 2*self.n_qubits:
            qubit = self.qubits[(ind-self.number_of_cnots)%self.n_qubits]
            for par, gate in zip(range(1),[cirq.rx]):
                new_param = "th_"+str(len(params))
                params.append(new_param)
                #adding noise
                ng = np.random.choice(self.channel_unitaries,1, p=self.channel_params)[0]
                circuit.append(ng.on(qubit))

                circuit.append(gate(sympy.Symbol(new_param)).on(qubit))
                index_to_symbols[len(list(index_to_symbols.keys()))] = new_param
            return circuit, params, index_to_symbols

    def give_circuit(self, lista):
        """
        retrieves circuit (cirq object), with a list of each continuous parameter (symbols) and dictionary index_to_symbols giving the symbols for each position (useful for simplifier)

        lista: list of integers in [0, 2*n + n(n-1)), with n = self.number_qubits. Each integer describes a possible unitary (CNOT, rx, rz)
        """
        circuit, symbols, index_to_symbols = [], [], {}
        for k in lista:
            circuit, symbols, index_to_symbols = self.append_to_circuit(k,circuit,symbols, index_to_symbols)
        circuit = cirq.Circuit(circuit)
        return circuit, symbols, index_to_symbols

    def give_inverse(self, indexed_circuit, resolver):
        """
        computes inverse of circuit, just go reverse and put a minus on rotations ;)
        """
        res_in = {}
        for sym_name, value in zip(list(resolver.keys()), list(resolver.values())[::-1]):
            res_in[sym_name] = -value
        unitary = cirq.resolve_parameters(self.give_circuit(indexed_circuit[::-1])[0], res_in)
        return indexed_circuit[::-1], res_in, unitary


    def give_circuit_with_noise(self, lista):
        """
        noisy version of self.give_circuit(list).
        It retrieves a batch of circuits, each one deterministically evolved under the channel (assuming the channel applyies unitary trasnformations with a given probability, given by self.channel_params).

        retrieves (batch of) circuits (cirq object), of len(self.q_batch_size), with a list of each continuous parameter (symbols) and dictionary index_to_symbols giving the symbols for each position (useful for simplifier).

        lista: list of integers in [0, 2*n + n(n-1)), with n = self.number_qubits. Each integer describes a possible unitary (CNOT, rx, rz).
        """
        qbatch=[]
        for cind in range(self.q_batch_size):
            circuit, symbols, index_to_symbols = [], [], {}
            for k in lista:
                circuit, symbols, index_to_symbols = self.append_to_circuit_with_noise(k,circuit,symbols, index_to_symbols)
            circuit = cirq.Circuit(circuit)
            qbatch.append(circuit)
        return qbatch, symbols, index_to_symbols


    def give_unitary(self,idx, res):
        """
        a shortcut to resolve parameters.

        idx: sequence of integers encoding the gates
        res: parameters dictionary
        """
        return cirq.resolve_parameters(self.give_circuit(idx)[0], res)

    def give_qubit(self, ind):
        """
        returns a list of qubits affected by gate indexed via ind
        used for cirq.insert_batch in the noise
        """
        if ind < self.number_of_cnots:
            return self.indexed_cnots[str(ind)]
        else:
            return [(ind-self.number_of_cnots)%self.n_qubits]


    def hea_layer(self,full=False, x=True,count=0):
        layer = []
        for ind in range(0,self.n_qubits):
            if x == True:
                layer.append(self.number_of_cnots + ind+ (self.n_qubits))
            else:
                layer.append(self.number_of_cnots + ind+ (2*self.n_qubits))
            layer.append(self.number_of_cnots + ind)
            if full:
                if x==True:
                    layer.append(self.number_of_cnots + ind+ (self.n_qubits))
                else:
                    layer.append(self.number_of_cnots + ind+ (2*self.n_qubits))
        if count%2==0:
            for ind in range(0,self.n_qubits-1,2):
                layer.append(self.cnots_index[str([ind, (ind+1)%self.n_qubits])])
        else:
            for ind in range(1,self.n_qubits,2):
                layer.append(self.cnots_index[str([ind, (ind+1)%self.n_qubits])])
        return layer

    def hea_ansatz_indexed_circuit(self, L=2, full=False):
        indexed_circuit=[]
        for l in range(L):
            indexed_circuit+=self.hea_layer(full=full, count=l)
        return indexed_circuit

    def count_cnots(self, indexed_circuit):
        cncount=0
        for k in indexed_circuit:
            if k < self.number_of_cnots:
                cncount+=1
        return cncount

    def count_params(self, indexed_circuit):
        cncount=0
        for k in indexed_circuit:
            if k >= self.number_of_cnots:
                cncount+=1
        return cncount

    def create_hea_w_params(self, nparams=10):
        indexed_circuit=[]
        s=0
        while self.count_params(indexed_circuit)<nparams:
            for ind in range(self.n_qubits)[::2]:
                if self.count_params(indexed_circuit)<nparams:
                    indexed_circuit.append(self.number_of_cnots + (ind+s)%self.n_qubits+ (self.n_qubits))
                    indexed_circuit.append(self.number_of_cnots + (ind+s)%self.n_qubits)
                    indexed_circuit.append(self.number_of_cnots + (ind+1+s)%self.n_qubits +(self.n_qubits))
                    indexed_circuit.append(self.number_of_cnots +(ind+1+s)%self.n_qubits)
                    indexed_circuit.append(self.cnots_index[str([(ind+s)%self.n_qubits, (ind+1+s)%self.n_qubits])])
                else:
                    break
            s+=1
        return indexed_circuit



    def create_hea_w_cnots(self, nconts=10):
        indexed_circuit=[]
        s=0
        while self.count_cnots(indexed_circuit)<nconts:
            for ind in range(self.n_qubits)[::2]:
                if self.count_cnots(indexed_circuit)<nconts:
                    indexed_circuit.append(self.number_of_cnots + (ind+s)%self.n_qubits+ (self.n_qubits))
                    indexed_circuit.append(self.number_of_cnots + (ind+s)%self.n_qubits)
                    indexed_circuit.append(self.number_of_cnots + (ind+1+s)%self.n_qubits +(self.n_qubits))
                    indexed_circuit.append(self.number_of_cnots +(ind+1+s)%self.n_qubits)
                    indexed_circuit.append(self.cnots_index[str([(ind+s)%self.n_qubits, (ind+1+s)%self.n_qubits])])
                else:
                    break
            s+=1
        return indexed_circuit


    def gate_counter_on_qubits(self, indexed_circuit):
        ngates = {k:[0,0] for k in range(len(self.qubits))}
        for ind in indexed_circuit:
            if ind < self.number_of_cnots:
                control, target = self.indexed_cnots[str(ind)]
                ngates[control][1]+=1
                ngates[target][1]+=1
            else:
                qind = (ind-self.number_of_cnots)%self.n_qubits
                ngates[qind][0]+=1
        return ngates

    def compute_ground_energy(self):
        ground_energy = np.min(np.linalg.eigvals(sum(self.observable).matrix()))
        return np.real(ground_energy)

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            print("hey")
            np.seed(datetime.now().microsecond + datetime.now().second)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    return decorator
