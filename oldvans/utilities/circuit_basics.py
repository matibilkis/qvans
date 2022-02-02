import cirq
import numpy as np
import sympy
import os
import os
import pandas as pd


class Basic:
    def __init__(self, n_qubits=3):
        """
        n_qubits: number of qubits on your circuit (+ environment)
        """
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.env_qbits = [] ### this is used to identify the qubits that act as environment.

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


    def append_to_circuit(self, gate_id, circuit, symbols, circuit_db):
        """
        gate_id: id of gate at dictionary; 0<=gte_id<self.number_of_cnots --> gives self.indexed_cnots[gate_id]
        self.number_of_cnots <= gate_id < 3*number_qubits retrives rotation (rz, rx, ry)
        3*n_qubits < =gate_id < 4*n_qubits retrieves Haddamard

        circuit: cirq object that describes the quantum circuit
        circuit_db: pandas database describing the ciruit. Contains symbols, params, trainable... (see give_gate_template method)
        """
        ind = gate_id["ind"]
        gate_index = len(list(circuit_db.keys()))
        circuit_db[gate_index] = gate_id
        #### add CNOT
        if ind < self.number_of_cnots:
            control, target = self.indexed_cnots[str(ind)]
            circuit.append(cirq.CNOT.on(self.qubits[control], self.qubits[target]))
            circuit_db[gate_index]["symbol"] = None
            return circuit, symbols, circuit_db
        #### add rotation
        elif self.number_of_cnots <= ind < self.number_of_cnots + 3*self.n_qubits:
            gate_type_index = (ind - self.number_of_cnots)//self.n_qubits
            qubit  = (ind-self.number_of_cnots)%self.n_qubits
            gate = self.cgates[gate_type_index]

            if (self.check_None(gate_id["param_value"]) is not True): ### PARAM PROVIDED
                if (gate_id["symbol"]) is None:  #NO SYMBOL
                    symbol_name = "th_"+str(len(symbols))
                else:
                    symbol_name = gate_id["symbol"]
                param_value = gate_id["param_value"]
            else: ### NO PARAM PROVIDED
                if (gate_id["symbol"]) is None: #NO SYMBOL
                    symbol_name = "th_"+str(len(symbols))
                else: #SYMBOL
                    symbol_name = gate_id["symbol"]
                param_value = sympy.Symbol(symbol_name)
            symbols.append(symbol_name)
            circuit_db[gate_index]["symbol"] = symbol_name
            circuit.append(gate(param_value).on(self.qubits[qubit]))
            return circuit, symbols, circuit_db
        ### add HADDAMARD
        elif self.number_of_cnots + 3*self.n_qubits <= ind < self.number_of_cnots + 4*self.n_qubits:
            qubit  = (ind-self.number_of_cnots)%self.n_qubits
            circuit.append(cirq.H.on(self.qubits[qubit]))
            circuit_db[gate_index]["symbol"] = None
            return circuit, symbols, circuit_db
        else:
            raise AttributeError("Wrong index!")


    def give_circuit(self, lista):
        """
        retrieves circuit (cirq object), with a list of each continuous parameter (symbols) and dictionary index_to_symbols giving the symbols for each position (useful for simplifier)

        lista: list of integers in [0, 2*n + n(n-1)), with n = self.number_qubits. Each integer describes a possible unitary (CNOT, rx, rz)
        """
        circuit, symbols, circuit_db = [], [], {}
        for k in lista:
            circuit, symbols, circuit_db = self.append_to_circuit(k,circuit,symbols, circuit_db)
        circuit = cirq.Circuit(circuit)
        circuit_db = pd.DataFrame.from_dict(circuit_db,orient="index")
        return circuit, symbols, circuit_db

    def give_channel_circuit_db(self, qubits_ind, eta, block_id=1, entire_circuit=False):
        """
        # AMPLITUDE DAMPING CHANNEL (TODO scpecify for ANY channel compiled into a cirucit)

        qubits_ind: list of indices of the qubits ---> [system, ancilla]
        block_id: number that VANs uses to identify the channel and not touch it.
        eta: damping strength (note that the rotation is twice the value!)
        """
        channel = []
        if not hasattr(self, "env_qubits"):
            self.env_qubits = []
        self.env_qubits.append(qubits_ind[1])
        ## controlled-Ry(2*eta)

        ### H on each qubit
        for qindex in qubits_ind: ##list on qubits suffering from the channel
            channel.append( self.number_of_cnots + 3*self.n_qubits + qindex )
        ###CNOT[q1,q0]
        channel.append(self.cnots_index[str(qubits_ind[::-1])])
        ###Ry(eta)(q1)
        channel.append(self.number_of_cnots + 2*self.n_qubits + qubits_ind[1] )
        ###CNOT[q1,q0]
        channel.append(self.cnots_index[str(qubits_ind[::-1])])
        ### H on eahc qubit
        for qindex in qubits_ind:
            channel.append( self.number_of_cnots + 3*self.n_qubits + qindex )
        ### Ry(2*eta)(q1)
        channel.append(self.number_of_cnots + 2*self.n_qubits + qubits_ind[1] )


        ## CNOT[q1,q0]
        channel.append(self.cnots_index[str(qubits_ind[::-1])])

        index_eta = self.number_of_cnots + 2*self.n_qubits + qubits_ind[1]
        give_value_eta = lambda x, eta_value: None if x!= index_eta else eta_value
        is_rotation = lambda x: True if self.number_of_cnots<=x<self.number_of_cnots+(3*self.n_qubits) else None
        return [self.give_gate_template(gate_id, block_id=block_id, trainable=False, param_value=give_value_eta(gate_id, eta), channel_param=is_rotation(gate_id)) for gate_id in channel]


    def give_gate_template(self, ind,**kwargs):
        """
        Creates a dictionary (that is later converted into pandas dataframe) describing rows of gate associated to ind. Used as initialization shortcut.

        gate_id: {"param_value":None, "trainable":True, "block_id":0, "movable":True}
        """
        dicti = {"ind": ind}
        dicti["param_value"] = self.what_if_none(kwargs.get("param_value"),None)
        dicti["trainable"] = self.what_if_none(kwargs.get("trainable"),True)
        dicti["block_id"] = self.what_if_none(kwargs.get("block_id"))
        dicti["symbol"] = self.what_if_none(kwargs.get("symbol"))
        dicti["channel_param"] = self.what_if_none(kwargs.get("channel_param"))
        return dicti


    def what_if_none(self,x,alternative=None):
        """
        aid method for give_gate_template
        """
        if x is None:
            return alternative
        else:
            try:
                pp = np.isnan(x)
            except TypeError:
                pp=False
            if pp == False:
                return x
            else:
                return alternative


    def give_circuit_from_list(self, list_of_indices):
        return self.give_circuit([self.give_gate_template(k) for k in list_of_indices])

    def check_None(self,x):
        if (x is None) or (np.isnan(x) is True):
            return True
        else:
            return False

   ###### database  ######
    def give_circuit_from_db(self, dd):
        return self.give_circuit([self.give_gate_template(**dict(dd.iloc[k])) for k in range(len(dd))])

    def give_symbol_to_value(self, circuit_db):
        convert_to_list_if_none = lambda x: [] if x is None else x
        symbol_to_value = {circuit_db.loc[k]["symbol"]: convert_to_list_if_none(circuit_db.loc[k]["param_value"]) for k in circuit_db["ind"].index}
        return symbol_to_value

    def give_index_to_symbols(self, circuit_db):
        index_to_symbols = {k: circuit_db.loc[k]["symbol"] for k in circuit_db["symbol"].dropna().index}
        return index_to_symbols


    #### channel discrimination ###s
    def prepare_channel_discrimination_circuits(self, circuit_db, other_eta):
        """
        method used for channel discrimination
        """
        batch=[self.give_circuit_from_db(circuit_db)]
        batch.append(self.give_copy_of_circuit_with_other_channel_params(circuit_db, other_eta))
        return batch

    def give_copy_of_circuit_with_other_channel_params(self, circuit_db,new_eta):
        """
        method used for channel discrimination
        """
        new_channel_params = [new_eta]*2
        circuit_db1 = circuit_db.copy()
        channel_params = circuit_db["param_value"][circuit_db["channel_param"]==True]
        channel_params.update({i: val for i,val in zip(channel_params.index, new_channel_params)})
        circuit_db1.update(channel_params)
        return self.give_circuit_from_db(circuit_db1)

    def update_circuit_db_param_values(self, circuit_db,symbol_to_value):
        symbol_db = circuit_db.loc[circuit_db["symbol"].isin(list(symbol_to_value))]["symbol"]
        circuit_db["param_value"].update({ind:val for ind,val in zip(symbol_db.index, symbol_to_value.values())})
        return circuit_db

    #### simplifier & unitary killer ####
    def give_trainable_blocks(self, circuit_db):
        return {k: (circuit_db[circuit_db.block_id == k]["trainable"] == True).unique()[0] for k in circuit_db["block_id"].unique()}

    def reindex_symbol(self,list_of_symbols, first_symbol_number):
        reindexed=[]
        for ind, sym in enumerate(list_of_symbols):
            if sym == [] or sym is None:
                reindexed.append(None)
            else:
                reindexed.append("th_{}".format(int(sym.replace("th_",""))+first_symbol_number))
        return reindexed

    def give_proper_indices(self,original_db):
        first_symbol = original_db["symbol"].dropna().iloc[0]
        original_ind = original_db.index[0]
        number = first_symbol.replace("th_","")
        return int(number), original_ind

    ##### prepare circuit for TFQ model ####
    def filter_trainable(self,x,y):
        if x is True:
            return None
        else:
            return y

    def hide_trainable_coefficients(self, dd):
        bb = dd.copy()
        bb["param_value"] = bb.apply(lambda x: self.filter_trainable(x.trainable, x.param_value), axis=1)
        return bb

    def prepare_circuit_for_model_from_db(self, dd):
        dd = self.hide_trainable_coefficients(dd)

    def give_symbols_from_db(self,circuit_db_training):
        trainable_symbols = circuit_db_training["symbol"][circuit_db_training["trainable"]==True]
        return list(trainable_symbols.dropna())

    def get_symbols_from_db(self,dd):
         return list(dd["symbol"].dropna())

    ### ANSATZ ####
    def give_inverse(self, indexed_circuit, resolver):
        """
        computes inverse of circuit, just go reverse and put a minus on rotations ;)
        """
        res_in = {}
        for sym_name, value in zip(list(resolver.keys()), list(resolver.values())[::-1]):
            res_in[sym_name] = -value
        unitary = cirq.resolve_parameters(self.give_circuit(indexed_circuit[::-1])[0], res_in)
        return indexed_circuit[::-1], res_in, unitary

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
        """
        if ind < self.number_of_cnots:
            return self.indexed_cnots[str(ind)]
        else:
            return [(ind-self.number_of_cnots)%self.n_qubits]

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

    def gate_counter_on_qubits(self, indexed_circuit):
        """
        Gives gate count for each qbit. First entry rotations, second CNOTS
        """
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

    def rotation_series(self, which=0, exclude=None):
        """
        returns rotations over all qubits (possible excluded, as inticated by exclude list).
        which::: rules the nature of the gate, 0 -> rz  ___  1-> rx ___  2-> ry ____ 3 -> H
            """
        indices=[]
        if not ((isinstance(exlude, list)) or (exlude is None)):
            raise TypeError('exclude should be a list or None')
        for k in range(which*self.n_qubits, (which+1)*self.n_qubits):
            if (k%self.n_qubits) not in exclude:
                indices.append(self.number_of_cnots +k )
        return indices

    def fill_values_with_nan(self,new_indexed_circuit, symbols, param_value):
        k=0
        ns = []
        np = []
        for ind in new_indexed_circuit:
            if (ind<self.number_of_cnots) and (ind < 4*self.n_qubits):
                ns.append(None)
                np.append(None)
            else:
                ns.append(symbols[k])
                np.append(param_value[k])
                k+=1
        return ns, np

    #### pre-defined ansatzes ###
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


    ### some shortcuts ####
    def rz(self, q):
        return self.number_of_cnots  + q
    def rx(self, q):
        return self.number_of_cnots + self.n_qubits + q
    def ry(self, q):
        return self.number_of_cnots + 2*self.n_qubits + q
    def cnot(self, q0, q1):
        return self.cnots_index[str([q0,q1])]
    def u1(self, q):
        return [self.rz(q), self.rx(q), self.rz(q)]
    def u2(self, q0, q1):
        """general two-qubit gate"""
        l=[ u for u in self.u1(q0)]
        for u in self.u1(q1):
            l.append(u)
        l.append(self.cnot(q0,q1))
        l.append(self.rz(q0))
        l.append(self.ry(q1))
        l.append(self.cnot(q1,q0))
        l.append(self.ry(q1))
        l.append(self.cnot(q0,q1))
        for u in self.u1(q0):
            l.append(u)
        for u in self.u1(q1):
            l.append(u)
        return l
