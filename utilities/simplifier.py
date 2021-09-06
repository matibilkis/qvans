from utilities.circuit_basics import Basic, timeout
import numpy as np
import cirq
import sympy
from datetime import datetime
import copy
from utilities.qmodels import QNN

class Simplifier(Basic):
    def __init__(self, n_qubits=3,testing=False):
        """
        Simplifies the circuit according to some rules that preserve the expected value of target hamiltornian.
        takes help from index_to_symbol (dict) and symbol_to_value (dict).
        Importantly, it keeps the parameter values of the untouched gates.

        Works on circuits containing CNOTS, Z-rotations and X-rotations.It applies the following rules:

        Rules:  1. CNOT just after initializing, it does nothing (if |0> initialization).
                2. Two consecutive and equal CNOTS compile to identity.
                3. Rotation around z axis of |0> only adds phase hence leaves invariant <H>. It kills it.
                4. two equal rotations: add the values.
                5. Scan for U_3 = Rz Rx Rz, or Rx Rz Rx; if found, abosrb consecutive rz/rx (until a CNOT is found)
                6. Scan qubits and abosrb rotations  (play with control-qubit of CNOT and rz)
                7. Scan qubits and absorb rotations if possible (play with target-qubit of CNOT and rx)
                8. Rz(control) and CNOT(control, target) Rz(control) --> Rz(control) CNOT
                9. Rx(target) and CNOT(control, target) Rx(target) --> Rx(target) CNOT


        Finally, if the circuit becomes too short, for example, there're no gates at a given qubit, an Rx(0) is placed.
        """
        super(Simplifier, self).__init__(n_qubits=n_qubits,testing=testing)
        self.single_qubit_unitaries = {"rx":cirq.rx, "rz":cirq.rz} #Do not touch this ! Important for rule 5
        self.testing = testing


    def simplify_step(self, indexed_circuit, symbol_to_value, index_to_symbols):
        """
        Returns the (simplified) indexed_circuit, index_to_symbols, symbol_to_value.
        """

        self.indexed_circuit = indexed_circuit
        self.symbol_to_value = symbol_to_value
        self.index_to_symbols = index_to_symbols

        connnections, places_gates = self.scan_qubits()

        new_indexed_circuit, NRE, symbols_on = self.simplify_intermediate(connnections, places_gates)
        Sindexed_circuit, Ssymbols_to_values, Sindex_to_symbols = self.translate_to_output(new_indexed_circuit, NRE, symbols_on)
        return Sindexed_circuit, Ssymbols_to_values, Sindex_to_symbols


    def check_qubits_on(self,circuit):
        """function that checks if all qubits are touched by a gate in the circuit"""
        check = True
        effective_qubits = list(circuit.all_qubits())
        for k in self.qubits:
            if k not in effective_qubits:
                check = False
                break
        return check

    def compute_difference_states(self, indices1, resolver1, indices2, resolver2):
        up = np.zeros(2**self.n_qubits)
        up[0] = 1
        return np.dot(cirq.unitary(self.give_unitary(indices1,resolver1)) - cirq.unitary(self.give_unitary(indices2,resolver2)), up )


    def reduce_circuit(self, indexed_circuit, symbol_to_value,index_to_symbols, max_its=None):
        """
        iterate many times simplify circuit, break when no simplifying any more.
        """

        l0 = len(indexed_circuit)
        reducing = True

        if max_its is None:
            max_its = l0

        if self.check_qubits_on(self.give_circuit(indexed_circuit)[0]) is False:
            raise Error("Not all qubits being touched by a rotation! Please is at least that expressible.")

        for its in range(max_its):
            indexed_circuit, symbol_to_value,index_to_symbols = self.simplify_step(indexed_circuit, symbol_to_value,index_to_symbols)
            if len(indexed_circuit) == l0:
                break
        return indexed_circuit, symbol_to_value,index_to_symbols



    def scan_qubits(self):
        """
        this function scans the circuit as described by {self.indexed_circuit}
        and returns a dictionary with the gates acting on each qubit and the order of appearence on the original circuit
        """
        connections={str(q):[] for q in range(self.n_qubits)} #this saves the gates at each qubit. It does not respects the order.
        places_gates = {str(q):[] for q in range(self.n_qubits)} #this saves, for each gate on each qubit, the position in the original indexed_circuit
        flagged = [False]*len(self.indexed_circuit) #used to check if you have seen a cnot already, so not to append it twice to the qubit's dictionary

        for nn,idq in enumerate(self.indexed_circuit): #sweep over all gates in original circuit's list
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
                            places_gates[str(q)].append(nn)
                            flagged[nn] = True
                        elif self.n_qubits <= idq-self.number_of_cnots <  2*self.n_qubits:
                            connections[str(q)].append("rx")
                            places_gates[str(q)].append(nn)
                            flagged[nn] = True #to check that all gates have been flagged
        ####quick testc570c173f199f3f675ab6a33451822d732377fa7
        for k in flagged:
            if k is False:
                raise Error("not all flags in flagged are True!")
        return connections, places_gates


    def rule_1(self, step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit):
        """
        1. if I have a CNOT just after initializing, it does nothing (if |0> initialization).
        """

        modification = False
        q, path, ind,gate = step
        if gate in range(self.number_of_cnots):
            if ind == 0 and (flags_indexed_circuit[places_gates[str(q)][ind]] != 1):
                others = self.indexed_cnots[str(gate)].copy()
                others.remove(int(q)) #the other qubit affected by the CNOT
                control, target = self.indexed_cnots[str(self.indexed_circuit[places_gates[str(q)][ind]])]
                for jind, jgate in enumerate(connections[str(others[0])]): ##Be sure it's the right gate
                    if (int(q) == control) and (jgate == gate) and (places_gates[str(q)][ind] == places_gates[str(others[0])][jind]):
                        new_indexed_circuit[places_gates[str(q)][ind]] = -1
                        flags_indexed_circuit[places_gates[str(q)][ind]] = 1
                        modification = True
                        if self.testing:
                            print("1")
                        break
        return modification, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit

    def rule_2(self, step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit):
        """
        2 consecutive and equal CNOTS compile to identity.
        """

        modification = False
        q, path, ind,gate = step
        c1 = gate in range(self.number_of_cnots)
        c2 = ind<len(path)-1

        if (c1 and c2) == True:
            c3 = flags_indexed_circuit[places_gates[str(q)][ind]] != 1
            c4 = flags_indexed_circuit[places_gates[str(q)][ind+1]] != 1
            c5 = path[ind+1] == gate
            if (c3 and c4 and c5) == True:
                others = copy.deepcopy(self.indexed_cnots[str(gate)])
                others.remove(int(q)) #the other qubit affected by the CNOT
                for jind, jgate in enumerate(connections[str(others[0])][:-1]): ##sweep the other qubit's gates until i find "gate"
                    if (jgate == gate) and (connections[str(others[0])][jind+1] == gate): ##i find the same gate that is repeated in both the original qubit and this one
                        if (places_gates[str(q)][ind] == places_gates[str(others[0])][jind]) and (places_gates[str(q)][ind+1] == places_gates[str(others[0])][jind+1]): #check that positions in the indexed_circuit are the same
                         ###maybe I changed before, so I have repeated in the original but one was shut down..
                            flags_indexed_circuit[places_gates[str(q)][ind]] = 1
                            flags_indexed_circuit[places_gates[str(q)][ind+1]] = 1
                            new_indexed_circuit[places_gates[str(q)][ind]] = -1 ###just kill the repeated CNOTS
                            new_indexed_circuit[places_gates[str(q)][ind+1]] = -1 ###just kill the repeated CNOTS
                            modification=True
                            if self.testing:
                                print("2")
                            break
        return modification, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit




    def rule_3(self, step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit):
        """
         3 Rotation around z axis of |0> does only add a phase, hence leaves invariant <H>. We kill it.
        """

        modification = False
        q, path, ind,gate = step
        # c0 = new_indexed_circuit[places_gates[str(q)][ind]] != -1
        c0 = flags_indexed_circuit[places_gates[str(q)][ind]] != 1
        c1 = (ind == 0)
        c2 = (gate == "rz")

        if (c1 and c2 and c0) == True:
            original_symbol = self.index_to_symbols[places_gates[str(q)][ind]]
            original_value = self.symbol_to_value[original_symbol]

            symbols_to_delete.append(original_symbol)
            new_indexed_circuit[places_gates[str(q)][ind]] = -1
            flags_indexed_circuit[places_gates[str(q)][ind]] = 1
            modification = True
            if self.testing:
                print("3")
        return modification, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit




    def rule_4(self, step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit):
        """
        Repeated rotations: add the values
        """

        modification = False
        q, path, ind,gate = step

        # c0 = new_indexed_circuit[places_gates[str(q)][ind]] != -1
        c0 = flags_indexed_circuit[places_gates[str(q)][ind]] != 1

        c1 = gate in ["rz","rx"]
        c2 = ind < len(path)

        if (c1 and c2 and c0) == True:
            deleted_here=[]
            original_symbol = self.index_to_symbols[places_gates[str(q)][ind]]
            original_value = self.symbol_to_value[original_symbol]
            values_to_add = []
            for ni, next_gates in enumerate(path[ind+1:]):
                if (flags_indexed_circuit[places_gates[str(q)][ind+1+ni]] != 1) and (path[ni+ind+1] == gate):
                    next_symbol = self.index_to_symbols[places_gates[str(q)][ni+ind+1]]
                    symbols_to_delete.append(next_symbol)
                    new_indexed_circuit[places_gates[str(q)][ni+ind+1]] = -1
                    flags_indexed_circuit[places_gates[str(q)][ind+ni+1]] = 1
                    values_to_add.append(self.symbol_to_value[next_symbol])
                    deleted_here.append(next_symbol)
                else:
                    break
            if len(values_to_add)>0:
                modification=True
                sname="th_"+str(len(list(NRE.keys())))
                NRE[sname] = original_value + np.sum(values_to_add)
                symbols_on[str(q)].append(sname)
                if self.testing == True:
                    print("4")
        return modification, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit

    def rule_5(self, step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit):
        """
        Scan for U_3 = Rz Rx Rz, or Rx Rz Rx; if found, abosrb consecutive rz/rx (until a CNOT is found)
        """

        modification = False
        q, path, ind,gate = step
        # c00 = new_indexed_circuit[places_gates[str(q)][ind]] != -1
        c00 = flags_indexed_circuit[places_gates[str(q)][ind]] != 1
        c1 = gate in ["rz","rx"]
        c2 = ind< len(path)-2

        if (c1 and c2 and c00) == True:

            # c01 = new_indexed_circuit[places_gates[str(q)][ind+1]] != -1
            # c02 = new_indexed_circuit[places_gates[str(q)][ind+2]] != -1
            c01 = flags_indexed_circuit[places_gates[str(q)][ind+1]] != 1
            c02 = flags_indexed_circuit[places_gates[str(q)][ind+2]] != 1

            original_symbol = self.index_to_symbols[places_gates[str(q)][ind]]
            original_value = self.symbol_to_value[original_symbol]

            gates = ["rz", "rx"] #which gate am I? Which gate are you?
            gates.remove(gate) #which gate am I? Which gate are you?
            other_gate = gates[0] #which gate are you? Which gate am I?

            if (path[ind+1] == other_gate) and (path[ind+2] == gate) and ((c01 and c02) == True):
                compile_gate = False
                gate_to_compile = [self.single_qubit_unitaries[gate](original_value).on(self.qubits[int(q)])]

                for pp in [1,2]: ##append next 2 gates to the gate_to_compile list (as appearing in the circuit)
                    gate_to_compile.append(self.single_qubit_unitaries[path[ind+pp]](self.symbol_to_value[self.index_to_symbols[places_gates[str(q)][ind+pp]]]).on(self.qubits[int(q)]))

                for ilum, next_gates_to_compile in enumerate(path[(ind+3):]): #Now scan the remaining part of that qubit line
                    if ((next_gates_to_compile in ["rz","rx"]) and (flags_indexed_circuit[places_gates[str(q)][ind+3+ilum]] != 1)) == True:
                        compile_gate = True #we'll compile!
                        new_indexed_circuit[places_gates[str(q)][ind+3+ilum]] = -1
                        flags_indexed_circuit[places_gates[str(q)][ind+3+ilum]] = 1
                        dele = self.index_to_symbols[places_gates[str(q)][ind+3+ilum]]
                        symbols_to_delete.append(dele)
                        gate_to_compile.append(self.single_qubit_unitaries[next_gates_to_compile](self.symbol_to_value[dele]).on(self.qubits[int(q)]))
                    else:
                        break
                if (compile_gate == True): ### if conditions are met s.t. you can absorb everything into U_3:
                    modification = True
                    u = cirq.unitary(cirq.Circuit(gate_to_compile))
                    vals = np.real(self.give_rz_rx_rz(u)[::-1]) #not entirely real since there's a finite number of iterations, we should do this variationally maybe

                    #### make sure this is rz rx rz. This was the trouble of some bug, and why I introduce flags_indexed_circuit
                    new_indexed_circuit[places_gates[str(q)][ind]] = self.number_of_cnots+int(q)
                    new_indexed_circuit[places_gates[str(q)][ind+1]] = self.number_of_cnots+int(q)+self.n_qubits
                    new_indexed_circuit[places_gates[str(q)][ind+2]] = self.number_of_cnots+int(q)

                    flags_indexed_circuit[places_gates[str(q)][ind]] = 1
                    flags_indexed_circuit[places_gates[str(q)][ind+1]] = 1
                    flags_indexed_circuit[places_gates[str(q)][ind+2]] = 1

                    for v in zip(list(vals)):
                        sname="th_"+str(len(list(NRE.keys())))
                        NRE[sname] = v[0]
                        symbols_on[str(q)].append(sname)
                    if self.testing:
                        print("5")
                        # print("qubit",q,"symdel",symbols_to_delete)

        return modification, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit

    def rule_6(self, step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit):
        """
        6. Rz(control) and CNOT(control, target) Rz(control) --> Rz(control) CNOT
        """

        modification = False
        q, path, ind,gate = step
        # c0 = (new_indexed_circuit[places_gates[str(q)][ind]] != -1)
        c0 = flags_indexed_circuit[places_gates[str(q)][ind]] != 1

        c1 = (gate == "rz")
        c2 = (ind< len(path)-2) ### we'll scan the path of that qubit, and if gates can be erased to the right, we will

        if (c0 and c1 and c2) ==True:
            original_symbol = self.index_to_symbols[places_gates[str(q)][ind]]
            original_value = self.symbol_to_value[original_symbol]

            # this means that the gates have not been flaged for erasure
            # c3 = (new_indexed_circuit[places_gates[str(q)][ind+1]] != -1)
            c3 = flags_indexed_circuit[places_gates[str(q)][ind+1]] != 1

            #if path[ind+1] is a CNOT and that CNOT is not marked to be removed.
            if (c3 and (path[ind+1] not in ["rx", "rz"])) == True:
                control, target = self.indexed_cnots[str(path[ind+1])]
                if (int(q) == control): #'cause rx commutes with CNOT, then we'll try to absorb as many gates as possible.
                    values_to_add=[]
                    for npip, pip in enumerate(path[ind+2:]): #path[ind+2:] exists since c2 is True
                        if ( flags_indexed_circuit[places_gates[str(q)][ind+2+npip]] != 1):
                        # new_indexed_circuit[places_gates[str(q)][ind+2+npip]] != -1): #if next element is marked for removal, then we stop everything
                            if pip in ["rx", "rz"]:
                                if pip == "rz":
                                    next_symbol = self.index_to_symbols[places_gates[str(q)][ind+2+npip]]
                                    symbols_to_delete.append(next_symbol)
                                    new_indexed_circuit[places_gates[str(q)][ind+2+npip]] = -1
                                    flags_indexed_circuit[places_gates[str(q)][ind+2+npip]] = 1

                                    values_to_add.append(self.symbol_to_value[next_symbol])
                                    modification = True
                                    if self.testing:
                                        print("6")
                                else:
                                    break
                            else:
                                if self.indexed_cnots[str(pip)][0] == int(q):
                                    pass #go on to see next ones...
                                else:
                                    break #else would mean that either we have a control
                        else:
                            break

                    if len(values_to_add)>0:
                        sname="th_"+str(len(list(NRE.keys()))) ## this is safe, since we are looping on the indices first, and the resolver dict is ordered
                        NRE[sname] = original_value + np.sum(values_to_add)
                        symbols_on[str(q)].append(sname)
        return modification, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit

    def rule_7(self, step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit):
        """
        7. If we get X rotations that can be erased from from a path, because target of CNOT commutes with rx, we merge everything into siingle rotation
        """

        modification = False
        q, path, ind,gate = step
        # c0 = (new_indexed_circuit[places_gates[str(q)][ind]] != -1)
        c0 = flags_indexed_circuit[places_gates[str(q)][ind]] != 1

        c1 = (gate == "rx")
        c2 = (ind< len(path)-2) ### we'll scan the path of that qubit, and if gates can be erased to the right, we will

        if (c0 and c1 and c2) ==True:
            original_symbol = self.index_to_symbols[places_gates[str(q)][ind]]
            original_value = self.symbol_to_value[original_symbol]

            # this means that the gates have not been flaged for erasure
            c3 = flags_indexed_circuit[places_gates[str(q)][ind+1]] != 1

            #if path[ind+1] is a CNOT and that CNOT is not marked to be removed.
            if c3 and (path[ind+1] not in ["rx", "rz"]):
                control, target = self.indexed_cnots[str(path[ind+1])]
                if (int(q) == target): #'cause rx commutes with CNOT, then we'll try to absorb as many gates as possible.
                    values_to_add=[]
                    for npip, pip in enumerate(path[ind+2:]): #path[ind+2:] exists since c2 is True
                        if (flags_indexed_circuit[places_gates[str(q)][ind+2+npip]]!=1):

                        # new_indexed_circuit[places_gates[str(q)][ind+2+npip]] != -1): #if next element is marked for removal, then we stop everything
                            if pip in ["rx", "rz"]:
                                if pip == "rx":
                                    next_symbol = self.index_to_symbols[places_gates[str(q)][ind+2+npip]]
                                    symbols_to_delete.append(next_symbol)
                                    new_indexed_circuit[places_gates[str(q)][ind+2+npip]] = -1
                                    flags_indexed_circuit[places_gates[str(q)][ind+2+npip]] = 1
                                    values_to_add.append(self.symbol_to_value[next_symbol])
                                    modification = True
                                    if self.testing:
                                        print("7")
                                else:
                                    break
                            else:
                                if self.indexed_cnots[str(pip)][1] == int(q):
                                    pass #go on to see next ones...
                                else:
                                    break #else would mean that either we have a control
                        else:
                            break
                        #merge all the values into the first guy.

                    if len(values_to_add)>0:
                        sname="th_"+str(len(list(NRE.keys()))) ## this is safe, since we are looping on the indices first, and the resolver dict is ordered
                        NRE[sname] = original_value + np.sum(values_to_add)
                        symbols_on[str(q)].append(sname)
        return modification, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit


    def rule_8(self, step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit):
        """
        move CNOTs to the left: X -> target-CNOT
        """

        modification = False
        q, path, ind,gate = step
        c0 = flags_indexed_circuit[places_gates[str(q)][ind]] != 1
        c1 = (gate == "rx")
        c2 = (ind< len(path)-1)

        if (c0 and c1 and c2) ==True:
            original_symbol = self.index_to_symbols[places_gates[str(q)][ind]]
            original_value = self.symbol_to_value[original_symbol]
            # this means that the gates have not been flaged for erasure (THE CNOT)
            c3 = flags_indexed_circuit[places_gates[str(q)][ind+1]] != 1
            #if path[ind+1] is a CNOT and that CNOT is not marked to be removed.
            if c3 and (path[ind+1] not in ["rx", "rz"]):
                control, target = self.indexed_cnots[str(path[ind+1])]
                the_other = [control,target]
                the_other.remove(int(q))
                the_other = the_other[0]
                if (int(q) == target): #'cause rx commutes with CNOT, we'll move it to the left

                    rotpos = places_gates[str(q)][ind]
                    cnotpos = places_gates[str(q)][ind+1]
                    #from here it is clear that rotpos < cnotpos. But we can't just swap them, right ?
                    #For instance, suppose there's a gate that doesn't commute with the CNOT, on the other qubit..


                    ## gates in between
                    swap = True
                    for intergate in new_indexed_circuit[rotpos:cnotpos+1]:
                        if intergate == -1:### this is not a big problem, since otherwise we can do this simplifiaciton step later
                            swap = False
                            break
                        else:
                            if intergate < self.number_of_cnots:
                                if q in self.indexed_cnots[str(intergate)]:
                                    swap = False
                                    break
                            if (intergate-self.number_of_cnots)%self.n_qubits == the_other:
                                if (intergate - self.number_of_cnots) >= self.n_qubits: #this means you are an Rx, so you won't commute with the control.
                                    swap = False
                                    break

                    if swap == True:
                        modification = True
                        if self.testing:
                            print("8")
                        sname="th_"+str(len(list(NRE.keys()))) ## this is safe, since we are looping on the indices first, and the resolver dict is ordered
                        NRE[sname] = original_value
                        symbols_on[str(q)].append(sname)

                        original_position_rotation = new_indexed_circuit[places_gates[str(q)][ind]]
                        original_position_cnot = new_indexed_circuit[places_gates[str(q)][ind+1]]
                        new_indexed_circuit[places_gates[str(q)][ind+1]] = original_position_rotation
                        new_indexed_circuit[places_gates[str(q)][ind]] = original_position_cnot

                        flags_indexed_circuit[places_gates[str(q)][ind]] = 1
                        flags_indexed_circuit[places_gates[str(q)][ind+1]] = 1


        return modification, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit



    def rule_9(self, step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit):
        """
        move CNOTs to the left: X -> target-CNOT
        """

        modification = False
        q, path, ind,gate = step
        c0 = flags_indexed_circuit[places_gates[str(q)][ind]] != 1
        c1 = (gate == "rz")
        c2 = (ind< len(path)-1)

        if (c0 and c1 and c2) ==True:
            original_symbol = self.index_to_symbols[places_gates[str(q)][ind]]
            original_value = self.symbol_to_value[original_symbol]
            # this means that the gates have not been flaged for erasure (THE CNOT)
            c3 = flags_indexed_circuit[places_gates[str(q)][ind+1]] != 1
            #if path[ind+1] is a CNOT and that CNOT is not marked to be removed.
            if c3 and (path[ind+1] not in ["rx", "rz"]):
                control, target = self.indexed_cnots[str(path[ind+1])]
                the_other = [control,target]
                the_other.remove(int(q))
                the_other = the_other[0]
                if (int(q) == control): #'cause rx commutes with CNOT, we'll move it to the left

                    rotpos = places_gates[str(q)][ind]
                    cnotpos = places_gates[str(q)][ind+1]
                    #from here it is clear that rotpos < cnotpos. But we can't just swap them, right ?
                    #For instance, suppose there's a gate that doesn't commute with the CNOT, on the other qubit..


                    ## gates in between
                    swap = True
                    for intergate in new_indexed_circuit[rotpos:cnotpos+1]:
                        if intergate == -1:### this is not a big problem, since otherwise we can do this simplifiaciton step later
                            swap = False
                            break
                        else:
                            if intergate < self.number_of_cnots and intergate:
                                if q in self.indexed_cnots[str(intergate)]:
                                    swap = False
                                    break
                            if (intergate-self.number_of_cnots)%self.n_qubits == the_other:
                                if (intergate - self.number_of_cnots) < self.n_qubits: #this means you are an Rz, so you won't commute with the control.
                                    swap = False
                                    break

                    if swap == True:
                        modification = True
                        if self.testing:
                            print("9")
                        sname="th_"+str(len(list(NRE.keys()))) ## this is safe, since we are looping on the indices first, and the resolver dict is ordered
                        NRE[sname] = original_value
                        symbols_on[str(q)].append(sname)

                        original_position_rotation = new_indexed_circuit[places_gates[str(q)][ind]]
                        original_position_cnot = new_indexed_circuit[places_gates[str(q)][ind+1]]
                        new_indexed_circuit[places_gates[str(q)][ind+1]] = original_position_rotation
                        new_indexed_circuit[places_gates[str(q)][ind]] = original_position_cnot

                        flags_indexed_circuit[places_gates[str(q)][ind]] = 1
                        flags_indexed_circuit[places_gates[str(q)][ind+1]] = 1


        return modification, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit



    def rule_handler(self, cnt_rule,  step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit):
        if cnt_rule == 1:
            return self.rule_1(step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit)
        elif cnt_rule == 2:
            return self.rule_2(step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit)
        elif cnt_rule == 3:
            return self.rule_3(step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit)
        elif cnt_rule == 4:
            return self.rule_4(step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit)
        elif cnt_rule == 5:
            return self.rule_5(step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit)
        elif cnt_rule == 6:
            return self.rule_6(step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit)
        elif cnt_rule == 7:
            return self.rule_7(step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit)
        elif cnt_rule == 8:
            return self.rule_8(step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit)
        elif cnt_rule == 9:
            return self.rule_9(step, connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit)
        else:
            print("cnt_rule wrong! not so many rules..", cnt_rule)

    def simplify_intermediate(self, connections, places_gates):
        """
        Scans each qubit, and apply rules listed below to the circuit.

        Rules:  1. CNOT's control is |0>, kill that cnot.
                2. Two consecutive and equal CNOTS compile to identity.
                3. Rotation around z axis of |0> only adds phase hence leaves invariant <H>. Two options: kill it or replace it by Rx (to avoid having no gates)
                4. two equal rotations: add the values.
                5. Scan for U_3 = Rz Rx Rz, or Rx Rz Rx; if found, abosrb consecutive rz/rx (until a CNOT is found)
                6. Scan a line (a qubit), find Rz and if control of CNOT next, try to absorb potential other Z-rotations coming next
                7. Scan a line (a qubit), find Rx and if target of CNOT next, try to absorb potential other X-rotations coming next
                8. Rz(control) and CNOT(control, target) Rz(control) --> Rz(control) CNOT
                9. Rx(target) and CNOT(control, target) Rx(target) --> Rx(target) CNOT
        """
        flags_indexed_circuit = np.zeros(len(self.indexed_circuit)) #intermediate list of gates, the deleted ones are set to -1
        new_indexed_circuit = copy.deepcopy(self.indexed_circuit) #intermediate list of gates, the deleted ones are set to -1
        symbols_to_delete=[] # list to store the symbols that will be deleted/modified
        symbols_on = {str(q):[] for q in list(connections.keys())} #this should give all qubits, otherwise an error will be raised somewhere.
        NRE ={} #NewREsolver
        for q, path in connections.items(): ###sweep over qubits: path is all the gates that act this qubit during the circuit
            for ind,gate in enumerate(path): ### for each qubit, sweep over the list of gates
                modified = False
                cnt_rule = 1
                step = [q, path, ind,gate]
                while (cnt_rule < 10) and (modified==False):
                    modified, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit = self.rule_handler(cnt_rule,  step ,connections, places_gates, new_indexed_circuit, symbols_to_delete, symbols_on, NRE,flags_indexed_circuit)
                    # if modified == True:
                    #     print("cnt_rule",cnt_rule)
                    cnt_rule+=1
                #If no modifications, add that gate to the new_circuit_index and parameter (if any) to NRE, symbols_on
                if (modified == False) and flags_indexed_circuit[places_gates[str(q)][ind]] != 1:
                    if gate in ["rx", "rz"]:
                        original_symbol = self.index_to_symbols[places_gates[str(q)][ind]]
                        original_value = self.symbol_to_value[original_symbol]
                        sname="th_"+str(len(list(NRE.keys())))
                        NRE[sname] = original_value
                        symbols_on[str(q)].append(sname)
                    else:
                        pass
        return new_indexed_circuit, NRE, symbols_on

    def translate_to_output(self, new_indexed_circuit, NRE, symbols_on):
        """
        Final building block for the simplifying function.

        this function takes a processed new_indexed_circuit (with marked gates to deleted to be -1)
        a new resolver NRE with symbols whose corresponding qubit is obtained from symbols_on dict. Symbols_dict contains, for each qubit, the list of symbols acting on it, apppearing in order, when sweeping on the original indexed_circuit. The current function sweeps again over the original indexed_circuit, and marks for each qubit the symbols already used.

        If all gates over a qubit are deleted (since they are not necessary), we add Rx(0), so to avoid problems
        with further continuous optimizations.

        """
        final=[]
        final_idx_to_symbols={}
        final_dict = {}

        index_gate=0

        # print(symbols_on)
        # print("\n")
        # print(new_indexed_circuit)
        # print("\n")
        # print(NRE)

        for gmarked in new_indexed_circuit:
            if not gmarked == -1:
                final.append(gmarked)
                if 0 <= gmarked - self.number_of_cnots < 2*self.n_qubits:
                    ### in which position we add this symbol ?
                    for indd, k in enumerate(symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)]):
                        if k != -1:
                            break
                    final_idx_to_symbols[int(len(final)-1)] = "th_"+str(len(list(final_dict.keys())))
                    final_dict["th_"+str(len(list(final_dict.keys())))] = NRE[symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)][indd]]
                    symbols_on[str((gmarked - self.number_of_cnots)%self.n_qubits)][indd]=-1
                else:
                    final_idx_to_symbols[int(len(final)-1)] = ""

        #### Now we scan again so to check we have not killed all the gates. If so, we add Rx(0)
        self.indexed_circuit = final
        self.symbol_to_value = final_dict
        self.index_to_symbols = final_idx_to_symbols

        connections, places_gates = self.scan_qubits()
        for q, path in connections.items():
            if len(path) == 0:
                final.append(self.number_of_cnots+self.n_qubits+int(q))
                new_symbol = "th_"+str(len(final_dict.keys()))
                final_dict[new_symbol] = 0
                final_idx_to_symbols[len(final)-1] = new_symbol

        return final, final_dict, final_idx_to_symbols


    def rotation(self,vals):
        """
        Rz(\alpha) Rx(\beta) Rz(\gamma)
        with R_n(\theta) = \Exp[ -\ii \vec{theta} \vec{n} /2]
        """

        alpha,beta,gamma = vals
        return np.array([[np.cos(beta/2)*np.cos(alpha/2 + gamma/2) - 1j*np.cos(beta/2)*np.sin(alpha/2 + gamma/2),
                 (-1j)*np.cos(alpha/2 - gamma/2)*np.sin(beta/2) - np.sin(beta/2)*np.sin(alpha/2 - gamma/2)],
                [(-1j)*np.cos(alpha/2 - gamma/2)*np.sin(beta/2) + np.sin(beta/2)*np.sin(alpha/2 - gamma/2),
                 np.cos(beta/2)*np.cos(alpha/2 + gamma/2) + 1j*np.cos(beta/2)*np.sin(alpha/2 + gamma/2)]])

    @timeout(1) #give only one second to solve..
    def symp_solve(self,t,a,b,g):
        return sympy.nsolve(t,[a,b,g],np.pi*np.array([np.random.random(),np.random.random(),np.random.random()]) ,maxsteps=3000, verify=True)

    def give_rz_rx_rz(self,u):
        """
        finds \alpha, \beta \gamma s.t m = Rz(\alpha) Rx(\beta) Rz(\gamma)
        ****
        input: 2x2 unitary matrix as numpy array
        output: [\alpha \beta \gamma]
        """
        a = sympy.Symbol("a")
        b = sympy.Symbol("b")
        g = sympy.Symbol("g")
        st = datetime.now()
        eqs = [sympy.exp(-sympy.I*.5*(a+g))*sympy.cos(.5*b) ,
               -sympy.I*sympy.exp(-sympy.I*.5*(a-g))*sympy.sin(.5*b),
                sympy.exp(sympy.I*.5*(a+g))*sympy.cos(.5*b)
              ]

        kk = np.reshape(u, (4,))
        s=[]
        for i,r in enumerate(kk):
            if i!=2:
                s.append(r)

        t=[]
        for eq, val in zip(eqs,s):
            t.append(eq-val)

        ### this while appears since the seed values may enter
        # in vanishing gradients and through Matrix-zero error.
        #it's not the most elegant method.
        #Furthermore it the sympy solver sometimes gets stucked
        #so we use signal to retrieve error after 1sec.
        error=True
        while error:
            now = (datetime.now()-st).total_seconds()
            try:
                solution = self.symp_solve(t,a,b,g)
                vals = np.array(solution.values()).astype(np.complex64)
                error=False
            except Exception:
                error=True
                if now > 5:
                    np.random.seed(datetime.now().microsecond + datetime.now().second)
                    if now > 100:
                        print("i'm delaying in the rz_rx_rz!! like ",(datetime.now()-st).total_seconds() , " secs")
                        print("\nunitary: ", u)
        return vals
