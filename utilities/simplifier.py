import numpy as np
from utilities.circuit_database import CirqTranslater
from utilities.templates import *
from utilities.misc import get_qubits_involved, reindex_symbol, shift_symbols_down, type_get, check_rot
from utilities.variational import Minimizer
from utilities.compiling import *


class Simplifier:
    """
    untouchable::: list of blocks which simplifier should not toch (for instance environment blocks), state preparation blocks, fixed measurement blocks, etc.


    rule_1: CNOT when control is |0> == identity.    RELATIVE
    rule_2: Two same CNOTS -> identity (delete both).   ABSOLUTE
    rule_3: Rz (|0>) ---> kill (only adds a phase killed tracing over, in case we are computing).   RELATIVE
    rule_4:  Repeated rotations: add the values.    ABSOLUTE
    rule_5: compile 1-qubit gates into euler rotations.  ABSOLUTE
    rule_6: move cnots to the left, rotations to the right ABSOLUTE

    TO DO:
            check if doing more than one loop of the rules help (in general it should, a stopping condition should be written)

    """

    def __init__(self,translator, untouchable=[], **kwargs):
        self.translator = translator
        self.max_cnt = kwargs.get('max_cnt',500)
        self.absolute_rules = [self.rule_2, self.rule_4,  self.rule_5, self.rule_6]# this are rules that should be applied to any block ,regardless of its position ..
        self.relative_rules = [self.rule_1, self.rule_3] ##rule 1 and 3 are not always applied since the incoming state is likely not |0> for general block_id's. (suppose you have a channel...)
        self.loop_the_rules = 1 ### one could think on looping more than ones the rule
        self.apply_relatives_to_first = False
        self.untouchable = untouchable ### for instance, channel blocks...

    def reduce_circuit(self, circuit_db):
        simplified_db = circuit_db.copy()
        blocked_circuit = {}
        nsimps=0
        for block in set(circuit_db["block_id"]):
            final_cnt = 0
            if (block in self.untouchable) == False: #### only simplify those blocks which we have control of!
                for routine_check in range(self.loop_the_rules):
                    blocked_circuit[block] = simplified_db[simplified_db["block_id"] == block]
                    for rule in self.absolute_rules:
                        cnt, blocked_circuit[block]  = self.apply_rule(blocked_circuit[block]  , rule)
                        final_cnt += cnt
                    if (block == 0) and (self.apply_relatives_to_first == True):
                        for rule in self.relative_rules:
                            cnt, blocked_circuit[block]  = self.apply_rule(blocked_circuit[block]  , rule)
                            final_cnt += cnt
                    nsimps +=final_cnt
                    if final_cnt < 2:
                        break
                    final_cnt = 0
        simplified_db = concatenate_dbs([sb for sb in blocked_circuit.values()])
        simplified_db = self.order_symbols(simplified_db) ## this is because from block to block there might be a gap in the symbols order!
        return simplified_db, nsimps

    def apply_rule(self, original_circuit_db, rule, **kwargs):
        simplified, cnt = True, 0
        original_circuit, original_circuit_db = self.translator.give_circuit(original_circuit_db)
        gates_on_qubit, on_qubit_order = self.get_positional_dbs(original_circuit, original_circuit_db)
        simplified_db = original_circuit_db.copy()
        while simplified and cnt < self.max_cnt:
            simplified, simplified_circuit_db = rule(simplified_db, on_qubit_order, gates_on_qubit)
            if simplified is True:
                print(rule)
            circuit, simplified_db = self.translator.give_circuit(simplified_circuit_db)
            gates_on_qubit, on_qubit_order = self.get_positional_dbs(circuit, simplified_db)
            cnt+=1
            if cnt>100:
                print("hey, i'm still simplifying, cnt{}".format(cnt))
        return cnt, simplified_db


    def get_positional_dbs(self, circuit, circuit_db):

        qubits_involved = get_qubits_involved(circuit, circuit_db)

        gates_on_qubit = {q:[] for q in qubits_involved}
        on_qubit_order = {q:[] for q in qubits_involved}

        for order_gate, ind_gate in enumerate( circuit_db["ind"]):
            if ind_gate < self.translator.number_of_cnots:
                control, target = self.translator.indexed_cnots[str(ind_gate)]
                gates_on_qubit[control].append(ind_gate)
                gates_on_qubit[target].append(ind_gate)
                on_qubit_order[control].append(order_gate)
                on_qubit_order[target].append(order_gate)
            else:
                gates_on_qubit[(ind_gate-self.translator.n_qubits)%self.translator.n_qubits].append(ind_gate)
                on_qubit_order[(ind_gate-self.translator.n_qubits)%self.translator.n_qubits].append(order_gate)
        return gates_on_qubit, on_qubit_order

    def order_symbols(self, simplified_db):
        shift_need = True
        ssdb = simplified_db.copy()
        while shift_need is True:
            ss = ssdb["symbol"].dropna()
            prev_s = int(list(ss)[0].replace("th_",""))
            for ind,s in zip(ss.index[1:], ss[1:]):
                current = int(s.replace("th_",""))
                if current - prev_s >1:
                    shift_need = True
                    from_ind = ind
                    ssdb = shift_symbols_down(self.translator, from_ind, ssdb)
                    break
                else:
                    shift_need = False
                    prev_s = current
        return ssdb

    def rule_1(self, simplified_db, on_qubit_order, gates_on_qubit):
        """
        CNOT when control is |0> == identity.    RELATIVE
        """
        simplification = False

        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path):
                if ind_gate < self.translator.number_of_cnots:
                    control, target = self.translator.indexed_cnots[str(ind_gate)]
                    if (q == control) and (order_gate_on_qubit == 0):
                        pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit]

                        block_id = circuit_db.loc[pos_gate_to_drop]["block_id"]
                        simplified_db.loc[int(pos_gate_to_drop)+0.1] = gate_template(self.translator.number_of_cnots + self.translator.n_qubits + control, param_value=0.0, block_id=circuit_db.loc[0]["block_id"])
                        simplified_db.loc[int(pos_gate_to_drop)+0.11] = gate_template(self.translator.number_of_cnots + self.translator.n_qubits + target, param_value=0.0, block_id=circuit_db.loc[0]["block_id"])

                        simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)

                        simplification = True
                        break
        simplified_db = simplified_db.sort_index().reset_index(drop=True)
        return simplification, simplified_db


    def rule_2(self, simplified_db, on_qubit_order, gates_on_qubit):
        """
        Two same CNOTS -> identity (delete both).   ABSOLUTE
        """
        simplification = False

        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-1]):

                next_ind_gate = qubit_gates_path[order_gate_on_qubit+1]
                if (ind_gate < self.translator.number_of_cnots) and (ind_gate == next_ind_gate):
                    control, target = self.translator.indexed_cnots[str(ind_gate)]
                    not_gates_in_between = False
                    this_qubit = q
                    other_qubits = [control, target]
                    other_qubits.remove(q)
                    other_qubit = other_qubits[0]

                    ## now we need to check what happens in the other_qubit
                    for qord_other, ind_gate_other in enumerate(gates_on_qubit[other_qubit][:-1]):
                        if (ind_gate_other == ind_gate) and (gates_on_qubit[other_qubit][qord_other +1] == ind_gate):
                            ## if we append the CNOT for q and other_q on the same call, and also for the consecutive
                            ## note that in between there can be other calls for other qubits
                            order_call_q = on_qubit_order[q][order_gate_on_qubit]
                            order_call_other_q = on_qubit_order[other_qubit][qord_other]

                            order_call_qP1 = on_qubit_order[q][order_gate_on_qubit+1]
                            order_call_other_qP1 = on_qubit_order[other_qubit][qord_other+1]

                            ## then it's kosher to say they are consecutively applied (if only looking at the two qubits)
                            if (order_call_q == order_call_other_q) and (order_call_qP1 == order_call_other_qP1):

                                pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit]
                                simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)
                                pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit+1]
                                simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)

                                simplification = True
                                break
                    if simplification is True:
                        break
        simplified_db = simplified_db.reset_index(drop=True)
        return simplification, simplified_db



    def rule_3(self, simplified_db, on_qubit_order, gates_on_qubit):
        """
        Rz (|0>) ---> kill (only adds a phase killed tracing over, in case we are computing).   RELATIVE
        """
        simplification = False
        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-1]):
                if order_gate_on_qubit == 0 and (self.translator.number_of_cnots <= ind_gate< self.translator.number_of_cnots+ self.translator.n_qubits ):
                    pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit]
                    simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)
                    simplified_db = simplified_db.reset_index(drop=True)
                    simplified_db = shift_symbols_down(self.translator, pos_gate_to_drop, simplified_db)
                    simplification = True
                    break
        return simplification, simplified_db



    def rule_4(self, simplified_db, on_qubit_order, gates_on_qubit):
        """
        Repeated rotations: add the values.    ABSOLUTE
        """
        simplification = False
        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-1]):
                if ind_gate>=self.translator.number_of_cnots:
                    next_ind_gate = qubit_gates_path[order_gate_on_qubit+1]
                    if next_ind_gate == ind_gate:
                        pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit]
                        pos_gate_to_add = on_qubit_order[q][order_gate_on_qubit+1]

                        value_1 = simplified_db.loc[pos_gate_to_drop]["param_value"]
                        value_2 = simplified_db.loc[pos_gate_to_add]["param_value"]

                        simplified_db.loc[pos_gate_to_add] = simplified_db.loc[pos_gate_to_add].replace(to_replace=value_2, value=value_1 + value_2)
                        simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)
                        simplified_db = simplified_db.reset_index(drop=True)

                        simplified_db = shift_symbols_down(self.translator, pos_gate_to_drop, simplified_db)
                        simplification = True
                        break
        return simplification, simplified_db




    def rule_5(self, simplified_db, on_qubit_order, gates_on_qubit):
        """
        compile 1-qubit gates into euler rotations.  ABSOLUTE
        """
        simplification = False
        original_db = simplified_db.copy() #this is not the most efficient thing, just a patch
        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-2]):

                if simplification is True:
                    break
                ind_gate_p1 = qubit_gates_path[order_gate_on_qubit+1]
                ind_gate_p2 = qubit_gates_path[order_gate_on_qubit+2]

                if (check_rot(ind_gate, self.translator) == True) and (check_rot(ind_gate_p1, self.translator) == True) and (check_rot(ind_gate_p2, self.translator) == True):

                    type_0 = type_get(ind_gate,self.translator)
                    type_1 = type_get(ind_gate_p1,self.translator)
                    type_2 = type_get(ind_gate_p2,self.translator)

                    if type_0 == type_2:
                        types = [type_0, type_1, type_2]
                        for next_order_gate_on_qubit, ind_gate_next in enumerate(qubit_gates_path[order_gate_on_qubit+3:]):
                            if (check_rot(ind_gate_next, self.translator) == True):# and (next_order_gate_on_qubit < len(qubit_gates_path[order_gate_on_qubit+3:])):
                                types.append(type_get(ind_gate_next, self.translator))
                                simplification=True
                            else:
                                break
                        if simplification == True:
                            indices_to_compile = [on_qubit_order[q][order_gate_on_qubit+k] for k in range(len(types))]
                            self.translator_ = CirqTranslater(n_qubits=2)
                            u_to_compile_db = simplified_db.loc[indices_to_compile]
                            u_to_compile_db["ind"] = self.translator_.n_qubits*type_get(u_to_compile_db["ind"], self.translator) + self.translator_.number_of_cnots
                            u_to_compile_db["symbol"] = None ##just to be sure it makes no interference with the compiler...

                            compile_circuit, compile_circuit_db = construct_compiling_circuit(self.translator_, u_to_compile_db)
                            minimizer = Minimizer(self.translator_, mode="compiling", hamiltonian="Z")

                            cost, resolver, history = minimizer.minimize([compile_circuit], symbols=self.translator.get_trainable_symbols(compile_circuit_db))

                            OneQbit_translator = CirqTranslater(n_qubits=1)
                            u1s = u1_db(OneQbit_translator, 0, params=True)
                            u1s["param_value"] = -np.array(list(resolver.values()))
                            resu_comp, resu_db = OneQbit_translator.give_circuit(u1s,unresolved=False)


                            u_to_compile_db_1q = u_to_compile_db.copy()
                            u_to_compile_db_1q["ind"] = u_to_compile_db["ind"] = type_get(u_to_compile_db["ind"], self.translator_)

                            cc, cdb = OneQbit_translator.give_circuit(u_to_compile_db_1q, unresolved=False)
                            c = cc.unitary()
                            r = resu_comp.unitary()

                            ## phase_shift if necessary
                            if np.abs(np.mean(c/r) -1) > 1:
                                u1s.loc[0] = u1s.loc[0].replace(to_replace=u1s["param_value"][0], value=u1s["param_value"][0] + 2*np.pi)# Rz(\th) = e^{-ii \theta \sigma_z / 2}c0, cdb0 = self.translator.give_circuit(pd.DataFrame([gate_template(0, param_value=2*np.pi)]), unresolved=False)
                            resu_comp, resu_db = self.translator.give_circuit(u1s,unresolved=False)

                            first_symbols = simplified_db["symbol"][indices_to_compile][:3]

                            for new_ind, typ, pval in zip(indices_to_compile[:3],[0,1,0], list(u1s["param_value"])):
                                simplified_db.loc[new_ind+0.1] = gate_template(self.translator.number_of_cnots + q + typ*self.translator.n_qubits,
                                                                                 param_value=pval, block_id=simplified_db.loc[new_ind]["block_id"],
                                                                                 trainable=True, symbol=first_symbols[new_ind])

                            for old_inds in indices_to_compile:
                                simplified_db = simplified_db.drop(labels=[old_inds],axis=0)#

                            simplified_db = simplified_db.sort_index().reset_index(drop=True)
                            killed_indices = indices_to_compile[3:]
                            db_follows = original_db[original_db.index>indices_to_compile[-1]]

                            if len(db_follows)>0:
                                gates_to_lower = list(db_follows.index)
                                number_of_shifts = len(killed_indices)
                                for k in range(number_of_shifts):
                                    simplified_db = shift_symbols_down(self.translator, gates_to_lower[0]-number_of_shifts, simplified_db)

        return simplification, simplified_db



    def rule_6(self, simplified_db, on_qubit_order, gates_on_qubit):
        """
        move cnots to the left, rotations to the right.

        IMPORTANT this won't work if the cirucit is too short!
        """


        simplification = False
        for q, qubit_gates_path in gates_on_qubit.items():
            if simplification is True:
                break
            for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-1]):
                if simplification is True:
                    break

                ind_gate_p1 = qubit_gates_path[order_gate_on_qubit+1]
                print(ind_gate_p1, qubit_gates_path, order_gate_on_qubit)

                if (check_rot(ind_gate, self.translator) == True) and (check_rot(ind_gate_p1, self.translator) == False):
                    type_0 = type_get(ind_gate, self.translator)

                    control, target = self.translator.indexed_cnots[str(ind_gate_p1)]

                    this_qubit = q
                    other_qubits = [control, target]
                    other_qubits.remove(q)
                    other_qubit = other_qubits[0]

                    if ((type_0 == 0) and (q==control)) or ((type_0== 1) and (q==target)):
                        ### free to pass...
                        if len(gates_on_qubit[other_qubit]) == 1:
                            simplification = True
                        for qord_other, ind_gate_other in enumerate(gates_on_qubit[other_qubit]):
                            if (ind_gate_other == ind_gate_p1): ## check if we find the same cnot on both qubits
                                cnot_call__q = on_qubit_order[q][order_gate_on_qubit+1]
                                if cnot_call__q == on_qubit_order[other_qubit][qord_other]:## now check if we are applying the gate on both qubits at same time
                                    ### it might happen that there's no gate on the other qbit before the cnot, in that case free to comute.
                                    if qord_other == 0:
                                        simplification = True
                                        break
                                    else:
                                        gate_in_other_qubit_before_cnot = simplified_db.loc[on_qubit_order[other_qubit][qord_other-1]]["ind"]
                                        if check_rot(gate_in_other_qubit_before_cnot, self.translator) == True:
                                            type_gate_other = type_get(gate_in_other_qubit_before_cnot, self.translator)
                                            if type_0 != type_gate_other:
                                                simplification = True
                                                break
                if simplification == True:
                    if len(on_qubit_order[q]) <2:
                        simplification=False
                    else:
                        info_rot = simplified_db.loc[on_qubit_order[q][order_gate_on_qubit]].copy()
                        info_cnot_control = simplified_db.loc[on_qubit_order[q][order_gate_on_qubit+1]].copy()

                        simplified_db.loc[on_qubit_order[q][order_gate_on_qubit]]  = info_cnot_control
                        simplified_db.loc[on_qubit_order[q][order_gate_on_qubit+1]] = info_rot
        return simplification, simplified_db




### saving this just in case
# shift_need = True
# ssdb = simplified_db.copy()
# while shift_need is True:
#     ss = ssdb["symbol"].dropna()
#     prev_s = int(list(ss)[0].replace("th_",""))
#     for ind,s in zip(ss.index[1:], ss[1:]):
#         current = int(s.replace("th_",""))
#         if current - prev_s >1:
#             print(current, prev_s)
#             shift_need = True
#             from_ind = ind
#             ssdb = shift_symbols_down(simplifier.translator, from_ind, ssdb)
#             break
#         else:
#             shift_need = False
#             prev_s = current
