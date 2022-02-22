def get_positional_dbs(circuit, circuit_db):

    qubits_involved = get_qubits_involved(circuit, circuit_db)

    gates_on_qubit = {q:[] for q in qubits_involved}
    on_qubit_order = {q:[] for q in qubits_involved}

    for order_gate, ind_gate in enumerate( circuit_db["ind"]):
        if ind_gate < translator.number_of_cnots:
            control, target = translator.indexed_cnots[str(ind_gate)]
            gates_on_qubit[control].append(ind_gate)
            gates_on_qubit[target].append(ind_gate)
            on_qubit_order[control].append(order_gate)
            on_qubit_order[target].append(order_gate)
        else:
            gates_on_qubit[(ind_gate-translator.n_qubits)%translator.n_qubits].append(ind_gate)
            on_qubit_order[(ind_gate-translator.n_qubits)%translator.n_qubits].append(order_gate)
    return gates_on_qubit, on_qubit_order




def rule_1(translator, simplified_db, on_qubit_order, gates_on_qubit):
    simplification = False

    for q, qubit_gates_path in gates_on_qubit.items():
        if simplification is True:
            break
        for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path):
            if ind_gate < translator.number_of_cnots:
                control, target = translator.indexed_cnots[str(ind_gate)]
                if (q == control) and (order_gate_on_qubit == 0):
                    pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit]

                    block_id = circuit_db.loc[pos_gate_to_drop]["block_id"]
                    simplified_db.loc[int(pos_gate_to_drop)+0.1] = gate_template(translator.number_of_cnots + translator.n_qubits + control, param_value=0.0, block_id=circuit_db.loc[0]["block_id"])
                    simplified_db.loc[int(pos_gate_to_drop)+0.11] = gate_template(translator.number_of_cnots + translator.n_qubits + target, param_value=0.0, block_id=circuit_db.loc[0]["block_id"])

                    simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)

                    simplification = True
                    break
    simplified_db = simplified_db.sort_index().reset_index(drop=True)
    return simplification, simplified_db


def rule_2(translator, simplified_db, on_qubit_order, gates_on_qubit):
    simplification = False

    for q, qubit_gates_path in gates_on_qubit.items():
        if simplification is True:
            break
        for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-1]):

            next_ind_gate = qubit_gates_path[order_gate_on_qubit+1]
            if (ind_gate < translator.number_of_cnots) and (ind_gate == next_ind_gate):
                control, target = translator.indexed_cnots[str(ind_gate)]
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



def rule_3(translator, simplified_db, on_qubit_order, gates_on_qubit):
    simplification = False
    for q, qubit_gates_path in gates_on_qubit.items():
        if simplification is True:
            break
        for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-1]):
            if order_gate_on_qubit == 0 and (translator.number_of_cnots <= ind_gate< translator.number_of_cnots+ translator.n_qubits ):
                pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit]
                simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)
                simplified_db = simplified_db.reset_index(drop=True)
                simplified_db = shift_symbols_down(translator, pos_gate_to_drop, simplified_db)
                simplification = True
                break
    return simplification, simplified_db



def rule_4(translator, simplified_db, on_qubit_order, gates_on_qubit):
    """
    Repeated rotations: add the values
    """
    simplification = False
    for q, qubit_gates_path in gates_on_qubit.items():
        if simplification is True:
            break
        for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-1]):
            if ind_gate>=translator.number_of_cnots:
                next_ind_gate = qubit_gates_path[order_gate_on_qubit+1]
                if next_ind_gate == ind_gate:
                    pos_gate_to_drop = on_qubit_order[q][order_gate_on_qubit]
                    pos_gate_to_add = on_qubit_order[q][order_gate_on_qubit+1]

                    value_1 = simplified_db.loc[pos_gate_to_drop]["param_value"]
                    value_2 = simplified_db.loc[pos_gate_to_add]["param_value"]

                    simplified_db.loc[pos_gate_to_add] = simplified_db.loc[pos_gate_to_add].replace(to_replace=value_2, value=value_1 + value_2)
                    simplified_db = simplified_db.drop(labels=[pos_gate_to_drop],axis=0)
                    simplified_db = simplified_db.reset_index(drop=True)

                    simplified_db = shift_symbols_down(translator, pos_gate_to_drop, simplified_db)
                    simplification = True
                    break
    return simplification, simplified_db




def rule_5(translator, simplified_db, on_qubit_order, gates_on_qubit):
    """
    compile 1-qubit gates into euler rotations
    """
    simplification = False

    type_get = lambda x, translator: (x-translator.number_of_cnots)//translator.n_qubits

    for q, qubit_gates_path in gates_on_qubit.items():
        if simplification is True:
            break
        for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-2]):
            if simplification is True:
                break
            ind_gate_p1 = qubit_gates_path[order_gate_on_qubit+1]
            ind_gate_p2 = qubit_gates_path[order_gate_on_qubit+2]
            check_rot = lambda ind_gate: translator.number_of_cnots<= ind_gate <(3*translator.n_qubits + translator.number_of_cnots)

            if (check_rot(ind_gate) == True) and (check_rot(ind_gate_p1) == True) and (check_rot(ind_gate_p2) == True):


                type_0 = type_get(ind_gate,translator)
                type_1 = type_get(ind_gate_p1,translator)
                type_2 = type_get(ind_gate_p2,translator)


                if type_0 == type_2:
                    types = [type_0, type_1, type_2]
                    for next_order_gate_on_qubit, ind_gate_next in enumerate(qubit_gates_path[order_gate_on_qubit+3:]):
                        if (check_rot(ind_gate_next) == True):# and (next_order_gate_on_qubit < len(qubit_gates_path[order_gate_on_qubit+3:])):
                            types.append(type_get(ind_gate_next, translator))
                            simplification=True
                        else:
                            break
                    if simplification == True:
                        indices_to_compile = [on_qubit_order[q][order_gate_on_qubit+k] for k in range(len(types))]
                        translator.translator_ = CirqTranslater(n_qubits=2)
                        u_to_compile_db = simplified_db.loc[indices_to_compile]
                        u_to_compile_db["ind"] = translator.translator_.n_qubits*type_get(u_to_compile_db["ind"], translator) + translator.translator_.number_of_cnots#type_get(u_to_compile_db["ind"], translator.translator_)#translator.translator_.n_qubits*(u_to_compile_db["ind"] - translator.number_of_cnots)//translator.n_qubits + translator.translator_.number_of_cnots
                        u_to_compile_db["symbol"] = None ##just to be sure it makes no interference with the compiler...


                        compile_circuit, compile_circuit_db = construct_compiling_circuit(translator.translator_, u_to_compile_db)
                        minimizer = Minimizer(translator.translator_, mode="compiling", hamiltonian="Z")

                        cost, resolver, history = minimizer.minimize([compile_circuit], symbols=translator.get_symbols(compile_circuit_db))

                        OneQbit_translator = CirqTranslater(n_qubits=1)
                        u1s = u1_db(OneQbit_translator, 0, params=True)
                        u1s["param_value"] = -np.array(list(resolver.values()))
                        resu_comp, resu_db = OneQbit_translator.give_circuit(u1s,unresolved=False)


                        u_to_compile_db_1q = u_to_compile_db.copy()
                        u_to_compile_db_1q["ind"] = u_to_compile_db["ind"] = type_get(u_to_compile_db["ind"], translator.translator_) ##type_get(u_to_compile_db["ind"],OneQbit_translator)# - translator.translator_.number_of_cnots)//translator.translator_.n_qubits


                        cc, cdb = OneQbit_translator.give_circuit(u_to_compile_db_1q, unresolved=False)
                        c = cc.unitary()
                        r = resu_comp.unitary()



                        ## phase_shift if necessary
                        if np.abs(np.mean(c/r) -1) > 1:
                            u1s.loc[0] = u1s.loc[0].replace(to_replace=u1s["param_value"][0], value=u1s["param_value"][0] + 2*np.pi)# Rz(\th) = e^{-ii \theta \sigma_z / 2}c0, cdb0 = translator.give_circuit(pd.DataFrame([gate_template(0, param_value=2*np.pi)]), unresolved=False)
                        resu_comp, resu_db = translator.give_circuit(u1s,unresolved=False)



                        first_symbols = simplified_db["symbol"][indices_to_compile][:3]

                        for new_ind, typ, pval in zip(indices_to_compile[:3],[0,1,0], list(u1s["param_value"])):
                            simplified_db.loc[new_ind+0.1] = gate_template(translator.number_of_cnots + q + typ*translator.n_qubits,
                                                                             param_value=pval, block_id=simplified_db.loc[new_ind]["block_id"],
                                                                             trainable=True, symbol=first_symbols[new_ind])

                        for old_inds in indices_to_compile:
                            simplified_db = simplified_db.drop(labels=[old_inds],axis=0)#

                        simplified_db = simplified_db.sort_index().reset_index(drop=True)
                        killed_indices = indices_to_compile[3:]
                        db_follows = circuit_db[circuit_db.index>indices_to_compile[-1]]

                        if len(db_follows)>0:
                            gates_to_lower = list(db_follows.index)
                            number_of_shifts = len(killed_indices)
                            for k in range(number_of_shifts):
                                simplified_db = shift_symbols_down(translator, gates_to_lower[0]-number_of_shifts, simplified_db)



        break
    return simplification, simplified_db













def apply_rule(original_circuit_db, rule, **kwargs):
    max_cnt = kwargs.get('max_cnt',10)
    simplified, cnt = True, 0
    original_circuit, original_circuit_db = translator.give_circuit(original_circuit_db)
    gates_on_qubit, on_qubit_order = get_positional_dbs(original_circuit, original_circuit_db)
    simplified_db = original_circuit_db.copy()
    while simplified and cnt < max_cnt:
        simplified, simplified_circuit_db = rule(translator, simplified_db, on_qubit_order, gates_on_qubit)
        circuit, simplified_db = translator.give_circuit(simplified_circuit_db)
        gates_on_qubit, on_qubit_order = get_positional_dbs(circuit, simplified_db)
        cnt+=1
    return cnt, simplified_db
