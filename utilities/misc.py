import numpy as np
import cirq
from datetime import datetime
from functools import wraps
import errno
import os
import signal
from ast import literal_eval
import pandas as pd

def overlap(st1, st2):
    return np.dot(np.conjugate(st1), st2)

def ket_bra(v1,v2):
    """
    Assuming v has shape (1,d)
    """
    return np.dot(v1.T,v2)

def bra_ket(v1,v2):
    """
    Assuming v has shape (1,d)
    """
    return np.dot(v1,v2.T)

def proj(v):
    if len(v.shape) < 2:
        v = np.expand_dims(v,axis=0)
    P= ket_bra(v,v)
    return P

def normalize(a):
    return np.array(a)/np.sqrt(np.sum(np.square(a)))



def get_def_path():
    import getpass
    user = getpass.getuser()
    if user == "cooper-cooper":
        defpath = '../data-vans/'
    else:
        defpath = "/data/uab-giq/scratch/matias/data-vans/"
    return defpath


def dict_to_json(dictionary):
    d="{"
    for k,v in dictionary.items():
        if isinstance(k,str):
            d+='\"{}\":\"{}\",'.format(k,v)
        else:
            d+='\"{}\":{},'.format(k,v)
    d=d[:-1]
    d+="}" #kill the comma
    return "\'"+d+ "\'"


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


def reindex_symbol(list_of_symbols, first_symbol_number):
    reindexed=[]
    for ind, sym in enumerate(list_of_symbols):
        if sym == [] or sym is None:
            reindexed.append(None)
        else:
            reindexed.append("th_{}".format(int(sym.replace("th_",""))+first_symbol_number))
    return reindexed


def get_qubits_involved(circuit, circuit_db):
    """
    retrieves the qubits that are touched by a circuit
    """
    all_ops = list(circuit.all_operations())
    ops_involved = [all_ops[k].qubits for k in range(len(all_ops))]
    qubits_involved = []
    for k in ops_involved:
        if len(k) == 2: #cnot
            for qq in [0,1]:
                qinv = literal_eval(k[qq].__str__())[-1]
                qubits_involved.append(qinv)
        else:
            qinv = literal_eval(k[0].__str__())[-1]
            qubits_involved.append(qinv)
    qubits_involved = list(set(qubits_involved)) #this gives you the set ordered
    return qubits_involved

def gate_counter_on_qubits(translator, circuit_db):
    """
    Gives gate count for each qbit. First entry rotations, second CNOTS
    """

    ngates = {k:[0,0] for k in range(translator.n_qubits)}
    for ind in circuit_db["ind"]:
        if ind < translator.number_of_cnots:
            control, target = translator.indexed_cnots[str(ind)]
            ngates[control][1]+=1
            ngates[target][1]+=1
        else:
            qind = (ind-translator.number_of_cnots)%translator.n_qubits
            ngates[qind][0]+=1
    return np.array(list(ngates.values()))




def get_symbol_number_from(insertion_index, circuit_db):
    ### check symbol number ###
    symbol_found=False
    for k in range(0, insertion_index+1)[::-1]:
        if type(circuit_db.loc[k]["symbol"]) == str:
            number_symbol = int(circuit_db.loc[k]["symbol"].replace("th_","")) +1
            symbol_found=True
            break
    if not symbol_found:
        number_symbol = 0
    return number_symbol


def shift_symbols_up(idinserter, indice, circuit_db):
    """
    indice is the place at which the gate was added.
    """
    for k in range(indice+2, circuit_db.shape[0]):
        if circuit_db.loc[k]["ind"] < idinserter.number_of_cnots or type(circuit_db.loc[k]["symbol"]) != str:
            pass
        else:
            old_value = circuit_db.loc[k]["symbol"]
            number_symbol = int(old_value.replace("th_","")) +1
            new_value = "th_{}".format(number_symbol)
            circuit_db.loc[k] = circuit_db.loc[k].replace(to_replace=old_value,value=new_value)
    return circuit_db


def shift_symbols_down(simplifier, indice, circuit_db):
    """
    indice is the place at which the gate was added.
    """
    for k in range(indice, circuit_db.shape[0]):
        if circuit_db.loc[k]["ind"] < simplifier.number_of_cnots or type(circuit_db.loc[k]["symbol"]) != str:
            pass
        else:
            old_value = circuit_db.loc[k]["symbol"]
            number_symbol = int(old_value.replace("th_","")) -1
            new_value = "th_{}".format(number_symbol)
            circuit_db.loc[k] = circuit_db.loc[k].replace(to_replace=old_value,value=new_value)
    return circuit_db


def check_symbols_ordered(circuit_db):
    symbol_int = list(circuit_db["symbol"].dropna().apply(lambda x: int(x.replace("th_",""))))
    return symbol_int == sorted(symbol_int)

def order_symbol_labels(circuit_db):
    """
    it happens that when a circuit is simplified, symbol labels get unsorted. This method corrects that (respecting the ordering in the gates)
    """
    if check_symbols_ordered(circuit_db) is True:
        inns = circuit_db["symbol"].dropna().index
        filtered_db = circuit_db.loc[inns]["symbol"].astype(str)
        news = ["th_{}".format(k) for k in np.sort(list(circuit_db["symbol"].dropna().apply(lambda x: int(x.replace("th_","")))))]
        sss = pd.Series(news, index=inns)
        nans = circuit_db["symbol"][circuit_db["symbol"].isna()]
        ser = pd.concat([nans,sss])
        ser = ser.sort_index()
        circuit_db = circuit_db.drop(["symbol"], axis=1)
        circuit_db.insert(loc=1, column="symbol",value=ser)
    return circuit_db

def type_get(x, translator):
    return (x-translator.number_of_cnots)//translator.n_qubits

def check_rot(ind_gate, translator):
    return translator.number_of_cnots<= ind_gate <(3*translator.n_qubits + translator.number_of_cnots)

def check_cnot(ind_gate, translator):
    return translator.number_of_cnots> ind_gate# <(3*translator.n_qubits + translator.number_of_cnots)

def qubit_get(x, translator):
    return (x-translator.number_of_cnots)%translator.n_qubits
