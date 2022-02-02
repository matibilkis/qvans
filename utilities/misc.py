import numpy as np
import cirq
from datetime import datetime
from functools import wraps
import errno
import os
import signal
from ast import literal_eval


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
