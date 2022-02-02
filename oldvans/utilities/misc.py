import numpy as np
import cirq
from datetime import datetime
from functools import wraps
import errno
import os
import signal


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


def scheduler_selector_temperature(energy, lowest_energy_found, when_on=10):
    relative_energy = np.abs((energy - lowest_energy_found)/lowest_energy_found)
    if relative_energy < 1e-1:
       return 1#
    else:
       return when_on

def scheduler_parameter_perturbation_wall(its_without_improvig, max_randomness=.8, min_randomness=.1, decrease_to=20):
    slope = (max_randomness-min_randomness)/decrease_to
    return np.min([min_randomness + slope*its_without_improvig, max_randomness])

def give_kronecker_of_list(lista):
    #lista=[auto_handler.zero_proj(vqe_handler.qubits[k]).matrix() for k in range(3)]
    m=[]
    for ind,k in enumerate(lista):
        if ind == 0:
            m.append(k)
        else:
            m.append(np.kron(m[-1],k))
    return m[-1]


def give_kr_prod(matrices):
    #matrices list of 2 (or more in principle) matrices
    while len(matrices) != 1:
        sm, smf=[],[]
        for ind in range(len(matrices)):
            sm.append(matrices[ind])
            if len(sm) == 2:
                smf.append(np.kron(*sm))
                sm=[]
        matrices = smf
    return matrices[0]


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
