import pandas as pd
import numpy as np


def u2_db(translator,a,b,**kwargs):
    block_id = kwargs.get("block_id",0)
    params = kwargs.get("params",True)
    def give_param(params, k):
        if params is not True:
            return None
        else:
            return np.random.normal(0,2*np.pi)
    return pd.DataFrame([gate_template(k, block_id=block_id, param_value = give_param(params,k)) for i,k in enumerate(u2(translator,a,b))])

def u1_db(translator,q,**kwargs):
    block_id = kwargs.get("block_id",0)
    params = kwargs.get("params",True)
    def give_param(params, k):
        if params is not True:
            return None
        else:
            return np.random.normal(0,2*np.pi)
    return pd.DataFrame([gate_template(k, block_id=block_id, param_value = give_param(params,k)) for i,k in enumerate(u1(translator,q))])

def u2_layer(translator,**kwargs):
    block_id = kwargs.get("block_id",0)
    dd = u2_db(translator,0,1)
    for i in range(1,translator.n_qubits):
        dd = concatenate_dbs([dd,u2_db(translator,i,(i+1)%translator.n_qubits, block_id=block_id)])
    return dd

def u1_layer(translator, **kwargs):
    block_id = kwargs.get("block_id",0)
    dd = u1_db(translator,0)
    for i in range(1,translator.n_qubits):
        dd = concatenate_dbs([dd,u1_db(translator,i, block_id=block_id)])
    return dd

def x_layer_db(translator, **kwargs):
    block_id = kwargs.get("block_id",0)
    xx = pd.DataFrame([gate_template(k, param_value=0.,block_id=block_id) for k in [translator.number_of_cnots + translator.n_qubits+j for j in range(translator.n_qubits)]])
    return xx

def z_layer_db(translator,**kwargs):
    block_id = kwargs.get("block_id",0)
    zz = pd.DataFrame([gate_template(k, param_value=0., block_id=block_id) for k in [translator.number_of_cnots +j for j in range(translator.n_qubits)]])
    return zz

def cnot_layer(translator, **kwargs):
    touching = kwargs.get("touching",False)
    block_id = kwargs.get("block_id",0)
    if touching is True:
        inds_cnots = range(0,translator.n_qubits,2)
    else:
        inds_cnots = range(0,translator.n_qubits)
    cnots = [translator.cnots_index[str([k,(k+1)%translator.n_qubits])] for k in inds_cnots]
    return pd.DataFrame([gate_template(k, block_id=block_id) for k in cnots])


def concatenate_dbs(dbs):
    d = dbs[0]
    for dd in dbs[1:]:
        d = pd.concat([d,dd])
        d = d.reset_index(drop=True)
    return d

def what_if_none(x,alternative=None):
    """
    aid method for give_gate_template
    """
    if x is None:
        return alternative
    else:
        return x

def gate_template(ind,**kwargs):
    """
    Creates a dictionary (that is later converted into pandas dataframe) describing rows of gate associated to ind. Used as initialization shortcut.
    gate_id: {"param_value":None, "trainable":True, "block_id":0, "movable":True}
    """
    dicti = {"ind": ind}
    dicti["symbol"] = what_if_none(kwargs.get("symbol"))
    dicti["param_value"] = what_if_none(kwargs.get("param_value"),None)
    dicti["trainable"] = what_if_none(kwargs.get("trainable"),True)
    dicti["block_id"] = what_if_none(kwargs.get("block_id"), 0)
    qubits = kwargs.get("qubits", None)
    if qubits is not None:
        dicti["qubits"] = qubits
    return dicti


def rz(translator, q):
    return translator.number_of_cnots  + q
def rx(translator, q):
    return translator.number_of_cnots + translator.n_qubits + q
def ry(translator, q):
    return translator.number_of_cnots + 2*translator.n_qubits + q
def cnot(translator, q0, q1):
    return translator.cnots_index[str([q0,q1])]
def u1(translator, q):
    return [rz(translator, q), rx(translator,q), rz(translator,q)]
def u2(translator, q0, q1):
    """general two-qubit gate"""
    l=[ u for u in u1(translator,q0)]
    for u in u1(translator, q1):
        l.append(u)
    l.append(cnot(translator,q0,q1))
    l.append(rz(translator,q0))
    l.append(ry(translator,q1))
    l.append(cnot(translator,q1,q0))
    l.append(ry(translator,q1))
    l.append(cnot(translator,q0,q1))
    for u in u1(translator, q0):
        l.append(u)
    for u in u1(translator, q1):
        l.append(u)
    return l
