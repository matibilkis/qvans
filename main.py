%load_ext autoreload
%autoreload 2

import sympy
import numpy as np
import pandas as pd
import tensorflow as tf
from utilities.circuit_database import CirqTranslater
from utilities.templates import *
from utilities.variational import Minimizer
from utilities.misc import get_qubits_involved, reindex_symbol, shift_symbols_down
import matplotlib.pyplot as plt
import tensorflow_quantum as tfq
import cirq
from utilities.compiling import *
from utilities.simplifier import Simplifier


translator = CirqTranslater(3)
circs={}
for block in [1,2]:
    xlayer = x_layer_db(translator, block_id=block)
    zlayer = z_layer_db(translator, block_id=block)
    cnots_layer = cnot_layer(translator, block_id=block)
    if block==1:
        circs[block] = concatenate_dbs([xlayer, zlayer,xlayer, zlayer, zlayer, cnots_layer])
    else:
        circs[block] = concatenate_dbs([xlayer, cnots_layer])

circuit_db = concatenate_dbs([c for c in circs.values()])
circuit, circuit_db = translator.give_circuit(circuit_db)

translator.give_circuit(circuit_db,unresolved=False)[0]
