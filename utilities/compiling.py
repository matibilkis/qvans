import cirq
from utilities.sanity import check_params
import tensorflow_quantum as tfq
import tensorflow as tf
import  numpy as np
import pandas as pd
from utilities.templates import gate_template, u1_db, concatenate_dbs


def conjugate_db(translator, v_to_compile_db):
    """
    conjugate pauli rotations and set trainable to False
    """
    conjugate_v_to_compile = v_to_compile_db.copy()
    conjugate_v_to_compile["trainable"] = False
    for ind, gate_id in conjugate_v_to_compile.iterrows():
        if translator.number_of_cnots <= gate_id["ind"] <= translator.number_of_cnots + 3*translator.n_qubits:
            mcof = [-1,-1,1][(gate_id["ind"]-translator.number_of_cnots)//translator.n_qubits] ###this conjugates paulis  rz, rx, ry
            conjugate_v_to_compile.loc[ind].replace(to_replace=gate_id["param_value"], value=gate_id["param_value"]*mcof)
    return conjugate_v_to_compile

def construct_compiling_circuit(translator, conjugate_v_to_compile_db):
    """
    compiling single-qubit unitary (for the moment)

    v_to_compile is a cirq.Circuit object (single-qubit for the moment)
    """
    qubits = translator.qubits[:2]
    systems = qubits[:int(len(qubits)/2)]
    ancillas = qubits[int(len(qubits)/2):]

    forward_bell = [translator.number_of_cnots + 3*translator.n_qubits + i for i in range(int(translator.n_qubits/2))]
    forward_bell += [translator.cnots_index[str([k, k+int(translator.n_qubits/2)])] for k in range(int(translator.n_qubits/2))]
    bell_db = pd.DataFrame([gate_template(k, param_value=None, trainable=False) for k in forward_bell])
    u1s = u1_db(translator, 0, params=True)
    #target_unitary_db = pd.DataFrame([gate_template(ind=-1, param_value = np.conjugate(v_to_compile), trainable=False, qubits=[1])])
    backward_bell_db = bell_db[::-1]
    id_comp = concatenate_dbs([bell_db, u1s, conjugate_v_to_compile_db, backward_bell_db])
    comp_circ, comp_db = translator.give_circuit(id_comp, unresolved=True)
    return comp_circ, comp_db


def give_observable_compiling(minimizer):
    return [cirq.Z.on(q) for q in minimizer.qubits]

def compute_lower_bound_cost_compiling(minimizer):
    return 0.

class QNN_Compiling(tf.keras.Model):
    def __init__(self, symbols, observable, batch_sizes=1):
        """
        symbols: symbolic variable [sympy.Symbol]*len(rotations_in_circuit)
        batch_size: how many circuits you feed the model at, at each call (this might )
        """
        super(QNN_Compiling,self).__init__()
        self.expectation_layer = tfq.layers.Expectation()
        self.symbols = symbols
        self.observable = tfq.convert_to_tensor([observable]*batch_sizes)
        self.cost_value = Metrica(name="cost")
        self.lr_value = Metrica(name="lr")
        self.gradient_norm = Metrica(name="grad_norm")

    def call(self, inputs):
        """
        inputs: tfq circuits (resolved or not, to train one has to feed unresolved).
        """
        feat = inputs
        f = self.expectation_layer(feat, operators=self.observable, symbol_names=self.symbols)
        f = tf.math.reduce_sum(f,axis=-1)
        return f

    def train_step(self, data):
        x,y=data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            preds = self(x,training=True)
            cost = self.compiled_loss(preds, preds) #notice that compiled loss takes care only about the preds
        train_vars = self.trainable_variables
        grads=tape.gradient(cost,train_vars)
        self.gradient_norm.update_state(tf.reduce_sum(tf.pow(grads[0],2)))
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.cost_value.update_state(cost)
        self.lr_value.update_state(self.optimizer.lr)
        return {k.name:k.result() for k in self.metrics}

    @property
    def metrics(self):
        return [self.cost_value, self.lr_value,self.gradient_norm]

class CompilingLoss(tf.keras.losses.Loss):
    def __init__(self, mode_var="compiling", **kwargs):
        super(CompilingLoss,self).__init__()
        self.mode_var = mode_var
        self.d = kwargs.get("d", 2) #dimension

    def call(self, y_true, y_pred):
        return 1.-tf.math.reduce_sum(y_pred,axis=-1)/self.d


class Metrica(tf.keras.metrics.Metric):
    def __init__(self, name):
        super(Metrica, self).__init__()
        self._name=name
        self.metric_variable = self.add_weight(name=name, initializer='zeros')

    def update_state(self, new_value, sample_weight=None):
        self.metric_variable.assign(new_value)

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable.assign(0.)
