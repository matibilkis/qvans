import numpy as np
from utilities.sdp import *

import cirq
from utilities.sanity import check_params
import tensorflow_quantum as tfq
import tensorflow as tf

class QNN_DISCRIMINATION(tf.keras.Model):
    def __init__(self, symbols, observable, batch_sizes=2):
        """
        symbols: symbolic variable [sympy.Symbol]*len(rotations_in_circuit)
        batch_size: how many circuits you feed the model at, at each call (this might )
        """
        super(QNN_DISCRIMINATION,self).__init__()
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
            energy = self.compiled_loss(preds, preds) #notice that compiled loss takes care only about the preds
        train_vars = self.trainable_variables
        grads=tape.gradient(energy,train_vars)
        self.gradient_norm.update_state(tf.reduce_sum(tf.pow(grads[0],2)))
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.cost_value.update_state(energy)
        self.lr_value.update_state(self.optimizer.lr)
        return {k.name:k.result() for k in self.metrics}

    @property
    def metrics(self):
        return [self.cost_value, self.lr_value,self.gradient_norm]









class PerrLoss(tf.keras.losses.Loss):
    """
    discard_qubits:::  qubits where you don't measure
    """
    def __init__(self, discard_qubits):
        super(PerrLoss,self).__init__()
        self.discard_qubits = discard_qubits

    def call(self, y_true, y_pred):
        p=0
        ### outcomes without environment (measurements are local thus you just environemnt ones)
        outs=[]
        for k in range(y_pred.shape[1]):
            if k not in self.self.discard_qubits:
                outs.append(y_pred[:,k])
        outcomes_env_discarded = tf.stack(outs, axis=1)

        prob_outcomes = tf.math.reduce_prod(outcomes_env_discarded, axis=1)
        probUp = (prob_outcomes+1)/2
        probDown = (1-prob_outcomes)/2
        p+=tf.math.reduce_max(probUp)
        p+=tf.math.reduce_max(probDown)
        return 1-p/2



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
