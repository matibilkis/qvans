import cirq
from utilities.sanity import check_params
import tensorflow_quantum as tfq
import tensorflow as tf
import  numpy as np

def give_observable(minimizer, hamiltonian, params):
    if hamiltonian.upper() == "TFIM":
        check_params(params,2)
        g, J = params
        observable = [-float(g)*cirq.Z.on(q) for q in minimizer.qubits]
        for q in range(len(minimizer.qubits)):
            observable.append(-float(J)*cirq.X.on(minimizer.qubits[q])*cirq.X.on(minimizer.qubits[(q+1)%len(minimizer.qubits)]))
        return observable
    elif hamiltonian.upper() == "XXZ":
        check_params(params,2)
        g, J = params
        observable = [float(g)*cirq.Z.on(q) for q in minimizer.qubits]
        for q in range(len(minimizer.qubits)):
            observable.append(cirq.X.on(minimizer.qubits[q])*cirq.X.on(minimizer.qubits[(q+1)%len(minimizer.qubits)]))
            observable.append(cirq.Y.on(minimizer.qubits[q])*cirq.Y.on(minimizer.qubits[(q+1)%len(minimizer.qubits)]))
            observable.append(float(J)*cirq.Z.on(minimizer.qubits[q])*cirq.Z.on(minimizer.qubits[(q+1)%len(minimizer.qubits)]))
        return observable
    else:
        raise NotImplementedError("Hamiltonian not implemented yet")

def compute_lower_bound_cost_vqe(minimizer):
    print("computing ground state energy...")
    return np.real(np.min(np.linalg.eigvals(sum(minimizer.observable).matrix())))


class QNN_VQE(tf.keras.Model):
    def __init__(self, symbols, observable, batch_sizes=1):
        """
        symbols: symbolic variable [sympy.Symbol]*len(rotations_in_circuit)
        batch_size: how many circuits you feed the model at, at each call (this might )
        """
        super(QNN_VQE,self).__init__()
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

class EnergyLoss(tf.keras.losses.Loss):
    def __init__(self, mode_var="vqe"):
        super(EnergyLoss,self).__init__()
        self.mode_var = mode_var
    def call(self, y_true, y_pred):
        return tf.math.reduce_sum(y_pred,axis=-1)

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
