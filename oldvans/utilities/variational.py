import numpy as np
import cirq
import tensorflow_quantum as tfq
from utilities.circuit_basics import Basic
import tensorflow as tf
import time
from utilities.qmodels import *
import copy
from utilities.cost_functions import give_observable, compute_lower_bound_cost


class Minimizer(Basic):
    def __init__(self,
                n_qubits=3,
                env_mode = 1,
                lr=0.01,
                optimizer="adam",
                epochs=1000,
                patience=200,
                problem_config="TFIM",
                lower_bound=None,
                compute_lower_bound=None,
                max_time_continuous=120,
                batch_size=2, #in discrimination, batch_size = #hypothesis. In standard VQE, batch_size = 1.
                parameter_noise=0.01,
                verbose=0): #parameter noise: variance of white noise added to paramteres before (re)training

        super(Minimizer, self).__init__(n_qubits=n_qubits)

        #### MACHINE LEARNING CONFIGURATION
        self.lr = lr
        self.epochs = epochs
        self.patience = patience

        self.max_time_training = max_time_continuous
        self.gpus=tf.config.list_physical_devices("GPU")
        self.optimizer = {"ADAM":tf.keras.optimizers.Adam,"ADAGRAD":tf.keras.optimizers.Adagrad,"SGD":tf.keras.optimizers.SGD}[optimizer.upper()]
        self.repe=0 #this is to have some control on the number of VQEs done (for tensorboard)
        self.env_mode = env_mode

        ##### HAMILTONIAN CONFIGURATION, see cost_functions
        self.observable = give_observable(self,problem_config)
        self.lower_bound_cost = compute_lower_bound_cost(self, lower_bound, compute_lower_bound)

        self.q_batch_size = batch_size
        self.parameter_noise = parameter_noise
        self.verbose = verbose

    def give_cost(self, batch):
        """
        indexed_circuit: list with integers that correspond to unitaries (target qubit deduced from the value)
        symbols_to_values: dictionary with the values of each symbol. Importantly, they should respect the order of indexed_circuit, i.e. list(symbols_to_values.keys()) = self.give_circuit(indexed_circuit)[1]
        """
        if not hasattr(self, "model"):
            raise AttributeError("You should first run the optimize routine!")
        p = self.model.compiled_loss(*[self.model(tfq.convert_to_tensor( [batch[k][0] for k in range(len(batch))] ))]*2)
        return p.numpy()

    def channel_discrimination(self, batch, symbols_to_values=None, parameter_perturbation_wall=0.5):
        """
        batch = [[circuit0, symbols0, circuit_db0], [circuit1, symbols1, circuit_db1] ]
        batch = give_circuit_batch(basic,etas=[0.01, 1.0])
        """

        circuit_db_training = self.hide_trainable_coefficients(batch[0][2])
        trainable_symbols = self.give_symbols_from_db(circuit_db_training)

        self.model = QNN(symbols = trainable_symbols, operators=self.observable)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss=PerrLoss(env_mode=self.env_mode))

        call1 = self.model(tfq.convert_to_tensor( [batch[k][0] for k in range(len(batch))] ))
        #data_out = np.array([[None],[None]], dtype=np.float32)

        if symbols_to_values is None:
            self.model.trainable_variables[0].assign(tf.convert_to_tensor(np.pi*4*np.random.randn(len(self.model.symbols)).astype(np.float32)))
        else:
            self.model.trainable_variables[0].assign(tf.convert_to_tensor(np.array(list(symbols_to_values.values())).astype(np.float32)))

        ### apply some random kink to the variables
        self.model.trainable_variables[0].assign(self.model.trainable_variables[0] + tf.convert_to_tensor(self.parameter_noise * np.pi*4*np.random.randn(len(trainable_symbols)).astype(np.float32)))

        calls=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience, mode="min", min_delta=0),TimedStopping(seconds=self.max_time_training)]

        if hasattr(self, "tensorboar_dir"):
            self.repe+=1
            calls.append(tf.keras.callbacks.TensorBoard(log_dir=self.tensorboar_dir+"/logs/{}".format(self.repe)))

        tfqcircuit = tfq.convert_to_tensor( [batch[k][0] for k in range(len(batch))] )
        if len(self.gpus)>0:
            with tf.device(self.gpus[0]):
                training_history = self.model.fit(x=tfqcircuit, y=tf.zeros((self.q_batch_size,)), verbose=self.verbose, epochs=self.epochs, batch_size=self.q_batch_size, callbacks=calls)
        else:
            training_history = self.model.fit(x=tfqcircuit, y=tf.zeros((self.q_batch_size,)),verbose=self.verbose, epochs=self.epochs, batch_size=self.q_batch_size, callbacks=calls)

        cost_value = self.model.cost_value.result()
        final_params = self.model.trainable_variables[0].numpy()
        resolver = {s:val  for s,val in zip(self.model.symbols, final_params)}
        return cost_value, resolver, training_history
