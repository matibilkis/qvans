import tensorflow as tf
import numpy as np
import time
from utilities.compiling import *
from utilities.vqe import *

class Minimizer:
    def __init__(self,
                translator,
                mode,
                lr=0.01,
                optimizer="adam",
                epochs=1000,
                patience=200,
                max_time_continuous=120,
                parameter_noise=0.01,
                **kwargs):

            ## training hyperparameters
            self.lr = lr
            self.epochs = epochs
            self.patience = patience
            self.max_time_training = max_time_continuous
            self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)
            self.parameter_noise = parameter_noise
            self.minimization_step=0 #used for tensorboard

            if mode.upper() == "VQE":
                hamiltonian = kwargs.get("hamiltonian")
                params = kwargs.get("params")
                self.observable = give_observable_vqe(translator,hamiltonian, params)
                self.loss = EnergyLoss()
                self.model_class = QNN_VQE
                self.lower_bound_cost = compute_lower_bound_cost_vqe(self) ## this will only work
                self.target_preds = None ##this is to compute the cost

            elif mode.upper() == "DISCRIMINATION":

                params = kwargs.get("params")
                self.observable = [cirq.Z.on(q) for q in translator.qubits]
                self.loss = Prob
                self.target_preds = None ##this is to compute the cost

            elif mode.upper() == "COMPILING":

                self.observable = give_observable_compiling(translator)
                self.loss = CompilingLoss(d = translator.n_qubits)
                self.model_class = QNN_Compiling
                self.lower_bound_cost = compute_lower_bound_cost_compiling(self) ## this will only work
                self.target_preds = None ##this is to compute the cost


    def give_cost(self, batched_cicuits, resolver, model=None):
        ### example: minimizer.give_cost(  [translator.give_circuit(circuit_db)[0]], resolver )
        if model is None:
            model = self.model_class(symbols = list(resolver.keys()), observable=self.observable, batch_sizes=len(batched_cicuits))
        tfqcircuit = tfq.convert_to_tensor([cirq.resolve_parameters(circuit,resolver) for circuit in batched_cicuits])
        return self.loss(self.target_preds, model(tfqcircuit))  #y_target y_pred

    def minimize(self, batched_circuits, symbols, parameter_values=None, parameter_perturbation_wall=1):
        """
        batched_circuits:: list of cirq.Circuits (should NOT be resolved or with Sympy.Symbol)
        symbols:: list of strings containing symbols for each rotation
        parameter_values:: values of previously optimized parameters
        parameter_perturbation_wall:: with some probability move away from the previously optimized parameters (different initial condition)
        """
        batch_size = len(batched_circuits)
        self.model = self.model_class(symbols=symbols, observable=self.observable, batch_sizes=batch_size)

        tfqcircuit = tfq.convert_to_tensor(batched_circuits)
        self.model(tfqcircuit) #this defines the weigths
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        #in case we have already travelled the parameter space,
        if parameter_values is not None:
            self.model.trainable_variables[0].assign(tf.convert_to_tensor(np.array(list(parameter_values.values())).astype(np.float32)))
        else:
            self.model.trainable_variables[0].assign(tf.convert_to_tensor(np.pi*4*np.random.randn(len(symbols)).astype(np.float32)))

        if np.random.uniform() < parameter_perturbation_wall:
            perturbation_strength = abs(np.random.normal(scale=np.max(np.abs(self.model.trainable_variables[0]))))
            self.model.trainable_variables[0].assign(self.model.trainable_variables[0] + tf.convert_to_tensor(perturbation_strength*np.random.randn(len(symbols)).astype(np.float32)))

        calls=[tf.keras.callbacks.EarlyStopping(monitor='cost', patience=self.patience, mode="min", min_delta=0),TimedStopping(seconds=self.max_time_training)]

        if hasattr(self, "tensorboarddata"):
            self.minimization_step+=1 #this is because we call the module many times !
            calls.append(tf.keras.callbacks.TensorBoard(log_dir=self.tensorboarddata+"/logs/{}".format(self.minimization_step)))

        training_history = self.model.fit(x=tfqcircuit, y=tf.zeros((batch_size,)),verbose=0, epochs=self.epochs, batch_size=batch_size, callbacks=calls)

        cost = self.model.cost_value.result()
        final_params = self.model.trainable_variables[0].numpy()
        resolver = {"th_"+str(ind):var  for ind,var in enumerate(final_params)}
        return cost, resolver, training_history



class TimedStopping(tf.keras.callbacks.Callback):
    '''Stop training when enough time has passed.
        # Arguments
        seconds: maximum time before stopping.
        verbose: verbosity mode.
    '''
    def __init__(self, seconds=None, verbose=1):
        super(TimedStopping, self).__init__()
        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose>0:
                print('Stopping after %s seconds.' % self.seconds)
