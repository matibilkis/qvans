import numpy as np
import cirq
import tensorflow_quantum as tfq
from utilities.circuit_basics import Basic, overlap
import tensorflow as tf
import time
from utilities.chemical import ChemicalObservable
from utilities.qmodels import *
import copy

class VQE(Basic):
    def __init__(self, n_qubits=3, lr=0.01, optimizer="sgd", epochs=1000, patience=200,
                 verbose=0, problem_config={}, return_lower_bound=True, max_vqe_time=120):
        """
        lr: learning_rate for each iteration of gradient descent
        optimizer: we give two choices, Adam and SGD. If SGD, we implement Algorithm 4 of qacq to adapt learning rate.
        epochs: number of gradient descent iterations (in this project)
        patience: EarlyStopping parameter
        verbose: display progress or not

        problem_config: dictionary that specifies the structure of the hamiltonian. Its keys will depend on the problem.
                        condensed matter:
                                problem_config["problem"] in ["XXZ", "TFIM"]
                                problem_config["g"]
                                problem_config["J"]
                        chemical:
                                problem_config["problem"] in [{molecule_name}] (for now should be H2)
                                problem_config["geometry"]
                                problem_config["charge"] (optional)
                                problem_config["multiplicity"] (optional)
                                problem_config["basis"]  (optional)


        Notes:
               (2) Hamiltonians:
               (2.1) &&ising model&& H = - g \sum_i \Z_i - (J) *\sum_i X_i X_{i+1}
               (2.2) &&xxz model$&&  H = \sum_i^{n} X_i X_{i+1} + Y_i Y_{i+1} + J Z_i Z_{i+1} + g \sum_i^{n} \sigma_i^{z}
               (2.3) molecular hamiltonians, see chemical.py
               (3) Some callbacks are used: EarlyStopping and TimeStopping.
               (4) Note that we construct the circuit according to the number of qubits required, we should add a bool check in case circuit's qubits are not enough in chemical.py
        """

        super(VQE, self).__init__(n_qubits=n_qubits)


        #### MACHINE LEARNING CONFIGURATION
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.verbose=verbose
        self.max_time_training = max_vqe_time #we give 85 to train per circuit, could be more,but it's the limit we have for 2000 iterations in the barcelona cluster..
        self.gpus=tf.config.list_physical_devices("GPU")
        self.optimizer = {"ADAM":tf.keras.optimizers.Adam,"ADAGRAD":tf.keras.optimizers.Adagrad,"SGD":tf.keras.optimizers.SGD}[optimizer.upper()]
        self.repe=0 #this is to have some control on the number of VQEs done (for tensorboard)
        self.return_lower_bound = return_lower_bound

        ##### HAMILTONIAN CONFIGURATION
        self.observable = self.give_observable(problem_config)
        ### this is inherited from circuit_basics:  self.q_batch_size
        if self.return_lower_bound is True:
            self.lower_bound_energy = self.compute_ground_energy()
        else:
            self.lower_bound_energy = -np.inf

    def give_observable(self,problem_config):
        """
        problem_config: dictionary that specifies the structure of the hamiltonian. Its keys will depend on the problem.
                        condensed matter:
                                problem_config["problem"] in ["XXZ", "TFIM"]
                                problem_config["g"]
                                problem_config["J"]
                        chemical:
                                problem_config["problem"] in [{molecule_name}] (for now should be H2)
                                problem_config["geometry"]
                                problem_config["charge"] (optional)
                                problem_config["multiplicity"]  (optional)
                                problem_config["basis"]  (optional)
        """
        with open("utilities/hamiltonians/cm_hamiltonians.txt") as f:
            hams = f.readlines()
        possible_hamiltonians = [x.strip().upper() for x in hams]
        cm_hams = possible_hamiltonians[:]

        with open("utilities/hamiltonians/chemical_hamiltonians.txt") as f:
            hams = f.readlines()
        possible_hamiltonians += ([x.strip().upper() for x in hams])
        chemical_hams = possible_hamiltonians[:]

        if problem_config["problem"] not in possible_hamiltonians:
            raise NameError("Hamiltonian {} is not invited to VANS yet. Available hamiltonians: {}\n".format(problem_config["problem"],possible_hamiltonians))

        #### CONDENSED MATTER HAMILTONIANS ####
        if problem_config["problem"].upper() in ["XXZ","TFIM"]:
            self.problem_nature = "cm"
            for field in ["g","J"]:
                if field not in problem_config.keys():
                    raise ValueError("You have not specified the fields correctly. Check out your problem_config back again. Current dict: {}".format(problem_config))
            if problem_config["problem"].upper()=="TFIM":
                #H = -J \sum_i^{n} X_i X_{i+1} - g \sum_i^{n} Z_i
                observable = [-float(problem_config["g"])*cirq.Z.on(q) for q in self.qubits]
                for q in range(len(self.qubits)):
                    observable.append(-float(problem_config["J"])*cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
                return observable
            elif problem_config["problem"].upper()=="XXZ":
                #H = \sum_i^{n} X_i X_{i+1} + Y_i Y_{i+1} + J Z_i Z_{i+1} + g \sum_i^{n} \sigma_i^{z}
                observable = [float(problem_config["g"])*cirq.Z.on(q) for q in self.qubits]
                for q in range(len(self.qubits)):
                    observable.append(cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
                    observable.append(cirq.Y.on(self.qubits[q])*cirq.Y.on(self.qubits[(q+1)%len(self.qubits)]))
                    observable.append(float(problem_config["J"])*cirq.Z.on(self.qubits[q])*cirq.Z.on(self.qubits[(q+1)%len(self.qubits)]))

                return observable

        elif problem_config["problem"].upper() in chemical_hams:
            self.problem_nature = "chemical"
            oo = ChemicalObservable()
            for key,defvalue in zip(["geometry","multiplicity", "charge", "basis"], [None,1,0,"sto-3g"]):
                if key not in list(problem_config.keys()):
                    raise ValueError("{} not specified in problem_config. Dictionary obtained: {}".format(key, problem_config))
            observable, self.lower_bound_energy =oo.give_observable(self.qubits, problem_config["geometry"], problem_config["multiplicity"], problem_config["charge"], problem_config["basis"],return_lower_bound=self.return_lower_bound)
            return observable
        else:
            raise NotImplementedError("The specified hamiltonian is in the list but we have not added to the code yet! Devs, take a look here!\problem_config[problem]: {}".format(problem_config["problem"].upper()))

    def give_energy(self, indexed_circuit, resolver):
        uni = cirq.unitary(self.give_unitary(indexed_circuit, resolver))
        st = uni[:,0]
        H = sum(self.observable).matrix()
        return overlap(st, np.dot(H,st))

    def vqe(self, indexed_circuit, symbols_to_values=None, parameter_perturbation_wall=0.5):
        """
        indexed_circuit: list with integers that correspond to unitaries (target qubit deduced from the value)

        symbols_to_values: dictionary with the values of each symbol. Importantly, they should respect the order of indexed_circuit, i.e. list(symbols_to_values.keys()) = self.give_circuit(indexed_circuit)[1]

        parameter_perturbation_wall:
        """
        circuit, symbols, index_to_symbol = self.give_circuit(indexed_circuit)
        tfqcircuit = tfq.convert_to_tensor([circuit])


        model = QNN(symbols=symbols, observable=self.observable, batch_sizes=self.q_batch_size)
        model(tfqcircuit) #this defines the weigths
        model.compile(optimizer=self.optimizer(lr=self.lr), loss=EnergyLoss())

        #in case we have already travelled the parameter space,
        if symbols_to_values is not None:
            model.trainable_variables[0].assign(tf.convert_to_tensor(np.array(list(symbols_to_values.values())).astype(np.float32)))
        else:
            model.trainable_variables[0].assign(tf.convert_to_tensor(np.pi*4*np.random.randn(len(symbols)).astype(np.float32)))

        if np.random.uniform() < parameter_perturbation_wall:
            perturbation_strength = min(1,abs(np.random.normal(scale=0.5)))
            model.trainable_variables[0].assign(model.trainable_variables[0] + tf.convert_to_tensor(np.pi*4*np.random.randn(len(symbols)).astype(np.float32))*perturbation_strength)

        calls=[tf.keras.callbacks.EarlyStopping(monitor='energy', patience=self.patience, mode="min", min_delta=0),TimedStopping(seconds=self.max_time_training)]

        if hasattr(self, "tensorboarddata"):
            self.repe+=1
            calls.append(tf.keras.callbacks.TensorBoard(log_dir=self.tensorboarddata+"/logs/{}".format(self.repe)))

        if len(self.gpus)>0:
            with tf.device(self.gpus[0]):
                training_history = model.fit(x=tfqcircuit, y=tf.zeros((self.q_batch_size,)), verbose=self.verbose, epochs=self.epochs, batch_size=self.q_batch_size, callbacks=calls)
        else:
            training_history = model.fit(x=tfqcircuit, y=tf.zeros((self.q_batch_size,)),verbose=self.verbose, epochs=self.epochs, batch_size=self.q_batch_size, callbacks=calls)

        energy = model.cost_value.result()
        final_params = model.trainable_variables[0].numpy()
        resolver = {"th_"+str(ind):var  for ind,var in enumerate(final_params)}
        return energy, resolver, training_history


    def test_circuit_qubits(self,circuit):
        """
        This function is only for testing. If there's not parametrized unitary on every qubit, raise error (otherwise TFQ runs into trouble).
        """

        effective_qubits = list(circuit[0].all_qubits())
        for k in self.qubits:
            if k not in effective_qubits:
                raise Error("NOT ALL QUBITS AFFECTED")



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





class Autoencoder(Basic):
    def __init__(self, many_indexed_circuits, many_symbols_to_values, n_qubits=3, lr=0.01, optimizer="sgd", epochs=1000, patience=200,
                 verbose=0,problem_config={},nb=1):

        super(Autoencoder, self).__init__(n_qubits=n_qubits)


        #### MACHINE LEARNING CONFIGURATION
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.verbose=verbose
        self.max_time_training = 85 #we give 85 to train per circuit, could be more,but it's the limit we have for 2000 iterations in the barcelona cluster..
        self.optimizer = {"ADAM":tf.keras.optimizers.Adam,"ADAGRAD":tf.keras.optimizers.Adagrad,"SGD":tf.keras.optimizers.SGD}[optimizer.upper()]
        self.repe=0 #this is to have some control on the number of VQEs done (for tensorboard)


        ##### HAMILTONIAN CONFIGURATION
        self.nb = nb

        self.observable = self.give_observable(mode="local")
        if self.nb > len(self.qubits):
            raise AttributeError("problem w/ # of autoencoder trash qubits")
        self.qbatch = self.give_batch_of_circuits(many_indexed_circuits, many_symbols_to_values)#mixed states


    def zero_proj(self,q):
        return (1 + cirq.Z(q)) / 2


    def give_observable(self,mode="local"):
        return [float(1/self.nb)*self.zero_proj(q) for q in self.qubits[:self.nb]]

    def give_batch_of_circuits(self, listas, resolvers):
        """
        this guy gives the mixed state i form of batched circuits (wegiths are added later)
        """
        qbatch=[]
        for pure_indexed, resolver in zip(listas, resolvers):
            circuit=self.give_circuit(pure_indexed)[0]
            preparation_circuit=cirq.resolve_parameters(circuit, resolver)
            qbatch.append(preparation_circuit)
        return qbatch

    def autoencoder(self, indexed_circuit, symbols_to_values=None, parameter_perturbation_wall=0.05):
        """
        blablabla
        """
        qbatch = []#self.give_batch_of_circuits(many_indexed_circuits, many_symbols_to_values)

        au_circuit,symbols  = self.give_circuit(indexed_circuit)[0:2]
        qbatch=[]
        qq=copy.deepcopy(self.qbatch)
        for qc in qq:
            qc.append(au_circuit)
            qbatch.append(qc)
        self.q_batch_size = len(qbatch)

        if symbols_to_values == None:
            model = QNN(symbols=symbols, observable=self.observable, batch_sizes=len(qbatch))
        else:
            model = QNN(symbols=list(symbols_to_values.keys()), observable=self.observable, batch_sizes=len(qbatch))

        tfqcircuit=tfq.convert_to_tensor(qbatch)
        model(tfqcircuit)
        model.compile(optimizer=self.optimizer(lr=self.lr), loss=EnergyLoss(mode_var="autoencoder"))
        #
        # #in case we have already travelled the parameter space,
        if symbols_to_values is not None:
            model.trainable_variables[0].assign(tf.convert_to_tensor(np.array(list(symbols_to_values.values())).astype(np.float32)))

        else:
            model.trainable_variables[0].assign(tf.convert_to_tensor(np.pi*4*np.random.randn(len(symbols)).astype(np.float32)))

        if np.random.uniform() < parameter_perturbation_wall:
            perturbation_strength = np.random.uniform()
            model.trainable_variables[0].assign(model.trainable_variables[0] + np.random.randn(len(symbols))*perturbation_strength)

        calls=[tf.keras.callbacks.EarlyStopping(monitor='energy', patience=self.patience, mode="min", min_delta=0.0),TimedStopping(seconds=self.max_time_training)]
        #
        if hasattr(self, "tensorboarddata"):
            self.repe+=1
            calls.append(tf.keras.callbacks.TensorBoard(log_dir=self.tensorboarddata+"/logs/{}".format(self.repe)))
        #

        training_history = model.fit(x=tfqcircuit, y=tf.zeros((self.q_batch_size,)),verbose=self.verbose, epochs=self.epochs, batch_size=self.q_batch_size, callbacks=calls)
        #
    #    antifidelity = 1-(((1/self.nb)*model.cost_value.result())/len(self.qbatch)) #equal priors.
        final_cost = model.cost_value.result()
        final_params = model.trainable_variables[0].numpy()
        resolver = {"th_"+str(ind):var  for ind,var in enumerate(final_params)}
        return final_cost, resolver, training_history
        #



    def test_circuit_qubits(self,circuit):
        """
        This function is only for testing. If there's not parametrized unitary on every qubit, raise error (otherwise TFQ runs into trouble).
        """

        effective_qubits = list(circuit[0].all_qubits())
        for k in self.qubits:
            if k not in effective_qubits:
                raise Error("NOT ALL QUBITS AFFECTED")
