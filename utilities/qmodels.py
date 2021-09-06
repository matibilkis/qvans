import tensorflow_quantum as tfq
import tensorflow as tf

#class UnitaryCompile(tf.keras.model)



class QNN(tf.keras.Model):
    def __init__(self, symbols, observable, batch_sizes=1):
        """
        symbols: symbolic variable [sympy.Symbol]*len(rotations_in_circuit)
        batch_size: how many circuits you feed the model at, at each call (this might )
        """
        super(QNN,self).__init__()
        self.expectation_layer = tfq.layers.Expectation()
        self.symbols = symbols
        self.observable = tfq.convert_to_tensor([observable]*batch_sizes)
        self.cost_value = Metrica(name="energy")
        self.lr_value = Metrica(name="lr")
        self.gradient_norm = Metrica(name="grad_norm")

    def call(self, inputs):
        """
        inputs: tfq circuits (resolved or not, to train one has to feed unresolved)
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

        if self.optimizer.get_config()["name"] == "SGD":
            self.qacq_gradients(energy,grads,x)
        else:
            self.optimizer.apply_gradients(zip(grads, train_vars))
        self.cost_value.update_state(energy)
        self.lr_value.update_state(self.optimizer.lr)
        return {k.name:k.result() for k in self.metrics}


    def qacq_gradients(self, cost, grads,x):
        """
        Algorithm 4 of https://arxiv.org/pdf/1807.00800.pdf
        """
        g=tf.reduce_sum(tf.pow(grads[0],2))
        initial_lr = tf.identity(self.optimizer.lr)
        initial_params = tf.identity(self.trainable_variables)

        #compute line 10
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        alpha1 = tf.identity(self.trainable_variables)
        preds1 = self(x)
        cost1 = self.compiled_loss(preds1,preds1)

        #compute line 11
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        alpha2 = tf.identity(self.trainable_variables)
        preds2=self(x)
        cost2 = self.compiled_loss(preds2,preds2)

        self.condi(tf.math.greater_equal(cost - cost2, initial_lr*g),tf.math.greater_equal(initial_lr*g/2,cost - cost1), initial_lr,alpha1, alpha2)
        return

    @tf.function
    def condi(self,var1, var2, initial_lr, alpha1, alpha2):
        if var1 == True:
            self.optimizer.lr.assign(2*initial_lr)
            self.trainable_variables[0].assign(alpha2[0])
        else:
            if var2 == True:
                #self.optimizer.lr.assign(tf.reduce_max([1e-4,initial_lr/2]))
                self.optimizer.lr.assign(initial_lr/2)
                self.trainable_variables[0].assign(alpha1[0])
            else:
                self.trainable_variables[0].assign(alpha1[0])

    @property
    def metrics(self):
        return [self.cost_value, self.lr_value,self.gradient_norm]


class EnergyLoss(tf.keras.losses.Loss):
    """
    this is a very simple loss that
    """
    def __init__(self, mode_var="vqe"):
        super(EnergyLoss,self).__init__()
        self.mode_var = mode_var
    def call(self, y_true, y_pred):
        #reduce_mean in case we have a batch of circuits (for the noise) otherwise it's
        if self.mode_var == "autoencoder":
            return 1-(tf.math.reduce_mean(y_pred,axis=-1))
        else:
            return tf.math.reduce_mean(y_pred,axis=-1)

class Metrica(tf.keras.metrics.Metric):
    """
    for testing purposes: this is intended
    """
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
