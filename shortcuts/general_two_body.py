import numpy as np
from utilities.circuit_database import CirqTranslater
from utilities.templates import *
from utilities.variational import Minimizer
from utilities.discrimination import *
from utilities.simplifier import Simplifier
from utilities.idinserter import IdInserter
from utilities.gate_killer import GateKiller
### discriminating amplitude damping channels ####

#### this program modifies a circuit_db object (pd.DataFrame) describing a unitary transofmation acting on 3 qubits. It optimizes both structure + parameters


translator = CirqTranslater(3, untouchable_blocks = [1])
translator.env_qubits = [2]

translator.encoder_id = 0
translator.channel_id = 1
translator.decoder_id = 2

### state prep

u2_layer_encoder = u2_db(translator, 0,1, block_id=translator.encoder_id)
channel_db = amplitude_damping_db(translator, qubits_ind=[1,2], eta=1, block_id = translator.channel_id)
u2_layer_decoder = u2_db(translator, 0,1, block_id=translator.decoder_id)

#communication protocol as a whole
circuit_db = concatenate_dbs([u2_layer_encoder, channel_db, u2_layer_decoder])
circuit, circuit_db = translator.give_circuit(circuit_db)

### minimizer first ###
minimizer = Minimizer(translator, mode="discrimination")
etas = [0.01, 1.]
batch_circuits, batch_circuits_db = channel_circuits(translator, circuit_db, etas ) ###prepare a batch of circuits so tfq compute the mean value of Z on each qubit
trainable_symbols = translator.get_trainable_symbols(batch_circuits_db[0])
trainable_params_value = translator.get_trainable_params_value(circuit_db)
cost, resolver, training_history = minimizer.minimize(batch_circuits, symbols = trainable_symbols, parameter_values = trainable_params_value )
circuit_db = translator.update_circuit_db_param_values(circuit_db, resolver)


#### insert identity resolutions (without affecting the channel_block)
inserter = IdInserter(translator.n_qubits, untouchable_blocks=translator.channel_id)
mutated_circuit_db = inserter.insert_many_mutations(circuit_db )

#### initialize simplifier ###
simplifier = Simplifier(translator)
circuit_db, nreds = simplifier.reduce_circuit(circuit_db)

batch_circuits, trainable_symbols, trainable_params_value = prepare_optimization_discrimination(translator, mutated_circuit_db, etas)
cost, resolver, training_history = minimizer.minimize(batch_circuits, symbols = trainable_symbols, parameter_values = trainable_params_value )

##initialize gate killer ###
killer = GateKiller(translator, mode="discrimination", params=etas)
killed_db, new_cost = killer.remove_irrelevant_gates(cost, circuit_db)
circuit_db = killed_db
