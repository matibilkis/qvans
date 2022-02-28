import numpy as np
from utilities.circuit_database import CirqTranslater
from utilities.templates import *
from utilities.variational import Minimizer
from utilities.discrimination import *
from utilities.simplifier import Simplifier
from utilities.idinserter import IdInserter
from utilities.gate_killer import GateKiller
from utilities.evaluator import Evaluator

### discriminating amplitude damping channels ####

translator = CirqTranslater(3, untouchable_blocks = [1])

translator.env_qubits = [2]
translator.encoder_id = 0
translator.channel_id = 1
translator.decoder_id = 2


simplifier = Simplifier(translator)

etas = [0.01, 1.]
minimizer = Minimizer(translator, mode="discrimination", params=etas)
killer = GateKiller(translator, mode="discrimination")
inserter = IdInserter(translator.n_qubits, untouchable_blocks=translator.channel_id)

args_evaluator = {"n_qubits":translator.n_qubits, "problem":"acd","params":etas}
evaluator = Evaluator(args=args_evaluator, lower_bound_cost=minimizer.lower_bound_cost, nrun=0)


# u2_layer_encoder = u2_db(translator, 0,1, block_id=translator.encoder_id)
# channel_db = amplitude_damping_db(translator, qubits_ind=[1,2], eta=1, block_id = translator.channel_id)
# u2_layer_decoder = u2_db(translator, 0,1, block_id=translator.decoder_id)
#circuit_db = concatenate_dbs([u2_layer_encoder, channel_db, u2_layer_decoder])

u1_layer_encoder = u1_layer(translator, inds= [0,1], block_id=translator.encoder_id)
channel_db = amplitude_damping_db(translator, qubits_ind=[1,2], eta=1, block_id = translator.channel_id)
u1_layer_decoder = u1_layer(translator, inds = [0,1], block_id=translator.decoder_id)
circuit_db = concatenate_dbs([u1_layer_encoder, channel_db, u1_layer_decoder])


batch_circuits, trainable_symbols, trainable_params_value = prepare_optimization_discrimination(translator, circuit_db, etas)
cost, resolver, training_history = minimizer.minimize(batch_circuits, symbols = trainable_symbols, parameter_values = trainable_params_value )
circuit_db = translator.update_circuit_db_param_values(circuit_db, resolver)


evaluator.add_step(circuit_db, cost, relevant=True)
evaluator.save_dicts_and_displaying()

circuit_db, cost = killer.remove_irrelevant_gates(circuit_db)
evaluator.add_step(circuit_db, cost, relevant=False)
evaluator.save_dicts_and_displaying()

max_iter_vans=2
for iter_vans in range(max_iter_vans):
    mutated_circuit_db = inserter.insert_many_mutations(circuit_db )
    simplified_db, nreds = simplifier.reduce_circuit(mutated_circuit_db)

    batch_circuits, trainable_symbols, trainable_params_value = prepare_optimization_discrimination(translator, simplified_db, etas)

    cost, resolver, training_history = minimizer.minimize(batch_circuits, symbols = trainable_symbols, parameter_values = trainable_params_value )
    mutation_db = translator.update_circuit_db_param_values(mutation_db, resolver)

    if evaluator.accept_cost(cost):
        circuit_db = mutation_db
        circuit_db, cost = killer.remove_irrelevant_gates(circuit_db)
        evaluator.add_step(circuit_db, cost, relevant=False)
        evaluator.save_dicts_and_displaying()
