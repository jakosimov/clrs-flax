from experiment_helpers import (
    DatasetConfig,
    MPNNConfig,
    AggregationMode,
    run_experiment,
)
import jax

algorithm = "matrix_chain_order"

standard_dataset = DatasetConfig(
    algorithm_name=algorithm,
    num_samples=1000,
    length=16,
    train_batch_size=32,
    test_batch_size=8,
    num_test_samples=32,
)

default_learning_rate = 1e-3

current_mpnn_config = MPNNConfig(
    aggregation_modes=[AggregationMode.SUM, AggregationMode.MAX],
    decoder_learning_rate=default_learning_rate,
    encoder_learning_rate=default_learning_rate,
    backbone_learning_rate=default_learning_rate,
    max_steps=10000,
    disable_jit=False,
    message_weight_decay=0.0,
    hint_teacher_forcing=0.0,
    hidden_dim=64,
    dropout_prob=0.0,
    nb_heads=4,
)

debug_dataset = DatasetConfig(
    algorithm_name=algorithm,
    num_samples=50,
    length=8,
    train_batch_size=8,
    test_batch_size=32,
    num_test_samples=32,
)

# standard_dataset = debug_dataset
print("Running on", jax.devices()[-1].device_kind)

import time

start_time = time.time()
run_experiment(standard_dataset, current_mpnn_config, log_every=10)
print("Execution time:", (time.time() - start_time) / 60, "minutes")
