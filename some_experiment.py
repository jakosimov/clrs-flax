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
    num_samples=100,
    length=16,
    train_batch_size=8,
    test_batch_size=50,
)

current_mpnn_config = MPNNConfig(
    aggregation_modes=[AggregationMode.SUM],
    decoder_learning_rate=1e-5,
    backbone_learning_rate=1e-3,
    max_steps=100,
)

debug_dataset = DatasetConfig(
    algorithm_name=algorithm,
    num_samples=50,
    length=8,
    train_batch_size=16,
    test_batch_size=50,
)

standard_dataset = debug_dataset
print(jax.devices())
print(jax.disable_jit(False))
import time

start_time = time.time()
run_experiment(standard_dataset, current_mpnn_config)
print("Execution time:", time.time() - start_time)
