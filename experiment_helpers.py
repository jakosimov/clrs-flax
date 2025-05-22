from attr import dataclass
import clrs
import jax

from jakobs import processors
from jakobs import baselines
from flax import nnx
import jax.random as random
import numpy as np

import wandb
import pickle
import os
from jakobs.processors import AggregationMode


def _iterate_sampler(sampler, batch_size):
    while True:
        yield sampler.next(batch_size)


SAMPLER_DIR = "samplers"


def _get_sampler(name, num_samples, length):
    filename = f"sampler_{name}_{num_samples}_{length}.data"
    path = os.path.join(SAMPLER_DIR, filename)
    print("Checking for sampler at", path)

    if not os.path.exists(SAMPLER_DIR):
        os.makedirs(SAMPLER_DIR)
        print("Created directory for samplers at", SAMPLER_DIR)

    if os.path.exists(path):
        with open(path, "rb") as f:
            sampler, spec = pickle.load(f)
            print(
                "Loaded sampler for algorithm:",
                name,
                "num_samples:",
                num_samples,
                "length:",
                length,
            )
        return sampler, spec

    sampler, spec = clrs.build_sampler(
        name=name,
        num_samples=num_samples,
        length=length,
    )
    with open(path, "wb") as f:
        pickle.dump((sampler, spec), f)
    print(
        "Generated sampler for algorithm:",
        name,
        "num_samples:",
        num_samples,
        "length:",
        length,
    )
    return sampler, spec


class DatasetConfig:
    def __init__(
        self, algorithm_name, num_samples, length, train_batch_size, test_batch_size
    ):
        self.algorithm_name = algorithm_name
        self.num_samples = num_samples
        self.length = length
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_sampler = None
        self.test_sampler = None

    def generate_train_sampler(self):
        if self.train_sampler is not None:
            return
        self.train_sampler, self.spec = _get_sampler(
            name=self.algorithm_name,
            num_samples=self.num_samples,
            length=self.length,
        )
        # print(
        #     "Generated train sampler for algorithm:",
        #     self.algorithm_name,
        #     "num_samples:",
        #     self.num_samples,
        #     "length:",
        #     self.length,
        # )

    def generate_test_sampler(self):
        if self.test_sampler is not None:
            return
        self.test_sampler, _ = _get_sampler(
            name=self.algorithm_name,
            num_samples=self.test_batch_size,
            length=self.length * 4,
        )
        # print(
        #     "Generated test sampler for algorithm:",
        #     self.algorithm_name,
        #     "test_samples:",
        #     self.test_batch_size,
        #     "test length:",
        #     4 * self.length,
        # )

    def get_train_sampler(self):
        self.generate_train_sampler()
        return _iterate_sampler(self.train_sampler, batch_size=self.train_batch_size)

    def get_test_sampler(self):
        self.generate_test_sampler()
        return _iterate_sampler(self.test_sampler, batch_size=self.test_batch_size)

    def get_samplers(self):
        return self.get_train_sampler(), self.get_test_sampler()

    def get_spec(self):
        self.generate_train_sampler()
        return self.spec

    def get_dummy_trajectory(self):
        train_sampler = self.get_train_sampler()
        return next(train_sampler)

    from attr import dataclass


@dataclass
class MPNNConfig:
    aggregation_modes: list[AggregationMode] = [AggregationMode.MAX]
    decoder_learning_rate: float = 1e-5
    backbone_learning_rate: float = 1e-3
    max_steps: int = 1000


def make_mpnn_model(mpnn_config, dataset):
    mpnn_processor_factory = processors.get_processor_factory(
        processors.ProcessorKind.MPNN,
        use_ln=True,
        nb_triplet_fts=32,
        nb_heads=4,
        reduction=mpnn_config.aggregation_modes,
    )

    mpnn_model_params = dict(
        processor_factory=mpnn_processor_factory,
        hidden_dim=32,
        encode_hints=True,
        decode_hints=True,
        use_lstm=False,
        checkpoint_path="/tmp/checkpt",
        freeze_processor=False,
        dropout_prob=0.0,
    )

    rngs = nnx.Rngs(params=0, dropout=random.key(1))
    mpnn_model = baselines.BaselineModel(
        spec=dataset.get_spec(),
        dummy_trajectory=dataset.get_dummy_trajectory(),
        rngs=rngs,
        **mpnn_model_params,
    )
    return mpnn_model


def _initialize_wandb(mpnn_config: MPNNConfig, dataset: DatasetConfig):
    config = {
        "epochs": mpnn_config.max_steps,
        "batch_size": dataset.train_batch_size,
        "decoder_learning_rate": mpnn_config.decoder_learning_rate,
        "backbone_learning_rate": mpnn_config.backbone_learning_rate,
        "aggregation_modes": [mode.name for mode in mpnn_config.aggregation_modes],
        "algorithm": dataset.algorithm_name,
        "num_training_samples": dataset.num_samples,
        "num_test_samples": dataset.test_batch_size,
        "graph_size": dataset.length,
    }
    wandb.init(
        project=f"{dataset.algorithm_name}-mpnn",
        config=config,
    )


# Initialize a new W&B run at the start of the notebook
def evaluate_model(
    model,
    val_feedback,
    test_feedback,
    step,
    rng_key,
    cur_loss,
):
    predictions_val, _ = model.predict(rng_key, val_feedback.features)
    out_val = clrs.evaluate(val_feedback.outputs, predictions_val)
    predictions, _ = model.predict(rng_key, test_feedback.features)
    out = clrs.evaluate(test_feedback.outputs, predictions)

    val_acc = out_val["score"]
    test_acc = out["score"]
    wandb.log(
        {
            "loss": float(cur_loss),  # training loss
            "val_acc": float(val_acc),  # validation accuracy
            "test_acc": float(test_acc),  # test accuracy
        },
        step=step,
    )

    print(
        f"step = {step} | loss = {cur_loss} | val_acc = {out_val['score']} | test_acc = {out['score']}"
    )


def _old_train_step(
    model,
    feedback,
    optimizer,
    rng_key,
):
    def loss_fn(model):
        return model.feedback(rng_key, feedback)

    cur_loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return cur_loss


def train_model(
    model,
    train_sampler,
    test_sampler,
    optimizer,
    train_step=None,
    max_steps=1000,
):
    if train_step is None:
        train_step = optimizer.make_train_step()
    rng = np.random.RandomState(1234)
    rng_key = jax.random.PRNGKey(rng.randint(2**32))
    step = 0
    while step <= max_steps:
        feedback, test_feedback = (
            next(train_sampler),
            next(test_sampler),
        )
        rng_key, new_rng_key = jax.random.split(rng_key)
        cur_loss = train_step(
            model=model,
            feedback=feedback,
            optimizer=optimizer,
            rng_key=rng_key,
        )
        rng_key = new_rng_key
        if step % 10 == 0:
            evaluate_model(
                model,
                feedback,
                test_feedback,
                step,
                rng_key,
                cur_loss,
            )

        step += 1


def run_experiment(dataset, mpnn_config):
    train_sampler, test_sampler = dataset.get_samplers()
    mpnn_model = make_mpnn_model(mpnn_config, dataset)
    optimizer = baselines.BaselineOptimizer(
        mpnn_model,
        backbone_lr=mpnn_config.backbone_learning_rate,
        decoder_lr=mpnn_config.decoder_learning_rate,
    )

    _initialize_wandb(mpnn_config, dataset)

    train_model(
        model=mpnn_model,
        train_sampler=train_sampler,
        test_sampler=test_sampler,
        optimizer=optimizer,
        max_steps=mpnn_config.max_steps,
    )
    wandb.finish()


if __name__ == "__main__":
    algorithm = "matrix_chain_order"
    DEBUG_DATASET = True

    debug_dataset = DatasetConfig(
        algorithm_name=algorithm,
        num_samples=50,
        length=8,
        train_batch_size=8,
        test_batch_size=50,
    )
    standard_dataset = DatasetConfig(
        algorithm_name=algorithm,
        num_samples=1000,
        length=16,
        train_batch_size=8,
        test_batch_size=100,
    )
    if DEBUG_DATASET:
        current_dataset = debug_dataset
    else:
        current_dataset = standard_dataset

    current_mpnn_config = MPNNConfig(
        aggregation_modes=[AggregationMode.MAX, AggregationMode.SUM],
        decoder_learning_rate=1e-5,
        backbone_learning_rate=1e-3,
        max_steps=1000,
    )

    run_experiment(current_dataset, current_mpnn_config)
