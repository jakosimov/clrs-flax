import os
import jax

from attr import dataclass
import clrs

from jakobs import processors
from jakobs import baselines
from flax import nnx
import jax.random as random
import numpy as np

import wandb
import pickle
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


test_length_multiplier = 4


class DatasetConfig:
    def __init__(
        self,
        algorithm_name,
        num_samples,
        length,
        train_batch_size,
        num_test_samples,
        test_batch_size=None,
    ):
        self.algorithm_name = algorithm_name
        self.num_samples = num_samples
        self.length = length
        self.train_batch_size = train_batch_size
        if test_batch_size is None:
            test_batch_size = num_test_samples
        self.test_batch_size = test_batch_size
        self.num_test_samples = num_test_samples
        self.test_length = length * test_length_multiplier
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
            num_samples=self.num_test_samples,
            length=self.test_length,
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
    encoder_learning_rate: float = 1e-3
    message_weight_decay: float = 0.0
    max_steps: int = 1000
    disable_jit: bool = False
    hint_teacher_forcing: float = 0.0
    hidden_dim: int = 32
    dropout_prob: float = 0.0
    nb_heads: int = 4


def make_mpnn_model(mpnn_config: MPNNConfig, dataset: DatasetConfig):
    mpnn_processor_factory = processors.get_processor_factory(
        processors.ProcessorKind.MPNN,
        use_ln=True,
        nb_triplet_fts=32,
        nb_heads=mpnn_config.nb_heads,
        reduction=mpnn_config.aggregation_modes,
    )

    rngs = nnx.Rngs(params=10, dropout=random.key(1))
    mpnn_model = baselines.BaselineModel(
        processor_factory=mpnn_processor_factory,
        spec=dataset.get_spec(),
        dummy_trajectory=dataset.get_dummy_trajectory(),
        hidden_dim=mpnn_config.hidden_dim,
        encode_hints=True,
        decode_hints=True,
        use_lstm=False,
        checkpoint_path="/tmp/checkpt",
        freeze_processor=False,
        dropout_prob=mpnn_config.dropout_prob,
        hint_teacher_forcing=mpnn_config.hint_teacher_forcing,
        rngs=rngs,
    )
    return mpnn_model


def _initialize_wandb(
    mpnn_config: MPNNConfig, dataset: DatasetConfig, log_every, experiment_name=None
):
    config = {
        "epochs": mpnn_config.max_steps,
        "train_batch_size": dataset.train_batch_size,
        "test_batch_size": dataset.test_batch_size,
        "decoder_learning_rate": mpnn_config.decoder_learning_rate,
        "processor_learning_rate": mpnn_config.backbone_learning_rate,
        "encoder_learning_rate": mpnn_config.encoder_learning_rate,
        "message_weight_decay": mpnn_config.message_weight_decay,
        "aggregation_modes": [mode.name for mode in mpnn_config.aggregation_modes],
        "algorithm": dataset.algorithm_name,
        "num_training_samples": dataset.num_samples,
        "num_test_samples": dataset.num_test_samples,
        "graph_size": dataset.length,
        "disable_jit": mpnn_config.disable_jit,
        "device_kind": jax.devices()[-1].device_kind,
        "test_length": dataset.test_length,
        "log_every": log_every,
        "hint_teacher_forcing": mpnn_config.hint_teacher_forcing,
        "hidden_dim": mpnn_config.hidden_dim,
        "dropout_prob": mpnn_config.dropout_prob,
        "nb_heads": mpnn_config.nb_heads,
    }
    wandb.init(
        project=f"{dataset.algorithm_name}-mpnn",
        name=experiment_name,
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
    grad_magnitude,
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
            "grad_magnitude": float(grad_magnitude),  # gradient magnitude
        },
        step=step,
    )

    print(
        f"step = {step} | loss = {cur_loss} | val_acc = {out_val['score']} | test_acc = {out['score']} | grad_magnitude = {grad_magnitude}"
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
    log_every=10,
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
        cur_loss, grad_magnitude = train_step(
            model=model,
            feedback=feedback,
            optimizer=optimizer,
            rng_key=rng_key,
        )
        rng_key = new_rng_key
        if step % log_every == 0:
            evaluate_model(
                model, feedback, test_feedback, step, rng_key, cur_loss, grad_magnitude
            )

        step += 1


def run_experiment(
    dataset: DatasetConfig, mpnn_config: MPNNConfig, log_every=10, experiment_name=None
):
    with jax.disable_jit(mpnn_config.disable_jit):
        if mpnn_config.disable_jit:
            print("JIT is disabled")
        train_sampler, test_sampler = dataset.get_samplers()
        mpnn_model = make_mpnn_model(mpnn_config, dataset)
        optimizer = baselines.BaselineOptimizer(
            mpnn_model,
            backbone_lr=mpnn_config.backbone_learning_rate,
            decoder_lr=mpnn_config.decoder_learning_rate,
            encoder_lr=mpnn_config.encoder_learning_rate,
            message_weight_decay=mpnn_config.message_weight_decay,
        )

        _initialize_wandb(
            mpnn_config, dataset, log_every=log_every, experiment_name=experiment_name
        )

        train_model(
            model=mpnn_model,
            train_sampler=train_sampler,
            test_sampler=test_sampler,
            optimizer=optimizer,
            max_steps=mpnn_config.max_steps,
            log_every=log_every,
        )
        wandb.finish()
