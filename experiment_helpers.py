import os
from typing import Any
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


def _iterate_samplers(
    samplers: list[Any], batch_size, keep_lengths_for_n_samples: int | None = None
):
    sampler_index = 0
    count = 0
    while True:
        if (
            keep_lengths_for_n_samples is None
            or count % keep_lengths_for_n_samples == 0
        ):
            sampler_index = np.random.randint(0, len(samplers))
        sampler = samplers[sampler_index]
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
        min_test_length=4,
        keep_lengths_for_n_samples: int | None = None,
    ):
        self.algorithm_name = algorithm_name
        self.num_samples = num_samples
        self.max_train_length = length
        self.train_batch_size = train_batch_size
        if test_batch_size is None:
            test_batch_size = num_test_samples
        self.test_batch_size = test_batch_size
        self.num_test_samples = num_test_samples
        self.test_length = length * test_length_multiplier
        self.train_samplers: list[Any] = []
        self.test_sampler = None
        self.spec = None
        self.min_train_length = min_test_length
        self.keep_lengths_for_n_samples: int | None = keep_lengths_for_n_samples

    def generate_train_samplers(self):
        if len(self.train_samplers) > 0:
            # Already generated
            return
        for i in range(self.min_train_length, self.max_train_length + 1):
            # Generate samplers for each length from min_test_length to length
            sampler, spec = _get_sampler(
                name=self.algorithm_name,
                num_samples=self.num_samples,
                length=i,
            )
            self.train_samplers.append(sampler)
        self.spec = spec
        # if self.train_samplers is not None:
        #     return
        # self.train_samplers, self.spec = _get_sampler(
        #     name=self.algorithm_name,
        #     num_samples=self.num_samples,
        #     length=self.length,
        # )
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
        self.generate_train_samplers()
        return _iterate_samplers(
            self.train_samplers,
            batch_size=self.train_batch_size,
            keep_lengths_for_n_samples=self.keep_lengths_for_n_samples,
        )

    def get_test_sampler(self):
        self.generate_test_sampler()
        return _iterate_sampler(self.test_sampler, batch_size=self.test_batch_size)

    def get_samplers(self):
        return self.get_train_sampler(), self.get_test_sampler()

    def get_spec(self):
        self.generate_train_samplers()
        return self.spec

    def get_dummy_trajectory(self):
        train_sampler = self.get_train_sampler()
        return next(train_sampler)

    from attr import dataclass


@dataclass
class MPNNConfig:
    decoder_learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-3
    encoder_learning_rate: float = 1e-1
    message_weight_lr: float = 1e-1
    message_weight_decay: float = 0.0
    dropout_prob: float = 0.0
    constant_aggregation_weight_init: float | None = (
        0.0  # If None, use random initialization
    )
    aggregation_weights_softmax: bool = True
    disable_jit: bool = False
    differential_messages: bool = False
    # These are not yet settled parameters, but can be used to control the training process
    aggregation_modes: list[AggregationMode] = [
        AggregationMode.MAX,
        AggregationMode.SUM,
        AggregationMode.MOD_SUM,
        AggregationMode.MEAN,
        AggregationMode.MIN,
    ]
    max_steps: int = 4000
    hidden_dim: int = 64
    hint_teacher_forcing: float = 0.0
    gated: bool = False
    use_triplets: bool = False
    modulus_n: float = 2.0
    softmax_temperature: float = 0.5
    mod_steepness: float = 50.0
    nb_msg_passing_steps: int = 1
    turn_off_forcing_at: int | None = None  # If None, never turn off forcing
    msg_weight_gumbel: bool = False  # If True, use Gumbel softmax for message weights
    msg_weight_softmax_temperature: float = 1.0  # Temperature for Gumbel softmax
    n_is_learned: bool = False  # If True, learn the modulus n parameter
    per_node_agg_weights: bool = False  # If True, use per-node aggregation weights


def make_mpnn_processor_factory(
    use_ln: bool,
    aggregation_modes: list[AggregationMode],
    use_triplets: bool,
    nb_triplet_fts: int,
    gated: bool,
    differential_messages: bool,
    aggregation_weights_softmax: bool,
    constant_aggregation_weight_init: float | None,
    modulus_n: float,
    softmax_temperature: float,
    mod_steepness: float,
    msg_weight_gumbel: bool,
    msg_weight_softmax_temperature: float,
    n_is_learned: bool,
    per_node_agg_weights: bool,
):
    def _factory(out_size: int, rngs: nnx.Rngs):
        return processors.MPNN(
            out_size=out_size,
            msgs_mlp_sizes=[out_size, out_size],
            use_ln=use_ln,
            use_triplets=use_triplets,
            nb_triplet_fts=nb_triplet_fts,
            gated=gated,
            rngs=rngs,
            reduction_modes=aggregation_modes,
            differential_messages=differential_messages,
            aggregation_weight_softmax=aggregation_weights_softmax,
            constant_aggregation_weight_init=constant_aggregation_weight_init,
            modulus_n=modulus_n,
            softmax_temperature=softmax_temperature,
            mod_steepness=mod_steepness,
            msg_weight_gumbel=msg_weight_gumbel,
            msg_weight_softmax_temperature=msg_weight_softmax_temperature,
            n_is_learnable=n_is_learned,
            per_node_agg_weights=per_node_agg_weights,
        )

    return _factory


def make_mpnn_model(mpnn_config: MPNNConfig, dataset: DatasetConfig):
    mpnn_processor_factory = make_mpnn_processor_factory(
        use_ln=True,
        aggregation_modes=mpnn_config.aggregation_modes,
        use_triplets=mpnn_config.use_triplets,
        nb_triplet_fts=mpnn_config.hidden_dim,
        gated=mpnn_config.gated,
        differential_messages=mpnn_config.differential_messages,
        aggregation_weights_softmax=mpnn_config.aggregation_weights_softmax,
        constant_aggregation_weight_init=mpnn_config.constant_aggregation_weight_init,
        modulus_n=mpnn_config.modulus_n,
        softmax_temperature=mpnn_config.softmax_temperature,
        mod_steepness=mpnn_config.mod_steepness,
        msg_weight_gumbel=mpnn_config.msg_weight_gumbel,
        msg_weight_softmax_temperature=mpnn_config.msg_weight_softmax_temperature,
        n_is_learned=mpnn_config.n_is_learned,
        per_node_agg_weights=mpnn_config.per_node_agg_weights,
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
        nb_msg_passing_steps=mpnn_config.nb_msg_passing_steps,
    )
    return mpnn_model


def _initialize_wandb(
    mpnn_config: MPNNConfig,
    dataset: DatasetConfig,
    log_every,
    experiment_name=None,
    project_name=None,
):
    if project_name is None:
        project_name = f"{dataset.algorithm_name}-mpnn"
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
        "min_train_graph_size": dataset.min_train_length,
        "max_train_graph_size": dataset.max_train_length,
        "disable_jit": mpnn_config.disable_jit,
        "device_kind": jax.devices()[-1].device_kind,
        "test_length": dataset.test_length,
        "log_every": log_every,
        "hint_teacher_forcing": mpnn_config.hint_teacher_forcing,
        "hidden_dim": mpnn_config.hidden_dim,
        "dropout_prob": mpnn_config.dropout_prob,
        "use_triplets": mpnn_config.use_triplets,
        "gated": mpnn_config.gated,
        "differential_messages": mpnn_config.differential_messages,
        "aggregation_weights_softmax": mpnn_config.aggregation_weights_softmax,
        "message_weight_lr": mpnn_config.message_weight_lr,
        "constant_aggregation_weight_init": mpnn_config.constant_aggregation_weight_init,
        "modulus_n": mpnn_config.modulus_n,
        "softmax_temperature": mpnn_config.softmax_temperature,
        "mod_steepness": mpnn_config.mod_steepness,
        "nb_msg_passing_steps": mpnn_config.nb_msg_passing_steps,
        "turn_off_forcing_at": mpnn_config.turn_off_forcing_at,
        "msg_weight_gumbel": mpnn_config.msg_weight_gumbel,
        "msg_weight_softmax_temperature": mpnn_config.msg_weight_softmax_temperature,
        "n_is_learned": mpnn_config.n_is_learned,
        "per_node_agg_weights": mpnn_config.per_node_agg_weights,
        "keep_lengths_for_n_samples": dataset.keep_lengths_for_n_samples,
    }
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=config,
    )


import jax.numpy as jnp


# Initialize a new W&B run at the start of the notebook
def evaluate_model(
    model,
    val_feedback,
    test_feedback,
    step,
    rng_key,
    cur_loss,
    grad_magnitudes,
    log_to_wandb=True,
    softmax_temperature=1.0,
    per_node_agg_weights=False,
):
    predictions_val, _ = model.predict(rng_key, val_feedback.features)
    out_val = clrs.evaluate(val_feedback.outputs, predictions_val)
    predictions, _ = model.predict(rng_key, test_feedback.features)
    out = clrs.evaluate(test_feedback.outputs, predictions)

    val_acc = out_val["score"]
    test_acc = out["score"]
    decoder_magnitude = grad_magnitudes[baselines.DECODER_LABEL]
    encoder_magnitude = grad_magnitudes[baselines.ENCODER_LABEL]
    processor_magnitude = grad_magnitudes[baselines.PROCESSOR_LABEL]
    if per_node_agg_weights:
        messages_magnitude = 0
    else:
        messages_magnitude = grad_magnitudes[baselines.MESSAGE_LABEL]

    intermediates = nnx.pop(model.net.processor, nnx.Intermediate)
    message_weights = intermediates["mean_message_weights"].value
    message_weights = [float(val) for val in message_weights]
    message_weights_dict = {
        f"message_weights_{i}": val for i, val in enumerate(message_weights)
    }

    if log_to_wandb:
        wandb.log(
            {
                "loss": float(cur_loss),  # training loss
                "val_acc": float(val_acc),  # validation accuracy
                "test_acc": float(test_acc),  # test accuracy
                "decoder_magnitude": float(
                    decoder_magnitude
                ),  # decoder gradient magnitude
                "encoder_magnitude": float(
                    encoder_magnitude
                ),  # encoder gradient magnitude
                "processor_magnitude": float(
                    processor_magnitude
                ),  # processor gradient magnitude
                "messages_magnitude": float(
                    messages_magnitude
                ),  # messages gradient magnitude
                **message_weights_dict,  # message weights
            },
            step=step,
        )

    print(
        f"step = {step} | loss = {cur_loss} | val_acc = {out_val['score']} | test_acc = {out['score']}"
    )
    print(
        " | ".join(
            [f"{key} = {value:.4f}" for key, value in message_weights_dict.items()]
        )  # message weights
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


from flax import serialization


def save_to_wandb(step, data, data_name):
    ckpt_bytes = serialization.to_bytes(
        data
    )  # Produces a bytes object [oai_citation:2‡flax-linen.readthedocs.io](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.serialization.html#:~:text=Save%20optimizer%20or%20other%20object,dict)
    filename = f"{data_name}_checkpoint_step{step}.msgpack"
    with open(filename, "wb") as f:
        f.write(ckpt_bytes)  # write the bytes from serialization.to_bytes

    # Create a W&B artifact and add the file
    artifact = wandb.Artifact(name=f"{data_name}-model-checkpoint", type="model")
    artifact.add_file(filename)  # attach the checkpoint file to the artifact
    # (You could also use artifact.new_file() to write bytes directly without a temp file)

    # Log the artifact to W&B
    wandb.log_artifact(artifact, aliases=[f"step_{step}", "latest"])


def train_model(
    model: baselines.BaselineModel,
    train_sampler,
    test_sampler,
    optimizer: baselines.BaselineOptimizer,
    train_step=None,
    max_steps=1000,
    log_every=10,
    turn_off_forcing_at=None,
    softmax_temperature=1.0,
    per_node_agg_weights=False,
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
        cur_loss, grad_magnitudes = train_step(
            model=model,
            feedback=feedback,
            optimizer=optimizer,
            rng_key=rng_key,
        )
        rng_key = new_rng_key
        if step % log_every == 0:
            evaluate_model(
                model,
                feedback,
                test_feedback,
                step,
                rng_key,
                cur_loss,
                grad_magnitudes,
                softmax_temperature=softmax_temperature,
                per_node_agg_weights=per_node_agg_weights,
            )
        if step % (log_every * 10) == 0:
            _, param_state = nnx.split(model)
            params = nnx.to_pure_dict(param_state)
            save_to_wandb(step, params, "params")
            save_to_wandb(step, optimizer.state, "optimizer_state")

        if turn_off_forcing_at is not None and step == turn_off_forcing_at:
            model.net._hint_teacher_forcing = 0.0
            print(f"Turning off hint teacher forcing at step {step}")

        step += 1


def run_experiment(
    dataset: DatasetConfig,
    mpnn_config: MPNNConfig,
    log_every=10,
    experiment_name=None,
    project_name=None,
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
            message_weight_lr=mpnn_config.message_weight_lr,
        )

        _initialize_wandb(
            mpnn_config,
            dataset,
            log_every=log_every,
            experiment_name=experiment_name,
            project_name=project_name,
        )

        train_model(
            model=mpnn_model,
            train_sampler=train_sampler,
            test_sampler=test_sampler,
            optimizer=optimizer,
            max_steps=mpnn_config.max_steps,
            log_every=log_every,
            turn_off_forcing_at=mpnn_config.turn_off_forcing_at,
            softmax_temperature=mpnn_config.msg_weight_softmax_temperature,
            per_node_agg_weights=mpnn_config.per_node_agg_weights,
        )
        wandb.finish()
