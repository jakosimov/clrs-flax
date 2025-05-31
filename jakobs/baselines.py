# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""JAX implementation of CLRS baseline models."""

from typing import List, Optional, Union


from jakobs import decoders
from jakobs import losses
from jakobs import model
from jakobs import nets
from jakobs import processors
from clrs._src import samplers
from clrs._src import specs

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.nnx as nnx

from jax import Array

from jakobs.encoders import EncoderInitialiser
from flax import traverse_util


_Features = samplers.Features
_Feedback = samplers.Feedback
_Location = specs.Location
_Seed = jnp.integer
_Spec = specs.Spec
_Key = Array


def print_value(name, x):
    """Prints the value of x."""
    jax.debug.print(name + ": {x}", x=x)
    return x


class BaselineModel(nnx.Module, model.Model):
    """Model implementation with selectable message passing algorithm."""

    def __init__(
        self,
        spec: Union[_Spec, List[_Spec]],
        dummy_trajectory: Union[List[_Feedback], _Feedback],
        processor_factory: processors.ProcessorFactory,
        rngs: nnx.Rngs,
        hidden_dim: int = 32,
        encode_hints: bool = False,
        decode_hints: bool = True,
        encoder_init: EncoderInitialiser = EncoderInitialiser.DEFAULT,
        use_lstm: bool = False,
        checkpoint_path: str = "/tmp/clrs3",
        freeze_processor: bool = False,
        dropout_prob: float = 0.0,
        hint_teacher_forcing: float = 0.0,
        hint_repred_mode: nets.HintRepredMode = nets.HintRepredMode.SOFT,
        name: str = "base_model",
        nb_msg_passing_steps: int = 1,
        debug: bool = False,
    ):
        """Constructor for BaselineModel.

        The model consists of encoders, processor and decoders. It can train
        and evaluate either a single algorithm or a set of algorithms; in the
        latter case, a single processor is shared among all the algorithms, while
        the encoders and decoders are separate for each algorithm.

        Args:
          spec: Either a single spec for one algorithm, or a list of specs for
            multiple algorithms to be trained and evaluated.
          dummy_trajectory: Either a single feedback batch, in the single-algorithm
            case, or a list of feedback batches, in the multi-algorithm case, that
            comply with the `spec` (or list of specs), to initialize network size.
          processor_factory: A callable that takes an `out_size` parameter
            and returns a processor (see `processors.py`).
          hidden_dim: Size of the hidden state of the model, i.e., size of the
            message-passing vectors.
          encode_hints: Whether to provide hints as model inputs.
          decode_hints: Whether to provide hints as model outputs.
          encoder_init: The initialiser type to use for the encoders.
          use_lstm: Whether to insert an LSTM after message passing.
          learning_rate: Learning rate for training.
          grad_clip_max_norm: if greater than 0, the maximum norm of the gradients.
          checkpoint_path: Path for loading/saving checkpoints.
          freeze_processor: If True, the processor weights will be frozen and
            only encoders and decoders (and, if used, the lstm) will be trained.
          dropout_prob: Dropout rate in the message-passing stage.
          hint_teacher_forcing: Probability of using ground-truth hints instead
            of predicted hints as inputs during training (only relevant if
            `encode_hints`=True)
          hint_repred_mode: How to process predicted hints when fed back as inputs.
            Only meaningful when `encode_hints` and `decode_hints` are True.
            Options are:
              - 'soft', where we use softmaxes for categoricals, pointers
                  and mask_one, and sigmoids for masks. This will allow gradients
                  to flow through hints during training.
              - 'hard', where we use argmax instead of softmax, and hard
                  thresholding of masks. No gradients will go through the hints
                  during training; even for scalar hints, which don't have any
                  kind of post-processing, gradients will be stopped.
              - 'hard_on_eval', which is soft for training and hard for evaluation.
          name: Model name.
          nb_msg_passing_steps: Number of message passing steps per hint.
          debug: If True, the model run in debug mode, outputting all hidden state.

        Raises:
          ValueError: if `encode_hints=True` and `decode_hints=False`.
        """
        super().__init__(spec=spec)

        if encode_hints and not decode_hints:
            raise ValueError("`encode_hints=True`, `decode_hints=False` is invalid.")

        assert hint_repred_mode in ["soft", "hard", "hard_on_eval"]

        self.decode_hints = decode_hints
        self.checkpoint_path = checkpoint_path
        self.name = name
        self._freeze_processor = freeze_processor

        self.nb_msg_passing_steps = nb_msg_passing_steps
        self.debug = debug

        self.nb_dims: list[dict[str, int]] = []
        if isinstance(dummy_trajectory, _Feedback):
            assert len(self._spec) == 1
            dummy_trajectory = [dummy_trajectory]
        for traj in dummy_trajectory:
            nb_dims: dict[str, int] = {}
            for inp in traj.features.inputs:
                nb_dims[inp.name] = inp.data.shape[-1]
            for hint in traj.features.hints:
                nb_dims[hint.name] = hint.data.shape[-1]
            for outp in traj.outputs:
                nb_dims[outp.name] = outp.data.shape[-1]
            self.nb_dims.append(nb_dims)

        self.net = nets.NetFlax(
            spec=self._spec,
            hidden_dim=hidden_dim,
            encode_hints=encode_hints,
            decode_hints=self.decode_hints,
            processor_factory=processor_factory,
            # use_lstm=use_lstm,
            encoder_init=encoder_init,
            dropout_prob=dropout_prob,
            hint_teacher_forcing=hint_teacher_forcing,
            hint_repred_mode=hint_repred_mode,
            nb_dims=self.nb_dims,
            nb_msg_passing_steps=self.nb_msg_passing_steps,
            debug=self.debug,
            rngs=rngs,
        )

        # self._jitted_loss = jax.jit(
        #     self._loss,
        #     static_argnames=["algorithm_index"],
        # )
        # self._jitted_predict = jax.jit(
        #     self._predict,
        #     static_argnames=["algorithm_index", "return_hints", "return_all_outputs"],
        # )

    def predict(
        self,
        rng_key: _Key,
        features: _Features,
        algorithm_index: Optional[int] = None,
        return_hints: bool = False,
        return_all_outputs: bool = False,
    ):
        """Model inference step."""
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0

        return self._predict(
            rng_key=rng_key,
            features=features,
            algorithm_index=algorithm_index,
            return_hints=return_hints,
            return_all_outputs=return_all_outputs,
        )

    def feedback(
        self, rng_key: _Key, feedback: _Feedback, algorithm_index=None
    ) -> float:
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0
        rng_keys = rng_key
        loss = self._loss(
            rng_key=rng_keys,
            feedback=feedback,
            algorithm_index=algorithm_index,
        )
        return loss

    def _loss(self, rng_key, feedback: _Feedback, algorithm_index):
        """Calculates model loss f(feedback; params)."""
        outputs = self.net(
            rng_key=rng_key,
            features_list=[feedback.features],
            repred=False,
            algorithm_index=algorithm_index,
            return_hints=True,
            return_all_outputs=False,
        )
        if self.debug:
            output_preds, hint_preds, _, _ = outputs
        else:
            output_preds, hint_preds, _ = outputs

        nb_nodes = _nb_nodes(feedback, is_chunked=False)
        lengths = feedback.features.lengths
        total_loss = 0.0

        # Calculate output loss.
        for truth in feedback.outputs:
            total_loss += losses.output_loss(
                truth=truth,
                pred=output_preds[truth.name],
                nb_nodes=nb_nodes,
            )

        # Optionally accumulate hint losses.
        if self.decode_hints:
            for truth in feedback.features.hints:
                total_loss += losses.hint_loss(
                    truth=truth,
                    preds=[x[truth.name] for x in hint_preds],
                    lengths=lengths,
                    nb_nodes=nb_nodes,
                )

        return total_loss

    def _predict(
        self,
        rng_key: _Key,
        features: _Features,
        algorithm_index: int,
        return_hints: bool,
        return_all_outputs: bool,
    ):
        net_outputs = self.net(
            features_list=[features],
            repred=True,
            algorithm_index=algorithm_index,
            return_hints=return_hints,
            return_all_outputs=return_all_outputs,
            rng_key=rng_key,
        )
        if self.debug:
            outs, hint_preds, hidden_states = net_outputs
        else:
            outs, hint_preds, mean_message_weights = net_outputs
        outs = decoders.postprocess(
            self._spec[algorithm_index],
            outs,
            sinkhorn_temperature=0.1,
            sinkhorn_steps=50,
            hard=True,
        )
        if self.debug:
            return outs, hint_preds, hidden_states
        else:
            return outs, hint_preds, mean_message_weights

    def get_params(self):
        _, params_state, _ = self.split_model()
        params = nnx.to_pure_dict(params_state)
        return params

    def split_model(self):
        def filter_fn(v, x):
            if "mean_message_weights" in v:
                return False
            if "dropout" in v:
                return False
            return True

        graphdef, params, fixed_state = nnx.split(
            self, filter_fn, lambda v, x: not filter_fn(v, x)
        )
        return graphdef, params, fixed_state

    def update_model_params(self, params):
        nnx.update(self, params)

    def get_graph_def(self):
        graph_def, _, _ = self.split_model()
        return graph_def

    def get_fixed_state(self):
        _, _, fixed_state = self.split_model()
        return fixed_state


def _nb_nodes(feedback: _Feedback, is_chunked) -> int:
    for inp in feedback.features.inputs:
        if inp.location in [_Location.NODE, _Location.EDGE]:
            if is_chunked:
                return inp.data.shape[2]  # inputs are time x batch x nodes x ...
            else:
                return inp.data.shape[1]  # inputs are batch x nodes x ...
    assert False


DECODER_LABEL = "decoders"
PROCESSOR_LABEL = "processor"
ENCODER_LABEL = "encoders"
MESSAGE_LABEL = "message_weights"
MESSAGE_WEIGHTS_MLP = "message_weight_mlp"
TRIPLET_MODULE_LABEL = "triplet_module"


class BaselineOptimizer:
    def __init__(
        self,
        model: BaselineModel,
        backbone_lr: float = 1e-3,
        decoder_lr: float = 1e-3,
        encoder_lr: float = 1e-1,
        message_weight_decay: float = 0.0,
        message_weight_lr: float = 1e-3,
        grad_clip_max_norm: float = 0.0,
    ):
        self.model = model
        graph_def = model.get_graph_def()
        params = model.get_params()
        self.graph_def = graph_def
        self.fixed_state = model.get_fixed_state()

        optimizers = {
            PROCESSOR_LABEL: self._mk_grad_clip_optimizer(
                learning_rate=backbone_lr, grad_clip_max_norm=grad_clip_max_norm
            ),
            DECODER_LABEL: self._mk_grad_clip_optimizer(
                learning_rate=decoder_lr, grad_clip_max_norm=grad_clip_max_norm
            ),
            TRIPLET_MODULE_LABEL: self._mk_grad_clip_optimizer(
                learning_rate=decoder_lr, grad_clip_max_norm=grad_clip_max_norm
            ),
            ENCODER_LABEL: self._mk_grad_clip_optimizer(
                learning_rate=encoder_lr, grad_clip_max_norm=grad_clip_max_norm
            ),
            MESSAGE_LABEL: self._mk_grad_clip_optimizer(
                learning_rate=message_weight_lr,
                weight_decay=message_weight_decay,
                grad_clip_max_norm=grad_clip_max_norm,
            ),
            MESSAGE_WEIGHTS_MLP: self._mk_grad_clip_optimizer(
                learning_rate=message_weight_lr,
                weight_decay=message_weight_decay,
                grad_clip_max_norm=grad_clip_max_norm,
            ),
        }

        param_labels = traverse_util.path_aware_map(
            lambda path, _: (
                DECODER_LABEL
                if DECODER_LABEL in path
                else (
                    ENCODER_LABEL
                    if ENCODER_LABEL in path
                    else (
                        MESSAGE_LABEL
                        if MESSAGE_LABEL in path
                        else (
                            MESSAGE_WEIGHTS_MLP
                            if MESSAGE_WEIGHTS_MLP in path
                            else (
                                TRIPLET_MODULE_LABEL
                                if TRIPLET_MODULE_LABEL in path
                                else PROCESSOR_LABEL
                            )
                        )
                    )
                )
            ),
            params,
        )

        # for path, label in traverse_util.flatten_dict(param_labels).items():
        #     print(path, "->", label)

        self.param_labels = {
            "/".join(map(str, path)): label
            for path, label in traverse_util.flatten_dict(param_labels).items()
        }

        self.tx: optax.GradientTransformationExtraArgs = optax.multi_transform(
            optimizers, param_labels
        )
        self.state = self.tx.init(params)

    def update(self, grads_state):
        grads = grads_state.to_pure_dict()
        params = self.model.get_params()
        updates, opt_state = self.tx.update(grads, self.state, params)
        new_params = optax.apply_updates(params, updates)
        self.model.update_model_params(new_params)
        self.state = opt_state

    @staticmethod
    def apply_updates(params, grads, opt_state, tx):
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    def _mk_grad_clip_optimizer(
        self, learning_rate: float, grad_clip_max_norm: float = 1.0, weight_decay=0.0
    ):
        optax_chain = []
        if weight_decay > 0:
            optax_chain.append(optax.add_decayed_weights(weight_decay))
        if grad_clip_max_norm > 0:
            optax_chain.append(optax.clip_by_global_norm(grad_clip_max_norm))
        optax_chain.append(optax.scale_by_adam())
        optax_chain.append(optax.scale(-learning_rate))
        return optax.chain(*optax_chain)

    def _compute_grad_magnitudes(self, grads):
        grad_sums = {label: 0.0 for label in set(self.param_labels.values())}
        grad_counts = {label: 0 for label in set(self.param_labels.values())}
        flat_grads = {
            "/".join(map(str, path)): value
            for path, value in traverse_util.flatten_dict(grads).items()
        }
        for path, grad in flat_grads.items():
            label = self.param_labels.get(path, "processor")
            if grad is not None:
                grad_norm = jnp.linalg.norm(jnp.ravel(grad))
                grad_sums[label] += grad_norm
                grad_counts[label] += 1

        # Compute mean gradient magnitude per group
        grad_means = {
            label: (
                grad_sums[label] / grad_counts[label] if grad_counts[label] > 0 else 0.0
            )
            for label in grad_sums
        }

        return grad_means

    def make_train_step(self):
        graphdef = self.graph_def
        fixed_state = self.fixed_state
        tx = self.tx
        compute_grad_magnitudes = self._compute_grad_magnitudes

        def train_step_f(params, opt_state, feedback, rng_key):
            def loss_fn(params):
                try:
                    model = nnx.merge(graphdef, params, fixed_state)
                except:
                    import jax.tree_util as tu

                    print(tu.tree_structure(params))
                    print(tu.tree_structure(graphdef))
                # model = nnx.merge(graphdef, params)
                loss = model.feedback(rng_key, feedback)
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(params)
            new_params, new_opt_state = BaselineOptimizer.apply_updates(
                params, grads, opt_state, tx
            )
            grad_mags = compute_grad_magnitudes(grads)
            return loss, grad_mags, new_params, new_opt_state

        train_step_jit = jax.jit(train_step_f)

        def train_step(model, feedback, optimizer, rng_key):
            params = model.get_params()
            loss, grad_mags, new_params, new_opt_state = train_step_jit(
                params, optimizer.state, feedback, rng_key
            )
            model.update_model_params(new_params)
            optimizer.state = new_opt_state
            return loss, grad_mags

        return train_step
