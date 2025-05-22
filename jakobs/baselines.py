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

# pytype: disable=signature-mismatch


# def _maybe_pick_first_pmapped(tree):
#     if jax.local_device_count() == 1:
#         return tree
#     return jax.tree_util.tree_map(lambda x: x[0], tree)


# @jax.jit
# def _restack_from_pmap(tree):
#     """Stack the results of a pmapped computation across the first two axes."""
#     restack_array = lambda x: jnp.reshape(x, (-1,) + x.shape[2:])
#     return jax.tree_util.tree_map(restack_array, tree)


# def _maybe_restack_from_pmap(tree):
#     if jax.local_device_count() == 1:
#         return tree
#     return _restack_from_pmap(tree)


# @functools.partial(jax.jit, static_argnums=[1, 2])
# def _pmap_reshape(x, n_devices, split_axis=0):
#     """Splits a pytree over n_devices on axis split_axis for pmapping."""

#     def _reshape(arr):
#         new_shape = (
#             arr.shape[:split_axis]
#             + (n_devices, arr.shape[split_axis] // n_devices)
#             + arr.shape[split_axis + 1 :]
#         )
#         return jnp.moveaxis(jnp.reshape(arr, new_shape), split_axis, 0)

#     return jax.tree_util.tree_map(_reshape, x)


# def _maybe_pmap_reshape(x, split_axis=0):
#     n_devices = jax.local_device_count()
#     if n_devices == 1:
#         return x
#     return _pmap_reshape(x, n_devices, split_axis)


# @functools.partial(jax.jit, static_argnums=1)
# def _pmap_data(data: Union[_Feedback, _Features], n_devices: int):
#     """Replicate/split feedback or features for pmapping."""
#     if isinstance(data, _Feedback):
#         features = data.features
#     else:
#         features = data
#     pmap_data = features._replace(
#         inputs=_pmap_reshape(features.inputs, n_devices),
#         hints=_pmap_reshape(features.hints, n_devices, split_axis=1),
#         lengths=_pmap_reshape(features.lengths, n_devices),
#     )
#     if isinstance(data, _Feedback):
#         pmap_data = data._replace(
#             features=pmap_data, outputs=_pmap_reshape(data.outputs, n_devices)
#         )
#     return pmap_data


# def _maybe_pmap_data(data: Union[_Feedback, _Features]):
#     n_devices = jax.local_device_count()
#     if n_devices == 1:
#         return data
#     return _pmap_data(data, n_devices)


# def _maybe_put_replicated(tree):
#     if jax.local_device_count() == 1:
#         return jax.device_put(tree)
#     else:
#         return jax.device_put_replicated(tree, jax.local_devices())


# def _maybe_pmap_rng_key(rng_key: _Key):
#     n_devices = jax.local_device_count()
#     if n_devices == 1:
#         return rng_key
#     pmap_rng_keys = jax.random.split(rng_key, n_devices)
#     return jax.device_put_sharded(list(pmap_rng_keys), jax.local_devices())


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
            output_preds, hint_preds, _ = outputs
        else:
            output_preds, hint_preds = outputs

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
            outs, hint_preds = net_outputs
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
            return outs, hint_preds

    # def get_optimizer(self):
    #     graphdef, params_state = nnx.split(self)
    #     params = nnx.to_pure_dict(params_state)

    #     optimizers = {
    #         "backbone": optax.adam(learning_rate=1e-2),
    #         "decoder": optax.adam(learning_rate=1e-5),
    #     }
    #     param_labels = traverse_util.path_aware_map(
    #         lambda path, _: "decoder"
    #         if "decoders" in path
    #         else "backbone",
    #         params,
    #     )
    #     # for path, label in flax.traverse_util.flatten_dict(
    #     #     param_labels
    #     # ).items():
    #     #     print(path, "->", label)
    #     multi_tx = optax.multi_transform(
    #         optimizers, param_labels
    #     )
    #     opt_state = multi_tx.init(params)

    #     return multi_tx, opt_state, params, graphdef

    # def update_model_params(self, params):
    #     pass

    def get_params(self):
        _, params_state = nnx.split(self)
        params = nnx.to_pure_dict(params_state)
        return params

    def update_model_params(self, params):
        nnx.update(self, params)

    # def init(self, features: Union[_Features, List[_Features]], seed: _Seed):
    #     if not isinstance(features, list):
    #         assert len(self._spec) == 1
    #         features = [features]
    #     self.params = self.net_fn.init(
    #         jax.random.PRNGKey(seed),
    #         features,
    #         True,  # pytype: disable=wrong-arg-types  # jax-ndarray
    #         algorithm_index=-1,
    #         return_hints=False,
    #         return_all_outputs=False,
    #     )
    #     self.opt_state = self.opt.init(self.params)
    #     # We will use the optimizer state skeleton for traversal when we
    #     # want to avoid updating the state of params of untrained algorithms.
    #     self.opt_state_skeleton = self.opt.init(jnp.zeros(1))

    # @property
    # def params(self):
    #     if self._device_params is None:
    #         return None
    #     return jax.device_get(_maybe_pick_first_pmapped(self._device_params))

    # @params.setter
    # def params(self, params):
    #     self._device_params = _maybe_put_replicated(params)

    # @property
    # def opt_state(self):
    #     if self._device_opt_state is None:
    #         return None
    #     return jax.device_get(_maybe_pick_first_pmapped(self._device_opt_state))

    # @opt_state.setter
    # def opt_state(self, opt_state):
    #     self._device_opt_state = _maybe_put_replicated(opt_state)

    # def _compute_grad(self, params, rng_key, feedback, algorithm_index):
    #     lss, grads = jax.value_and_grad(self._loss)(
    #         params, rng_key, feedback, algorithm_index
    #     )
    #     return self._maybe_pmean(lss), self._maybe_pmean(grads)

    # def _feedback(self, params, rng_key, feedback, opt_state, algorithm_index):
    #     lss, grads = jax.value_and_grad(self._loss)(
    #         params, rng_key, feedback, algorithm_index
    #     )
    #     params, opt_state = self._update_params(
    #         params, grads, opt_state, algorithm_index
    #     )
    #     lss = self._maybe_pmean(lss)
    #     return lss, params, opt_state

    # def compute_grad(
    #     self,
    #     rng_key: _Key,
    #     feedback: _Feedback,
    #     algorithm_index: Optional[int] = None,
    # ) -> Tuple[float, _Array]:
    #     """Compute gradients."""

    #     if algorithm_index is None:
    #         assert len(self._spec) == 1
    #         algorithm_index = 0
    #     assert algorithm_index >= 0

    #     # Calculate gradients.
    #     rng_keys = _maybe_pmap_rng_key(
    #         rng_key
    #     )  # pytype: disable=wrong-arg-types  # numpy-scalars
    #     feedback = _maybe_pmap_data(feedback)
    #     loss, grads = self.jitted_grad(
    #         self._device_params, rng_keys, feedback, algorithm_index
    #     )
    #     loss = _maybe_pick_first_pmapped(loss)
    #     grads = _maybe_pick_first_pmapped(grads)

    #     return loss, grads

    # def _update_params(self, params, grads, opt_state, algorithm_index):
    #     updates, opt_state = filter_null_grads(
    #         grads, self.opt, opt_state, self.opt_state_skeleton, algorithm_index
    #     )
    #     if self._freeze_processor:
    #         params_subset = _filter_out_processor(params)
    #         updates_subset = _filter_out_processor(updates)
    #         assert len(params) > len(params_subset)
    #         assert params_subset
    #         new_params = optax.apply_updates(params_subset, updates_subset)
    #         new_params = hk.data_structures.merge(params, new_params)
    #     else:
    #         new_params = optax.apply_updates(params, updates)

    #     return new_params, opt_state

    # def update_model_params_accum(self, grads) -> None:
    #     grads = _maybe_put_replicated(grads)
    #     self._device_params, self._device_opt_state = self.jitted_accum_opt_update(
    #         self._device_params,
    #         grads,
    #         self._device_opt_state,
    #         self.opt,
    #         self._freeze_processor,
    #     )

    # def verbose_loss(self, feedback: _Feedback, extra_info) -> Dict[str, _Array]:
    #     """Gets verbose loss information."""
    #     hint_preds = extra_info

    #     nb_nodes = _nb_nodes(feedback, is_chunked=False)
    #     lengths = feedback.features.lengths
    #     losses_ = {}

    #     # Optionally accumulate hint losses.
    #     if self.decode_hints:
    #         for truth in feedback.features.hints:
    #             losses_.update(
    #                 losses.hint_loss(
    #                     truth=truth,
    #                     preds=[x[truth.name] for x in hint_preds],
    #                     lengths=lengths,
    #                     nb_nodes=nb_nodes,
    #                     verbose=True,
    #                 )
    #             )

    #     return losses_


def _nb_nodes(feedback: _Feedback, is_chunked) -> int:
    for inp in feedback.features.inputs:
        if inp.location in [_Location.NODE, _Location.EDGE]:
            if is_chunked:
                return inp.data.shape[2]  # inputs are time x batch x nodes x ...
            else:
                return inp.data.shape[1]  # inputs are batch x nodes x ...
    assert False


# def _param_in_processor(module_name):
#     return processors.PROCESSOR_TAG in module_name


# def _filter_out_processor(params: hk.Params) -> hk.Params:
#     return hk.data_structures.filter(
#         lambda module_name, n, v: not _param_in_processor(module_name), params
#     )


# def _filter_in_processor(params: hk.Params) -> hk.Params:
#     return hk.data_structures.filter(
#         lambda module_name, n, v: _param_in_processor(module_name), params
#     )


# def accum_opt_update(params, grads, opt_state, opt, freeze_processor):
#     """Update params from gradients collected from several algorithms."""
#     # Average the gradients over all algos
#     grads = jax.tree_util.tree_map(
#         lambda *x: sum(x) / (sum([jnp.any(k) for k in x]) + 1e-12), *grads
#     )
#     updates, opt_state = opt.update(grads, opt_state)
#     if freeze_processor:
#         params_subset = _filter_out_processor(params)
#         assert len(params) > len(params_subset)
#         assert params_subset
#         updates_subset = _filter_out_processor(updates)
#         new_params = optax.apply_updates(params_subset, updates_subset)
#         new_params = hk.data_structures.merge(params, new_params)
#     else:
#         new_params = optax.apply_updates(params, updates)

#     return new_params, opt_state


# @functools.partial(jax.jit, static_argnames=["opt"])
# def opt_update(opt, flat_grads, flat_opt_state):
#     return opt.update(flat_grads, flat_opt_state)


# def filter_null_grads(grads, opt, opt_state, opt_state_skeleton, algo_idx):
#     """Compute updates ignoring params that have no gradients.

#     This prevents untrained params (e.g., encoders/decoders for algorithms
#     that are not being trained) to accumulate, e.g., momentum from spurious
#     zero gradients.

#     Note: this works as intended for "per-parameter" optimizer state, such as
#       momentum. However, when the optimizer has some global state (such as the
#       step counts in Adam), the global state will be updated every time,
#       affecting also future updates of parameters that had null gradients in the
#       current step.

#     Args:
#       grads: Gradients for all parameters.
#       opt: Optax optimizer.
#       opt_state: Optimizer state.
#       opt_state_skeleton: A "skeleton" of optimizer state that has been
#         initialized with scalar parameters. This serves to traverse each parameter
#         of the otpimizer state during the opt state update.
#       algo_idx: Index of algorithm, to filter out unused encoders/decoders.
#         If None, no filtering happens.
#     Returns:
#       Updates and new optimizer state, where the parameters with null gradient
#         have not been taken into account.
#     """

#     def _keep_in_algo(k, v):
#         """Ignore params of encoders/decoders irrelevant for this algo."""
#         # Note: in shared pointer decoder modes, we should exclude shared params
#         #       for algos that do not have pointer outputs.
#         if (processors.PROCESSOR_TAG in k) or (f"algo_{algo_idx}_" in k):
#             return v
#         return jax.tree_util.tree_map(lambda x: None, v)

#     if algo_idx is None:
#         masked_grads = grads
#     else:
#         masked_grads = {k: _keep_in_algo(k, v) for k, v in grads.items()}
#     flat_grads, treedef = jax.tree_util.tree_flatten(
#         masked_grads, is_leaf=lambda x: x is None
#     )
#     flat_opt_state = jax.tree_util.tree_map(
#         lambda _, x: (
#             x  # pylint:disable=g-long-lambda
#             if isinstance(x, (np.ndarray, jax.Array))
#             else treedef.flatten_up_to(x)
#         ),
#         opt_state_skeleton,
#         opt_state,
#     )

#     # Compute updates only for the params with gradient.
#     flat_updates, flat_opt_state = opt_update(opt, flat_grads, flat_opt_state)

#     def unflatten(flat, original):
#         """Restore tree structure, filling missing (None) leaves with original."""
#         if isinstance(flat, (np.ndarray, jax.Array)):
#             return flat
#         return jax.tree_util.tree_map(
#             lambda x, y: x if y is None else y, original, treedef.unflatten(flat)
#         )

#     # Restore the state and updates tree structure.
#     new_opt_state = jax.tree_util.tree_map(
#         lambda _, x, y: unflatten(x, y), opt_state_skeleton, flat_opt_state, opt_state
#     )
#     updates = unflatten(flat_updates, jax.tree_util.tree_map(lambda x: 0.0, grads))
#     return updates, new_opt_state


class BaselineOptimizer:
    def __init__(self, model, backbone_lr: float = 1e-2, decoder_lr: float = 1e-5):
        self.model = model
        graph_def, params_state = nnx.split(model)
        self.graph_def = graph_def
        params = nnx.to_pure_dict(params_state)

        optimizers = {
            "backbone": optax.adam(learning_rate=backbone_lr),
            "decoder": optax.adam(learning_rate=decoder_lr),
        }
        param_labels = traverse_util.path_aware_map(
            lambda path, _: "decoder" if "decoders" in path else "backbone",
            params,
        )

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
        """Apply updates to the model parameters."""
        # params = nnx.to_pure_dict(params_state)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    def _mk_grad_clip_optimizer(self, grad_clip_max_norm: float, learning_rate: float):
        if grad_clip_max_norm > 0:
            optax_chain = [
                optax.clip_by_global_norm(grad_clip_max_norm),
                optax.scale_by_adam(),
                optax.scale(-learning_rate),
            ]
            return optax.chain(*optax_chain)
        else:
            return optax.adam(learning_rate=learning_rate)

    def make_train_step(self):
        graphdef = self.graph_def
        tx = self.tx

        @jax.jit
        def train_step_jit(params, opt_state, feedback, rng_key):
            def loss_fn(params):
                model = nnx.merge(graphdef, params)
                loss = model.feedback(rng_key, feedback)
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(params)
            new_params, new_opt_state = BaselineOptimizer.apply_updates(
                params, grads, opt_state, tx
            )
            return loss, new_params, new_opt_state

        def train_step(model, feedback, optimizer, rng_key):
            params = model.get_params()
            loss, new_params, new_opt_state = train_step_jit(
                params, optimizer.state, feedback, rng_key
            )
            model.update_model_params(new_params)
            optimizer.state = new_opt_state
            return loss

        return train_step
