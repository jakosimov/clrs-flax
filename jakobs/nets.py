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

"""JAX implementation of CLRS basic network."""

import functools

from typing import Dict, List, Optional, Tuple

import chex

from jakobs import decoders
from jakobs import encoders
from clrs._src import probing
from jakobs import processors
from clrs._src import samplers
from clrs._src import specs

import haiku as hk
import jax
import jax.numpy as jnp


_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = samplers.Features
_FeaturesChunked = samplers.FeaturesChunked
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type


@chex.dataclass
class _MessagePassingScanState:
  hint_preds: chex.Array
  output_preds: chex.Array
  hiddens: chex.Array
  lstm_state: Optional[hk.LSTMState]


@chex.dataclass
class _MessagePassingOutputChunked:
  hint_preds: chex.Array
  output_preds: chex.Array


@chex.dataclass
class MessagePassingStateChunked:
  inputs: chex.Array
  hints: chex.Array
  is_first: chex.Array
  hint_preds: chex.Array
  hiddens: chex.Array
  lstm_state: Optional[hk.LSTMState]



class Net(hk.Module):
  """Building blocks (networks) used to encode and decode messages."""

  def __init__(
      self,
      spec: List[_Spec],
      hidden_dim: int,
      encode_hints: bool,
      decode_hints: bool,
      processor_factory: processors.ProcessorFactory,
      use_lstm: bool,
      encoder_init: str,
      dropout_prob: float,
      hint_teacher_forcing: float,
      hint_repred_mode='soft',
      nb_dims=None,
      nb_msg_passing_steps=1,
      debug=False,
      name: str = 'net',
  ):
    """Constructs a `Net`."""
    super().__init__(name=name)

    self._dropout_prob = dropout_prob
    self._hint_teacher_forcing = hint_teacher_forcing
    self._hint_repred_mode = hint_repred_mode
    self.spec = spec
    self.hidden_dim = hidden_dim
    self.encode_hints = encode_hints
    self.decode_hints = decode_hints
    self.processor_factory = processor_factory
    self.nb_dims = nb_dims
    self.use_lstm = use_lstm
    self.encoder_init = encoder_init
    self.nb_msg_passing_steps = nb_msg_passing_steps
    self.debug = debug

  def _msg_passing_step(self,
                        mp_state: _MessagePassingScanState,
                        i: int,
                        hints: List[_DataPoint],
                        repred: bool,
                        lengths: chex.Array,
                        batch_size: int,
                        nb_nodes: int,
                        inputs: _Trajectory,
                        first_step: bool,
                        spec: _Spec,
                        encs: Dict[str, List[hk.Module]],
                        decs: Dict[str, Tuple[hk.Module]],
                        return_hints: bool,
                        return_all_outputs: bool
                        ):
    if self.decode_hints and not first_step:
      assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
      hard_postprocess = (self._hint_repred_mode == 'hard' or
                          (self._hint_repred_mode == 'hard_on_eval' and repred))
      decoded_hint = decoders.postprocess(spec,
                                          mp_state.hint_preds,
                                          sinkhorn_temperature=0.1,
                                          sinkhorn_steps=25,
                                          hard=hard_postprocess)
    if repred and self.decode_hints and not first_step:
      cur_hint = []
      for hint in decoded_hint:
        cur_hint.append(decoded_hint[hint])
    else:
      cur_hint = []
      needs_noise = (self.decode_hints and not first_step and
                     self._hint_teacher_forcing < 1.0)
      if needs_noise:
        # For noisy teacher forcing, choose which examples in the batch to force
        force_mask = jax.random.bernoulli(
            hk.next_rng_key(), self._hint_teacher_forcing,
            (batch_size,))
      else:
        force_mask = None
      for hint in hints:
        hint_data = jnp.asarray(hint.data)[i]
        _, loc, typ = spec[hint.name]
        if needs_noise:
          if (typ == _Type.POINTER and
              decoded_hint[hint.name].type_ == _Type.SOFT_POINTER):
            # When using soft pointers, the decoded hints cannot be summarised
            # as indices (as would happen in hard postprocessing), so we need
            # to raise the ground-truth hint (potentially used for teacher
            # forcing) to its one-hot version.
            hint_data = hk.one_hot(hint_data, nb_nodes)
            typ = _Type.SOFT_POINTER
          hint_data = jnp.where(_expand_to(force_mask, hint_data),
                                hint_data,
                                decoded_hint[hint.name].data)
        cur_hint.append(
            probing.DataPoint(
                name=hint.name, location=loc, type_=typ, data=hint_data))

    hiddens, output_preds_cand, hint_preds, lstm_state = self._one_step_pred(
        inputs, cur_hint, mp_state.hiddens,
        batch_size, nb_nodes, mp_state.lstm_state,
        spec, encs, decs, repred)

    if first_step:
      output_preds = output_preds_cand
    else:
      output_preds = {}
      for outp in mp_state.output_preds:
        is_not_done = _is_not_done_broadcast(lengths, i,
                                             output_preds_cand[outp])
        output_preds[outp] = is_not_done * output_preds_cand[outp] + (
            1.0 - is_not_done) * mp_state.output_preds[outp]

    new_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
        hint_preds=hint_preds,
        output_preds=output_preds,
        hiddens=hiddens,
        lstm_state=lstm_state)
    # Save memory by not stacking unnecessary fields
    accum_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
        hint_preds=hint_preds if return_hints else None,
        output_preds=output_preds if return_all_outputs else None,
        hiddens=hiddens if self.debug else None, lstm_state=None)

    # Complying to jax.scan, the first returned value is the state we carry over
    # the second value is the output that will be stacked over steps.
    return new_mp_state, accum_mp_state

  def __call__(self, features_list: List[_Features], repred: bool,
               algorithm_index: int,
               return_hints: bool,
               return_all_outputs: bool):
    """Process one batch of data.

    Args:
      features_list: A list of _Features objects, each with the inputs, hints
        and lengths for a batch o data corresponding to one algorithm.
        The list should have either length 1, at train/evaluation time,
        or length equal to the number of algorithms this Net is meant to
        process, at initialization.
      repred: False during training, when we have access to ground-truth hints.
        True in validation/test mode, when we have to use our own
        hint predictions.
      algorithm_index: Which algorithm is being processed. It can be -1 at
        initialisation (either because we are initialising the parameters of
        the module or because we are intialising the message-passing state),
        meaning that all algorithms should be processed, in which case
        `features_list` should have length equal to the number of specs of
        the Net. Otherwise, `algorithm_index` should be
        between 0 and `length(self.spec) - 1`, meaning only one of the
        algorithms will be processed, and `features_list` should have length 1.
      return_hints: Whether to accumulate and return the predicted hints,
        when they are decoded.
      return_all_outputs: Whether to return the full sequence of outputs, or
        just the last step's output.

    Returns:
      A 2-tuple with (output predictions, hint predictions)
      for the selected algorithm.
    """
    if algorithm_index == -1:
      algorithm_indices = range(len(features_list))
    else:
      algorithm_indices = [algorithm_index]
    assert len(algorithm_indices) == len(features_list)

    self.encoders, self.decoders = self._construct_encoders_decoders()
    self.processor = self.processor_factory(self.hidden_dim)
    print('__call__ called')

    # Optionally construct LSTM.
    if self.use_lstm:
      self.lstm = hk.LSTM(
          hidden_size=self.hidden_dim,
          name='processor_lstm')
      lstm_init = self.lstm.initial_state
    else:
      self.lstm = None
      lstm_init = lambda x: 0

    for algorithm_index, features in zip(algorithm_indices, features_list):
      inputs = features.inputs
      hints = features.hints
      lengths = features.lengths

      batch_size, nb_nodes = _data_dimensions(features)

      nb_mp_steps = max(1, hints[0].data.shape[0] - 1)
      hiddens = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))

      if self.use_lstm:
        lstm_state = lstm_init(batch_size * nb_nodes)
        lstm_state = jax.tree_util.tree_map(
            lambda x, b=batch_size, n=nb_nodes: jnp.reshape(x, [b, n, -1]),
            lstm_state)
      else:
        lstm_state = None

      mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
          hint_preds=None, output_preds=None,
          hiddens=hiddens, lstm_state=lstm_state)

      # Do the first step outside of the scan because it has a different
      # computation graph.
      common_args = dict(
          hints=hints,
          repred=repred,
          inputs=inputs,
          batch_size=batch_size,
          nb_nodes=nb_nodes,
          lengths=lengths,
          spec=self.spec[algorithm_index],
          encs=self.encoders[algorithm_index],
          decs=self.decoders[algorithm_index],
          return_hints=return_hints,
          return_all_outputs=return_all_outputs,
          )
      mp_state, lean_mp_state = self._msg_passing_step(
          mp_state,
          i=0,
          first_step=True,
          **common_args)

      # Then scan through the rest.
      scan_fn = functools.partial(
          self._msg_passing_step,
          first_step=False,
          **common_args)

      output_mp_state, accum_mp_state = hk.scan(
          scan_fn,
          mp_state,
          jnp.arange(nb_mp_steps - 1) + 1,
          length=nb_mp_steps - 1)

    # We only return the last algorithm's output. That's because
    # the output only matters when a single algorithm is processed; the case
    # `algorithm_index==-1` (meaning all algorithms should be processed)
    # is used only to init parameters.
    accum_mp_state = jax.tree_util.tree_map(
        lambda init, tail: jnp.concatenate([init[None], tail], axis=0),
        lean_mp_state, accum_mp_state)

    def invert(d):
      """Dict of lists -> list of dicts."""
      if d:
        return [dict(zip(d, i)) for i in zip(*d.values())]

    if return_all_outputs:
      output_preds = {k: jnp.stack(v)
                      for k, v in accum_mp_state.output_preds.items()}
    else:
      output_preds = output_mp_state.output_preds
    hint_preds = invert(accum_mp_state.hint_preds)

    if self.debug:
      hiddens = jnp.stack([v for v in accum_mp_state.hiddens])
      return output_preds, hint_preds, hiddens

    return output_preds, hint_preds

  def _construct_encoders_decoders(self):
    """Constructs encoders and decoders, separate for each algorithm."""
    encoders_ = []
    decoders_ = []
    enc_algo_idx = None
    for (algo_idx, spec) in enumerate(self.spec):
      enc = {}
      dec = {}
      for name, (stage, loc, t) in spec.items():
        if stage == _Stage.INPUT or (
            stage == _Stage.HINT and self.encode_hints):
          # Build input encoders.
          if name == specs.ALGO_IDX_INPUT_NAME:
            if enc_algo_idx is None:
              enc_algo_idx = [hk.Linear(self.hidden_dim,
                                        name=f'{name}_enc_linear')]
            enc[name] = enc_algo_idx
          else:
            enc[name] = encoders.construct_encoders(
                stage, loc, t, hidden_dim=self.hidden_dim,
                init=self.encoder_init,
                name=f'algo_{algo_idx}_{name}')

        if stage == _Stage.OUTPUT or (
            stage == _Stage.HINT and self.decode_hints):
          # Build output decoders.
          dec[name] = decoders.construct_decoders(
              loc, t, hidden_dim=self.hidden_dim,
              nb_dims=self.nb_dims[algo_idx][name],
              name=f'algo_{algo_idx}_{name}')
      encoders_.append(enc)
      decoders_.append(dec)

    return encoders_, decoders_

  def _one_step_pred(
      self,
      inputs: _Trajectory,
      hints: _Trajectory,
      hidden: _Array,
      batch_size: int,
      nb_nodes: int,
      lstm_state: Optional[hk.LSTMState],
      spec: _Spec,
      encs: Dict[str, List[hk.Module]],
      decs: Dict[str, Tuple[hk.Module]],
      repred: bool,
  ):
    """Generates one-step predictions."""

    # Initialise empty node/edge/graph features and adjacency matrix.
    node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
    edge_fts = jnp.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim))
    graph_fts = jnp.zeros((batch_size, self.hidden_dim))
    adj_mat = jnp.repeat(
        jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)

    # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Encode node/edge/graph features from inputs and (optionally) hints.
    trajectories = [inputs]
    if self.encode_hints:
      trajectories.append(hints)

    for trajectory in trajectories:
      for dp in trajectory:
        try:
          dp = encoders.preprocess(dp, nb_nodes)
          assert dp.type_ != _Type.SOFT_POINTER
          adj_mat = encoders.accum_adj_mat(dp, adj_mat)
          encoder = encs[dp.name]
          edge_fts = encoders.accum_edge_fts(encoder, dp, edge_fts)
          node_fts = encoders.accum_node_fts(encoder, dp, node_fts)
          graph_fts = encoders.accum_graph_fts(encoder, dp, graph_fts)
        except Exception as e:
          raise Exception(f'Failed to process {dp}') from e

    # PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nxt_hidden = hidden
    for _ in range(self.nb_msg_passing_steps):
      nxt_hidden, nxt_edge = self.processor(
          node_fts,
          edge_fts,
          graph_fts,
          adj_mat,
          nxt_hidden,
          batch_size=batch_size,
          nb_nodes=nb_nodes,
      )

    if not repred:      # dropout only on training
      nxt_hidden = hk.dropout(hk.next_rng_key(), self._dropout_prob, nxt_hidden)

    if self.use_lstm:
      # lstm doesn't accept multiple batch dimensions (in our case, batch and
      # nodes), so we vmap over the (first) batch dimension.
      nxt_hidden, nxt_lstm_state = jax.vmap(self.lstm)(nxt_hidden, lstm_state)
    else:
      nxt_lstm_state = None

    h_t = jnp.concatenate([node_fts, hidden, nxt_hidden], axis=-1)
    if nxt_edge is not None:
      e_t = jnp.concatenate([edge_fts, nxt_edge], axis=-1)
    else:
      e_t = edge_fts

    # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Decode features and (optionally) hints.
    hint_preds, output_preds = decoders.decode_fts(
        decoders=decs,
        spec=spec,
        h_t=h_t,
        adj_mat=adj_mat,
        edge_fts=e_t,
        graph_fts=graph_fts,
        inf_bias=self.processor.inf_bias,
        inf_bias_edge=self.processor.inf_bias_edge,
        repred=repred,
    )

    return nxt_hidden, output_preds, hint_preds, nxt_lstm_state

def _data_dimensions(features: _Features) -> Tuple[int, int]:
  """Returns (batch_size, nb_nodes)."""
  for inp in features.inputs:
    if inp.location in [_Location.NODE, _Location.EDGE]:
      return inp.data.shape[:2]
  assert False

def _expand_to(x: _Array, y: _Array) -> _Array:
  while len(y.shape) > len(x.shape):
    x = jnp.expand_dims(x, -1)
  return x


def _is_not_done_broadcast(lengths, i, tensor):
  is_not_done = (lengths > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):  # pytype: disable=attribute-error  # numpy-scalars
    is_not_done = jnp.expand_dims(is_not_done, -1)
  return is_not_done
