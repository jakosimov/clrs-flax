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
"""Encoder utilities."""

from enum import StrEnum
from jax import Array
from clrs._src import probing
from clrs._src import specs
import haiku as hk
import flax.nnx as nnx
import jax.numpy as jnp

_DataPoint = probing.DataPoint
_Location = specs.Location
_Stage = specs.Stage
_Type = specs.Type


class EncoderInitialiser(StrEnum):
    """Initialiser for the encoder.

    The initialiser is used to set the weights of the encoder. The default
    initialiser is a truncated normal distribution with standard deviation
    1/sqrt(hidden_dim). This is the same as the Xavier initialisation.
    """

    DEFAULT = "default"
    XAVIER_ON_SCALARS = "xavier_on_scalars"


class Encoder(nnx.Module):
    """Encoder module.

    This module is used to encode the input data into a feature vector.
    The encoder is a linear layer followed by a non-linear activation function.
    The activation function is ReLU by default, but can be changed to other
    functions such as sigmoid or tanh.
    """

    def __init__(self, output_dim: int, rngs: nnx.Rngs):
        super().__init__()

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError("Encoder not implemented.")


class OneWayEncoder(Encoder):
    """One-way encoder module.

    This module is used to encode the input data into a feature vector.
    The encoder is a linear layer followed by a non-linear activation function.
    The activation function is ReLU by default, but can be changed to other
    functions such as sigmoid or tanh.
    """

    def __init__(self, output_dim: int, rngs: nnx.Rngs, input_dim=1):
        super().__init__(output_dim, rngs)
        self.linear = nnx.Linear(input_dim, output_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.linear(x)


class TwoWayEncoder(Encoder):
    """Two-way encoder module.

    This module is used to encode the input data into a feature vector.
    The encoder is a linear layer followed by a non-linear activation function.
    The activation function is ReLU by default, but can be changed to other
    functions such as sigmoid or tanh.
    """

    def __init__(self, output_dim: int, rngs: nnx.Rngs):
        super().__init__(output_dim, rngs)
        self.linear_1 = nnx.Linear(1, output_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(1, output_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.linear_1(x)

    def secondary(self, x: Array) -> Array:
        return self.linear_2(x)


def construct_encoders_flax(
    stage: str,
    loc: str,
    t: str,
    hidden_dim: int,
    init: EncoderInitialiser,
    name: str,
    rngs: nnx.Rngs,
    nb_dims: int,
) -> Encoder:
    """Constructs encoders."""
    if (
        init == EncoderInitialiser.XAVIER_ON_SCALARS
        and stage == _Stage.HINT
        and t == _Type.SCALAR
    ):
        initialiser = hk.initializers.TruncatedNormal(stddev=1.0 / jnp.sqrt(hidden_dim))
    else:
        initialiser = None
    # ^ Fix this to use flax initializers

    # idk what the input dimension is here
    if loc == _Location.EDGE and t == _Type.POINTER:
        return TwoWayEncoder(hidden_dim, rngs)
    elif t == _Type.CATEGORICAL:
        return OneWayEncoder(input_dim=nb_dims, output_dim=hidden_dim, rngs=rngs)
    else:
        return OneWayEncoder(hidden_dim, rngs)
    # encoders = [linear(hidden_dim)]
    # if loc == _Location.EDGE and t == _Type.POINTER:
    #     # Edge pointers need two-way encoders.
    #     encoders.append(linear(hidden_dim))

    # return encoders


def preprocess(dp: _DataPoint, nb_nodes: int) -> _DataPoint:
    """Pre-process data point.

    Make sure that the data is ready to be encoded into features.
    If the data is of POINTER type, we expand the compressed index representation
    to a full one-hot. But if the data is a SOFT_POINTER, the representation
    is already expanded and we just overwrite the type as POINTER so that
    it is treated as such for encoding.

    Args:
      dp: A DataPoint to prepare for encoding.
      nb_nodes: Number of nodes in the graph, necessary to expand pointers to
        the right dimension.
    Returns:
      The datapoint, with data and possibly type modified.
    """
    new_type = dp.type_
    if dp.type_ == _Type.POINTER:
        data = hk.one_hot(dp.data, nb_nodes)
    else:
        data = dp.data.astype(jnp.float32)
        if dp.type_ == _Type.SOFT_POINTER:
            new_type = _Type.POINTER
    dp = probing.DataPoint(
        name=dp.name, location=dp.location, type_=new_type, data=data
    )  # pytype: disable=wrong-args

    return dp


def accum_adj_mat(dp: _DataPoint, adj_mat: Array) -> Array:
    """Accumulates adjacency matrix."""
    if dp.location == _Location.NODE and dp.type_ in [
        _Type.POINTER,
        _Type.PERMUTATION_POINTER,
    ]:
        adj_mat += (dp.data + jnp.transpose(dp.data, (0, 2, 1))) > 0.5
    elif dp.location == _Location.EDGE and dp.type_ == _Type.MASK:
        adj_mat += (dp.data + jnp.transpose(dp.data, (0, 2, 1))) > 0.0

    return (adj_mat > 0.0).astype(
        "float32"
    )  # pytype: disable=attribute-error  # numpy-scalars


def accum_edge_fts(encoders: Encoder, dp: _DataPoint, edge_fts: Array) -> Array:
    """Encodes and accumulates edge features."""
    if dp.location == _Location.NODE and dp.type_ in [
        _Type.POINTER,
        _Type.PERMUTATION_POINTER,
    ]:
        assert isinstance(encoders, OneWayEncoder)
        encoding = _encode_inputs(encoders, dp)
        edge_fts += encoding

    elif dp.location == _Location.EDGE:
        encoding = _encode_inputs(encoders, dp)
        if dp.type_ == _Type.POINTER:
            assert isinstance(encoders, TwoWayEncoder)
            # Aggregate pointer contributions across sender and receiver nodes.
            encoding_2 = encoders.secondary(jnp.expand_dims(dp.data, -1))
            edge_fts += jnp.mean(encoding, axis=1) + jnp.mean(encoding_2, axis=2)
        else:
            edge_fts += encoding

    return edge_fts


def accum_node_fts(encoders: Encoder, dp: _DataPoint, node_fts: Array) -> Array:
    """Encodes and accumulates node features."""
    is_pointer = dp.type_ in [_Type.POINTER, _Type.PERMUTATION_POINTER]
    if (dp.location == _Location.NODE and not is_pointer) or (
        dp.location == _Location.GRAPH and dp.type_ == _Type.POINTER
    ):
        encoding = _encode_inputs(encoders, dp)
        node_fts += encoding

    return node_fts


def accum_graph_fts(encoders, dp: _DataPoint, graph_fts: Array) -> Array:
    """Encodes and accumulates graph features."""
    if dp.location == _Location.GRAPH and dp.type_ != _Type.POINTER:
        encoding = _encode_inputs(encoders, dp)
        graph_fts += encoding

    return graph_fts


def _encode_inputs(encoders: Encoder, dp: _DataPoint) -> Array:
    if dp.type_ == _Type.CATEGORICAL:
        encoding = encoders(dp.data)
    else:
        encoding = encoders(jnp.expand_dims(dp.data, -1))
    return encoding
