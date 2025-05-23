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
"""decoders utilities."""

from typing import Dict, Optional

from clrs._src import probing
from clrs._src import specs
import haiku as hk
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax import Array

_DataPoint = probing.DataPoint
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Type = specs.Type


def log_sinkhorn(
    x: jax.Array,
    steps: int,
    temperature: float,
    zero_diagonal: bool,
    noise_rng_key: Optional[Array],
) -> Array:
    """Sinkhorn operator in log space, to postprocess permutation pointer logits.

    Args:
      x: input of shape [..., n, n], a batch of square matrices.
      steps: number of iterations.
      temperature: temperature parameter (as temperature approaches zero, the
        output approaches a permutation matrix).
      zero_diagonal: whether to force the diagonal logits towards -inf.
      noise_rng_key: key to add Gumbel noise.

    Returns:
      Elementwise logarithm of a doubly-stochastic matrix (a matrix with
      non-negative elements whose rows and columns sum to 1).
    """
    assert x.ndim >= 2
    assert x.shape[-1] == x.shape[-2]
    if noise_rng_key is not None:
        # Add standard Gumbel noise (see https://arxiv.org/abs/1802.08665)
        noise = -jnp.log(
            -jnp.log(jax.random.uniform(noise_rng_key, x.shape) + 1e-12) + 1e-12
        )
        x = x + noise
    x /= temperature
    if zero_diagonal:
        x = x - 1e6 * jnp.eye(x.shape[-1])
    for _ in range(steps):
        x = jax.nn.log_softmax(x, axis=-1)
        x = jax.nn.log_softmax(x, axis=-2)
    return x


class Decoder(nnx.Module):
    """Decoder module.

    This module is used to decode the feature vector into the output data.
    The decoder is a linear layer followed by a non-linear activation function.
    The activation function is ReLU by default, but can be changed to other
    functions such as sigmoid or tanh.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x: jax.Array) -> Array:
        raise NotImplementedError("Decoder not implemented.")


class NodeFeatureDecoder(Decoder):
    def __init__(self, output_dim: int, rngs: nnx.Rngs, h_t_dim: int):
        super().__init__()
        self.linear = nnx.Linear(h_t_dim, output_dim, rngs=rngs)

    def __call__(self, h_t: jax.Array) -> Array:
        return self.linear(h_t)


class NodePointerDecoder(Decoder):
    def __init__(
        self,
        rngs: nnx.Rngs,
        h_dim: int,
        edge_fts_dim: int,
        mid_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.linear_1 = nnx.Linear(h_dim, mid_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(h_dim, mid_dim, rngs=rngs)
        self.linear_3 = nnx.Linear(edge_fts_dim, mid_dim, rngs=rngs)
        self.linear_4 = nnx.Linear(mid_dim, output_dim, rngs=rngs)

    def __call__(self, h_t: jax.Array, edge_fts: jax.Array) -> Array:
        p_1, p_2, p_3 = self.linear_1(h_t), self.linear_2(h_t), self.linear_3(edge_fts)

        p_e = jnp.expand_dims(p_2, -2) + p_3
        p_m = jnp.maximum(jnp.expand_dims(p_1, -2), jnp.transpose(p_e, (0, 2, 1, 3)))

        return self.linear_4(p_m)


class EdgeFeatureDecoder(Decoder):
    def __init__(
        self,
        output_dim1: int,
        output_dim2: int,
        output_dim3: int,
        rngs: nnx.Rngs,
        h_t_dim: int,
        edge_fts_dim: int,
    ):
        super().__init__()
        self.linear_1 = nnx.Linear(h_t_dim, output_dim1, rngs=rngs)
        self.linear_2 = nnx.Linear(h_t_dim, output_dim2, rngs=rngs)
        self.linear_3 = nnx.Linear(edge_fts_dim, output_dim3, rngs=rngs)

    def __call__(self, h_t: Array, edge_fts: Array) -> tuple[Array, Array, Array]:
        pred_1 = self.linear_1(h_t)
        pred_2 = self.linear_2(h_t)
        pred_e = self.linear_3(edge_fts)
        return pred_1, pred_2, pred_e


class EdgePointerDecoder(Decoder):
    def __init__(
        self,
        h_t_dim: int,
        edge_fts_dim: int,
        rngs: nnx.Rngs,
        mid_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.linear_1 = nnx.Linear(h_t_dim, mid_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(h_t_dim, mid_dim, rngs=rngs)
        self.linear_3 = nnx.Linear(edge_fts_dim, mid_dim, rngs=rngs)
        self.linear_4 = nnx.Linear(h_t_dim, mid_dim, rngs=rngs)
        self.linear_5 = nnx.Linear(mid_dim, output_dim, rngs=rngs)

    def __call__(self, h_t: Array, edge_fts: Array) -> Array:
        pred_1 = self.linear_1(h_t)
        pred_2 = self.linear_2(h_t)
        pred_e = self.linear_3(edge_fts)
        pred = jnp.expand_dims(pred_1, -2) + jnp.expand_dims(pred_2, -3) + pred_e
        pred_2 = self.linear_4(h_t)
        p_m = jnp.maximum(
            jnp.expand_dims(pred, -2), jnp.expand_dims(jnp.expand_dims(pred_2, -3), -3)
        )
        result = self.linear_5(p_m)
        return result


class GraphFeatureDecoder(Decoder):
    def __init__(
        self,
        output_dim1: int,
        output_dim2: int,
        rngs: nnx.Rngs,
        gr_emb_dim: int,
        graph_fts_dim: int,
    ):
        super().__init__()
        self.linear_1 = nnx.Linear(gr_emb_dim, output_dim1, rngs=rngs)
        self.linear_2 = nnx.Linear(graph_fts_dim, output_dim2, rngs=rngs)

    def __call__(self, gr_emb: jax.Array, graph_fts: jax.Array) -> tuple[Array, Array]:
        return self.linear_1(gr_emb), self.linear_2(graph_fts)


class GraphPointerDecoder(Decoder):
    def __init__(
        self,
        output_dim: int,
        rngs: nnx.Rngs,
        gr_emb_dim: int,
        graph_fts_dim: int,
        mid_dim: int,
    ):
        super().__init__()
        self.linear_1 = nnx.Linear(gr_emb_dim, mid_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(graph_fts_dim, mid_dim, rngs=rngs)
        self.linear_3 = nnx.Linear(mid_dim, output_dim, rngs=rngs)

    def __call__(self, gr_emb: Array, graph_fts: Array) -> tuple[Array, Array, Array]:
        pred_n = self.linear_1(gr_emb)
        pred_g = self.linear_2(graph_fts)
        pred_comb = self.linear_3(pred_n + pred_g)
        return pred_n, pred_g, pred_comb


def construct_decoders_flax(
    loc: str, t: str, hidden_dim: int, nb_dims: int, name: str, rngs: nnx.Rngs
) -> Decoder:
    """Constructs decoders."""
    h_t_dim = 3 * hidden_dim
    edge_fts_dim = hidden_dim
    gr_emb_dim = hidden_dim
    graph_fts_dim = hidden_dim
    # linear = lambda out_dims: nnx.Linear(hidden_dim, out_dims, rngs=rngs)
    if loc == _Location.NODE:
        # Node decoders.
        if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            decoders = NodeFeatureDecoder(output_dim=1, rngs=rngs, h_t_dim=h_t_dim)
        elif t == _Type.CATEGORICAL:
            decoders = NodeFeatureDecoder(
                output_dim=nb_dims, rngs=rngs, h_t_dim=h_t_dim
            )
        elif t in [_Type.POINTER, _Type.PERMUTATION_POINTER]:
            decoders = NodePointerDecoder(
                rngs=rngs,
                mid_dim=hidden_dim,
                h_dim=h_t_dim,
                edge_fts_dim=edge_fts_dim,
                output_dim=1,
            )
        else:
            raise ValueError(f"Invalid Type {t}")

    elif loc == _Location.EDGE:
        # Edge decoders.
        if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            decoders = EdgeFeatureDecoder(
                1, 1, 1, rngs=rngs, h_t_dim=h_t_dim, edge_fts_dim=edge_fts_dim
            )
        elif t == _Type.CATEGORICAL:
            decoders = EdgeFeatureDecoder(
                nb_dims,
                nb_dims,
                nb_dims,
                rngs=rngs,
                h_t_dim=h_t_dim,
                edge_fts_dim=edge_fts_dim,
            )
        elif t == _Type.POINTER:
            decoders = EdgePointerDecoder(
                rngs=rngs,
                h_t_dim=h_t_dim,
                edge_fts_dim=edge_fts_dim,
                mid_dim=hidden_dim,
                output_dim=1,
            )
        else:
            raise ValueError(f"Invalid Type {t}")

    elif loc == _Location.GRAPH:
        # Graph decoders.
        if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            decoders = GraphFeatureDecoder(
                output_dim1=1,
                output_dim2=1,
                rngs=rngs,
                gr_emb_dim=gr_emb_dim,
                graph_fts_dim=graph_fts_dim,
            )
        elif t == _Type.CATEGORICAL:
            decoders = GraphFeatureDecoder(
                output_dim1=nb_dims,
                output_dim2=nb_dims,
                rngs=rngs,
                gr_emb_dim=gr_emb_dim,
                graph_fts_dim=graph_fts_dim,
            )
        elif t == _Type.POINTER:
            decoders = GraphPointerDecoder(
                output_dim=1,
                mid_dim=1,
                rngs=rngs,
                graph_fts_dim=graph_fts_dim,
                gr_emb_dim=gr_emb_dim,
            )
        else:
            raise ValueError(f"Invalid Type {t}")

    else:
        raise ValueError(f"Invalid Location {loc}")

    return decoders


def postprocess(
    spec: _Spec,
    preds: Dict[str, Array] | None,
    sinkhorn_temperature: float,
    sinkhorn_steps: int,
    hard: bool,
) -> Dict[str, _DataPoint]:
    """Postprocesses decoder output.

    This is done on outputs in order to score performance, and on hints in
    order to score them but also in order to feed them back to the model.
    At scoring time, the postprocessing mode is "hard", logits will be
    arg-maxed and masks will be thresholded. However, for the case of the hints
    that are fed back in the model, the postprocessing can be hard or soft,
    depending on whether we want to let gradients flow through them or not.

    Args:
      spec: The spec of the algorithm whose outputs/hints we are postprocessing.
      preds: Output and/or hint predictions, as produced by decoders.
      sinkhorn_temperature: Parameter for the sinkhorn operator on permutation
        pointers.
      sinkhorn_steps: Parameter for the sinkhorn operator on permutation
        pointers.
      hard: whether to do hard postprocessing, which involves argmax for
        MASK_ONE, CATEGORICAL and POINTERS, thresholding for MASK, and stop
        gradient through for SCALAR. If False, soft postprocessing will be used,
        with softmax, sigmoid and gradients allowed.
    Returns:
      The postprocessed `preds`. In "soft" post-processing, POINTER types will
      change to SOFT_POINTER, so encoders know they do not need to be
      pre-processed before feeding them back in.
    """
    assert preds is not None
    result = {}
    for name in preds.keys():
        _, loc, t = spec[name]
        new_t = t
        data = preds[name]
        if t == _Type.SCALAR:
            if hard:
                data = jax.lax.stop_gradient(data)
        elif t == _Type.MASK:
            if hard:
                data = (data > 0.0) * 1.0
            else:
                data = jax.nn.sigmoid(data)
        elif t in [_Type.MASK_ONE, _Type.CATEGORICAL]:
            cat_size = data.shape[-1]
            if hard:
                best = jnp.argmax(data, -1)
                data = hk.one_hot(best, cat_size)
            else:
                data = jax.nn.softmax(data, axis=-1)
        elif t == _Type.POINTER:
            if hard:
                data = jnp.argmax(data, -1).astype(float)
            else:
                data = jax.nn.softmax(data, -1)
                new_t = _Type.SOFT_POINTER
        elif t == _Type.PERMUTATION_POINTER:
            # Convert the matrix of logits to a doubly stochastic matrix.
            data = log_sinkhorn(
                x=data,
                steps=sinkhorn_steps,
                temperature=sinkhorn_temperature,
                zero_diagonal=True,
                noise_rng_key=None,
            )
            data = jnp.exp(data)
            if hard:
                data = jax.nn.one_hot(jnp.argmax(data, axis=-1), data.shape[-1])
        else:
            raise ValueError("Invalid type")
        result[name] = probing.DataPoint(
            name=name, location=loc, type_=new_t, data=data
        )

    return result


def decode_fts(
    decoders: Dict[str, Decoder],
    spec: _Spec,
    h_t: jax.Array,
    adj_mat: jax.Array,
    edge_fts: jax.Array,
    graph_fts: jax.Array,
    inf_bias: bool,
    inf_bias_edge: bool,
    repred: bool,
):
    """Decodes node, edge and graph features."""
    output_preds = {}
    hint_preds = {}

    for name in decoders:
        decoder = decoders[name]
        stage, loc, t = spec[name]

        if loc == _Location.NODE:
            preds = _decode_node_fts(
                decoder, t, h_t, edge_fts, adj_mat, inf_bias, repred
            )
        elif loc == _Location.EDGE:
            preds = _decode_edge_fts(decoder, t, h_t, edge_fts, adj_mat, inf_bias_edge)
        elif loc == _Location.GRAPH:
            preds = _decode_graph_fts(decoder, t, h_t, graph_fts)
        else:
            raise ValueError("Invalid output type")

        if stage == _Stage.OUTPUT:
            output_preds[name] = preds
        elif stage == _Stage.HINT:
            hint_preds[name] = preds
        else:
            raise ValueError(f"Found unexpected decoder {name}")

    return hint_preds, output_preds


def _decode_node_fts(
    decoder: Decoder,
    t: str,
    h_t: jax.Array,
    edge_fts: jax.Array,
    adj_mat: jax.Array,
    inf_bias: bool,
    repred: bool,
) -> Array:
    """Decodes node features."""

    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
        assert isinstance(decoder, NodeFeatureDecoder)
        preds = jnp.squeeze(decoder(h_t), -1)
    elif t == _Type.CATEGORICAL:
        assert isinstance(decoder, NodeFeatureDecoder)
        preds = decoder(h_t)
    elif t in [_Type.POINTER, _Type.PERMUTATION_POINTER]:
        assert isinstance(decoder, NodePointerDecoder)
        pred = decoder(h_t, edge_fts)

        preds = jnp.squeeze(pred, -1)

        if inf_bias:
            per_batch_min = jnp.min(preds, axis=range(1, preds.ndim), keepdims=True)
            preds = jnp.where(
                adj_mat > 0.5, preds, jnp.minimum(-1.0, per_batch_min - 1.0)
            )
        if t == _Type.PERMUTATION_POINTER:
            if repred:  # testing or validation, no Gumbel noise
                preds = log_sinkhorn(
                    x=preds,
                    steps=10,
                    temperature=0.1,
                    zero_diagonal=True,
                    noise_rng_key=None,
                )
            else:  # training, add Gumbel noise
                preds = log_sinkhorn(
                    x=preds,
                    steps=10,
                    temperature=0.1,
                    zero_diagonal=True,
                    noise_rng_key=hk.next_rng_key(),
                )
    else:
        raise ValueError("Invalid output type")

    return preds


def _decode_edge_fts(
    decoders: Decoder,
    t: str,
    h_t: Array,
    edge_fts: Array,
    adj_mat: Array,
    inf_bias_edge: bool,
) -> Array:
    """Decodes edge features."""

    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
        assert isinstance(decoders, EdgeFeatureDecoder)
        pred_1, pred_2, pred_e = decoders(h_t, edge_fts)
        pred = jnp.expand_dims(pred_1, -2) + jnp.expand_dims(pred_2, -3) + pred_e
        preds = jnp.squeeze(pred, -1)
    elif t == _Type.CATEGORICAL:
        assert isinstance(decoders, EdgeFeatureDecoder)
        pred_1, pred_2, pred_e = decoders(h_t, edge_fts)
        pred = jnp.expand_dims(pred_1, -2) + jnp.expand_dims(pred_2, -3) + pred_e
        preds = pred
    elif t == _Type.POINTER:
        assert isinstance(decoders, EdgePointerDecoder)
        pred = decoders(h_t, edge_fts)
        preds = jnp.squeeze(pred, -1)
    else:
        raise ValueError("Invalid output type")
    if inf_bias_edge and t in [_Type.MASK, _Type.MASK_ONE]:
        per_batch_min = jnp.min(preds, axis=range(1, preds.ndim), keepdims=True)
        preds = jnp.where(adj_mat > 0.5, preds, jnp.minimum(-1.0, per_batch_min - 1.0))

    return preds


def _decode_graph_fts(
    decoder: Decoder, t: str, h_t: Array, graph_fts: jax.Array
) -> Array:
    """Decodes graph features."""

    gr_emb = jnp.max(h_t, axis=-2)
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
        assert isinstance(decoder, GraphFeatureDecoder)
        pred_n, pred_g = decoder(gr_emb, graph_fts)
        pred = pred_n + pred_g
        preds = jnp.squeeze(pred, -1)
    elif t == _Type.CATEGORICAL:
        assert isinstance(decoder, GraphFeatureDecoder)
        pred_n, pred_g = decoder(gr_emb, graph_fts)
        pred = pred_n + pred_g
        preds = pred
    elif t == _Type.POINTER:
        assert isinstance(decoder, GraphPointerDecoder)
        pred_n, pred_g, pred_comb = decoder(gr_emb, graph_fts)
        ptr_p = jnp.expand_dims(pred_n + pred_g, 1) + jnp.transpose(
            pred_comb, (0, 2, 1)
        )
        preds = jnp.squeeze(ptr_p, 1)
    else:
        raise ValueError("Invalid output type")

    return preds
