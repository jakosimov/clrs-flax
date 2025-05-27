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

"""JAX implementation of baseline processor networks."""

import abc
from enum import StrEnum
from typing import Any, Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax import Array


_Fn = Callable[..., Any]
BIG_NUMBER = 1e6
PROCESSOR_TAG = "clrs_processor"


class Processor(nnx.Module):
    """Processor abstract base class."""

    def __init__(self, name: str):
        # if not name.endswith(PROCESSOR_TAG):
        #     name = name + "_" + PROCESSOR_TAG
        super().__init__()

    @abc.abstractmethod
    def __call__(
        self,
        node_fts: Array,
        edge_fts: Array,
        graph_fts: Array,
        adj_mat: Array,
        hidden: Array,
        **kwargs,
    ) -> Tuple[Array, Optional[Array]]:
        """Processor inference step.

        Args:
          node_fts: Node features.
          edge_fts: Edge features.
          graph_fts: Graph features.
          adj_mat: Graph adjacency matrix.
          hidden: Hidden features.
          **kwargs: Extra kwargs.

        Returns:
          Output of processor inference step as a 2-tuple of (node, edge)
          embeddings. The edge embeddings can be None.
        """
        pass

    @property
    def inf_bias(self):
        return False

    @property
    def inf_bias_edge(self):
        return False

    @property
    def using_triplets(self):
        return False


class GAT(Processor):
    """Graph Attention Network (Velickovic et al., ICLR 2018)."""

    def __init__(
        self,
        out_size: int,
        nb_heads: int,
        rngs: nnx.Rngs,
        activation: Optional[_Fn] = jax.nn.relu,
        residual: bool = True,
        use_ln: bool = False,
        name: str = "gat_aggr",
    ):
        super().__init__(name=name)
        self.out_size = out_size
        self.nb_heads = nb_heads
        if out_size % nb_heads != 0:
            raise ValueError("The number of attention heads must divide the width!")
        self.head_size = out_size // nb_heads
        self.activation = activation
        self.residual = residual
        self.use_ln = use_ln

        self.m = nnx.Linear(self.out_size, self.out_size, rngs=rngs)
        self.skip = nnx.Linear(self.out_size, self.out_size, rngs=rngs)

        self.a_1 = nnx.Linear(self.out_size, self.out_size, rngs=rngs)
        self.a_2 = nnx.Linear(self.out_size, self.out_size, rngs=rngs)
        self.a_e = nnx.Linear(self.out_size, self.out_size, rngs=rngs)
        self.a_g = nnx.Linear(self.out_size, self.out_size, rngs=rngs)

        if self.use_ln:
            self.ln = nnx.LayerNorm(
                num_features=self.out_size,
                # axis=-1,
                feature_axes=-1,  # maybe this should be reduction_axes?
                use_bias=True,
                use_scale=True,
                rngs=rngs,
            )

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
        self,
        node_fts: Array,
        edge_fts: Array,
        graph_fts: Array,
        adj_mat: Array,
        hidden: Array,
        **unused_kwargs,
    ) -> Tuple[Array, Optional[Array]]:
        """GAT inference step."""

        b, n, _ = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        z = jnp.concatenate([node_fts, hidden], axis=-1)

        bias_mat = (adj_mat - 1.0) * 1e9
        bias_mat = jnp.tile(
            bias_mat[..., None], (1, 1, 1, self.nb_heads)
        )  # [B, N, N, H]
        bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

        values = self.m(z)  # [B, N, H*F]
        values = jnp.reshape(
            values, values.shape[:-1] + (self.nb_heads, self.head_size)
        )  # [B, N, H, F]
        values = jnp.transpose(values, (0, 2, 1, 3))  # [B, H, N, F]

        att_1 = jnp.expand_dims(self.a_1(z), axis=-1)
        att_2 = jnp.expand_dims(self.a_2(z), axis=-1)
        att_e = self.a_e(edge_fts)
        att_g = jnp.expand_dims(self.a_g(graph_fts), axis=-1)

        logits = (
            jnp.transpose(att_1, (0, 2, 1, 3))  # + [B, H, N, 1]
            + jnp.transpose(att_2, (0, 2, 3, 1))  # + [B, H, 1, N]
            + jnp.transpose(att_e, (0, 3, 1, 2))  # + [B, H, N, N]
            + jnp.expand_dims(att_g, axis=-1)  # + [B, H, 1, 1]
        )  # = [B, H, N, N]
        coefs = jax.nn.softmax(jax.nn.leaky_relu(logits) + bias_mat, axis=-1)
        ret = jnp.matmul(coefs, values)  # [B, H, N, F]
        ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
        ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

        if self.residual:
            ret += self.skip(z)

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            # ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = self.ln(ret)

        return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class GATFull(GAT):
    """Graph Attention Network with full adjacency matrix."""

    def __call__(
        self,
        node_fts: Array,
        edge_fts: Array,
        graph_fts: Array,
        adj_mat: Array,
        hidden: Array,
        **unused_kwargs,
    ) -> Tuple[Array, Optional[Array]]:
        adj_mat = jnp.ones_like(adj_mat)
        return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class GATv2(Processor):
    """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

    def __init__(
        self,
        out_size: int,
        nb_heads: int,
        rngs: nnx.Rngs,
        mid_size: Optional[int] = None,
        activation: Optional[_Fn] = jax.nn.relu,
        residual: bool = True,
        use_ln: bool = False,
        name: str = "gatv2_aggr",
    ):
        super().__init__(name=name)
        if mid_size is None:
            self.mid_size = out_size
        else:
            self.mid_size = mid_size
        self.out_size = out_size
        self.nb_heads = nb_heads
        if out_size % nb_heads != 0:
            raise ValueError("The number of attention heads must divide the width!")
        self.head_size = out_size // nb_heads
        if self.mid_size % nb_heads != 0:
            raise ValueError("The number of attention heads must divide the message!")
        self.mid_head_size = self.mid_size // nb_heads
        self.activation = activation
        self.residual = residual
        self.use_ln = use_ln

        self.m = nnx.Linear(self.mid_size, self.mid_size, rngs=rngs)
        self.skip = nnx.Linear(self.mid_size, self.mid_size, rngs=rngs)
        self.w_1 = nnx.Linear(self.mid_size, self.mid_size, rngs=rngs)
        self.w_2 = nnx.Linear(self.mid_size, self.mid_size, rngs=rngs)
        self.w_e = nnx.Linear(self.mid_size, self.mid_size, rngs=rngs)
        self.w_g = nnx.Linear(self.mid_size, self.mid_size, rngs=rngs)

        self.a_heads: list[nnx.Linear] = []
        for _ in range(self.nb_heads):
            self.a_heads.append(nnx.Linear(1, 1, rngs=rngs))

        if self.use_ln:
            self.ln = nnx.LayerNorm(
                num_features=self.out_size,
                # axis=-1,
                feature_axes=-1,  # maybe this should be reduction_axes?
                use_bias=True,
                use_scale=True,
                rngs=rngs,
            )

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
        self,
        node_fts: Array,
        edge_fts: Array,
        graph_fts: Array,
        adj_mat: Array,
        hidden: Array,
        **unused_kwargs,
    ) -> Tuple[Array, Optional[Array]]:
        """GATv2 inference step."""

        b, n, _ = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        z = jnp.concatenate([node_fts, hidden], axis=-1)

        bias_mat = (adj_mat - 1.0) * 1e9
        bias_mat = jnp.tile(
            bias_mat[..., None], (1, 1, 1, self.nb_heads)
        )  # [B, N, N, H]
        bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

        values = self.m(z)  # [B, N, H*F]
        values = jnp.reshape(
            values, values.shape[:-1] + (self.nb_heads, self.head_size)
        )  # [B, N, H, F]
        values = jnp.transpose(values, (0, 2, 1, 3))  # [B, H, N, F]

        pre_att_1 = self.w_1(z)
        pre_att_2 = self.w_2(z)
        pre_att_e = self.w_e(edge_fts)
        pre_att_g = self.w_g(graph_fts)

        pre_att = (
            jnp.expand_dims(pre_att_1, axis=1)  # + [B, 1, N, H*F]
            + jnp.expand_dims(pre_att_2, axis=2)  # + [B, N, 1, H*F]
            + pre_att_e  # + [B, N, N, H*F]
            + jnp.expand_dims(pre_att_g, axis=(1, 2))  # + [B, 1, 1, H*F]
        )  # = [B, N, N, H*F]

        pre_att = jnp.reshape(
            pre_att, pre_att.shape[:-1] + (self.nb_heads, self.mid_head_size)
        )  # [B, N, N, H, F]

        pre_att = jnp.transpose(pre_att, (0, 3, 1, 2, 4))  # [B, H, N, N, F]

        # This part is not very efficient, but we agree to keep it this way to
        # enhance readability, assuming `nb_heads` will not be large.
        logit_heads = []
        for head in range(self.nb_heads):
            logit_heads.append(
                jnp.squeeze(
                    self.a_heads[head](jax.nn.leaky_relu(pre_att[:, head])), axis=-1
                )
            )  # [B, N, N]

        logits = jnp.stack(logit_heads, axis=1)  # [B, H, N, N]

        coefs = jax.nn.softmax(logits + bias_mat, axis=-1)
        ret = jnp.matmul(coefs, values)  # [B, H, N, F]
        ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
        ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

        if self.residual:
            ret += self.skip(z)

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ret = self.ln(ret)

        return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class GATv2FullD2(GATv2):
    """Graph Attention Network v2 with full adjacency matrix and D2 symmetry."""

    def d2_forward(
        self,
        node_fts: List[Array],
        edge_fts: List[Array],
        graph_fts: List[Array],
        adj_mat: Array,
        hidden: Array,
        **unused_kwargs,
    ) -> List[Array]:
        num_d2_actions = 4

        d2_inverses = [0, 1, 2, 3]  # All members of D_2 are self-inverses!

        d2_multiply = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ]

        assert len(node_fts) == num_d2_actions
        assert len(edge_fts) == num_d2_actions
        assert len(graph_fts) == num_d2_actions

        ret_nodes = []
        adj_mat = jnp.ones_like(adj_mat)

        for g in range(num_d2_actions):
            emb_values = []
            for h in range(num_d2_actions):
                gh = d2_multiply[d2_inverses[g]][h]
                node_features = jnp.concatenate((node_fts[g], node_fts[gh]), axis=-1)
                edge_features = jnp.concatenate((edge_fts[g], edge_fts[gh]), axis=-1)
                graph_features = jnp.concatenate((graph_fts[g], graph_fts[gh]), axis=-1)
                cell_embedding = super().__call__(
                    node_fts=node_features,
                    edge_fts=edge_features,
                    graph_fts=graph_features,
                    adj_mat=adj_mat,
                    hidden=hidden,
                )
                emb_values.append(cell_embedding[0])
            ret_nodes.append(jnp.mean(jnp.stack(emb_values, axis=0), axis=0))

        return ret_nodes


class GATv2Full(GATv2):
    """Graph Attention Network v2 with full adjacency matrix."""

    def __call__(
        self,
        node_fts: Array,
        edge_fts: Array,
        graph_fts: Array,
        adj_mat: Array,
        hidden: Array,
        **unused_kwargs,
    ) -> Tuple[Array, Optional[Array]]:
        adj_mat = jnp.ones_like(adj_mat)
        return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class TripletMessageModule(nnx.Module):
    """Triplet message module."""

    def __init__(
        self,
        nb_triplet_fts: int,
        z_size: int,
        edge_fts_size: int,
        graph_fts_size: int,
        rngs: nnx.Rngs,
    ):
        self.t_1 = nnx.Linear(z_size, nb_triplet_fts, rngs=rngs)
        self.t_2 = nnx.Linear(z_size, nb_triplet_fts, rngs=rngs)
        self.t_3 = nnx.Linear(z_size, nb_triplet_fts, rngs=rngs)
        self.t_e_1 = nnx.Linear(edge_fts_size, nb_triplet_fts, rngs=rngs)
        self.t_e_2 = nnx.Linear(edge_fts_size, nb_triplet_fts, rngs=rngs)
        self.t_e_3 = nnx.Linear(edge_fts_size, nb_triplet_fts, rngs=rngs)
        self.t_g = nnx.Linear(graph_fts_size, nb_triplet_fts, rngs=rngs)

    def __call__(self, z, edge_fts, graph_fts):
        """Triplet messages, as done by Dudzik and Velickovic (2022)."""
        tri_1 = self.t_1(z)
        tri_2 = self.t_2(z)
        tri_3 = self.t_3(z)
        tri_e_1 = self.t_e_1(edge_fts)
        tri_e_2 = self.t_e_2(edge_fts)
        tri_e_3 = self.t_e_3(edge_fts)
        tri_g = self.t_g(graph_fts)

        return (
            jnp.expand_dims(tri_1, axis=(2, 3))  #   (B, N, 1, 1, H)
            + jnp.expand_dims(tri_2, axis=(1, 3))  # + (B, 1, N, 1, H)
            + jnp.expand_dims(tri_3, axis=(1, 2))  # + (B, 1, 1, N, H)
            + jnp.expand_dims(tri_e_1, axis=3)  # + (B, N, N, 1, H)
            + jnp.expand_dims(tri_e_2, axis=2)  # + (B, N, 1, N, H)
            + jnp.expand_dims(tri_e_3, axis=1)  # + (B, 1, N, N, H)
            + jnp.expand_dims(tri_g, axis=(1, 2, 3))  # + (B, 1, 1, 1, H)
        )


class MLP(nnx.Module):
    """MLP module."""

    def __init__(self, in_size: int, sizes: List[int], rngs: nnx.Rngs):
        super().__init__()
        self.in_size = in_size
        self.layers = []
        for i in range(len(sizes)):
            if i == 0:
                in_features = in_size
            else:
                in_features = sizes[i - 1]
            out_features = sizes[i]
            self.layers.append(nnx.Linear(in_features, out_features, rngs=rngs))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:
                x = jax.nn.relu(x)
        return x


class MessageModule(nnx.Module):
    def __call__(self, z: Array, edge_fts: Array, graph_fts: Array) -> Array:
        """Message module."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class SingleMessageModule(MessageModule):
    """Message module."""

    def __init__(
        self,
        z_size: int,
        edge_fts_size: int,
        graph_fts_size: int,
        mid_size: int,
        rngs: nnx.Rngs,
        msg_mlp_sizes: Optional[List[int]] = None,
        mid_act: Optional[_Fn] = None,
    ):
        super().__init__()
        self.mid_size = mid_size
        self.msg_mlp_sizes = msg_mlp_sizes
        self.mid_act = mid_act
        self.sender_message_f = nnx.Linear(z_size, self.mid_size, rngs=rngs)
        self.receiver_msg_f = nnx.Linear(z_size, self.mid_size, rngs=rngs)
        self.edge_msg_f = nnx.Linear(edge_fts_size, self.mid_size, rngs=rngs)
        self.graph_msg_f = nnx.Linear(graph_fts_size, self.mid_size, rngs=rngs)
        if msg_mlp_sizes is not None:
            self.msg_mlp_transform = MLP(
                in_size=self.mid_size, sizes=msg_mlp_sizes, rngs=rngs
            )

    def __call__(self, z: Array, edge_fts: Array, graph_fts: Array) -> Array:
        msg_receiver = self.sender_message_f(z)  # (B, N, H)
        msg_sender = self.receiver_msg_f(z)  # (B, N, H)
        msg_edge = self.edge_msg_f(edge_fts)  # (B, N, N, H)
        msg_graph = self.graph_msg_f(graph_fts)  # (B, N, H)
        msgs = (
            jnp.expand_dims(msg_receiver, axis=1)  #   (B, 1, N, H)
            + jnp.expand_dims(msg_sender, axis=2)  # + (B, N, 1, H)
            + msg_edge  # + (B, N, N, H)
            + jnp.expand_dims(msg_graph, axis=(1, 2))  # + (B, 1, 1, H)
        )

        if self.msg_mlp_sizes is not None:
            msgs = self.msg_mlp_transform(jax.nn.relu(msgs))  # (B, N, N, H)

        if self.mid_act is not None:
            msgs = self.mid_act(msgs)  # (B, N, N, H)

        return msgs


class DifferentialMessageModule(MessageModule):
    def __init__(
        self,
        z_size: int,
        edge_fts_size: int,
        graph_fts_size: int,
        mid_size: int,
        rngs: nnx.Rngs,
        msg_mlp_sizes: Optional[List[int]] = None,
        mid_act: Optional[_Fn] = None,
    ):
        super().__init__()
        self.positive_module = SingleMessageModule(
            z_size=z_size,
            edge_fts_size=edge_fts_size,
            graph_fts_size=graph_fts_size,
            mid_size=mid_size,
            rngs=rngs,
            msg_mlp_sizes=msg_mlp_sizes,
            mid_act=mid_act,
        )
        self.negative_module = SingleMessageModule(
            z_size=z_size,
            edge_fts_size=edge_fts_size,
            graph_fts_size=graph_fts_size,
            mid_size=mid_size,
            rngs=rngs,
            msg_mlp_sizes=msg_mlp_sizes,
            mid_act=mid_act,
        )

    def __call__(self, z: Array, edge_fts: Array, graph_fts: Array) -> Array:
        """Differential message module."""
        pos_msgs = self.positive_module(z, edge_fts, graph_fts)
        neg_msgs = self.negative_module(z, edge_fts, graph_fts)
        msgs = pos_msgs - neg_msgs
        return msgs


def smooth_floor(x, k=10.0, n_min=None, n_max=None):
    """
    Approximates floor(x) using a sum of sigmoid steps.

    Args:
        x: A scalar or JAX array of real values.
        k: Sharpness of the sigmoid (higher = closer to hard floor).
        n_range: How many integer steps to consider on either side of x.

    Returns:
        A smooth approximation to floor(x).
    """

    # Determine the integer range around x to evaluate sigmoids
    # x_floor = jnp.floor(x)
    if n_min is None or n_max is None:
        # If not provided, calculate min and max based on x
        # This assumes x is a JAX array
        x_min = jnp.min(x) - 1
        x_max = jnp.max(x) + 1

        n_min = jnp.floor(x_min)
        n_max = jnp.ceil(x_max)

    # Create a range of integers to sum over
    n_vals = jnp.arange(n_min, n_max)

    # Broadcast and compute sigmoids
    sigmoid_terms = jax.nn.sigmoid(k * (x[..., None] - n_vals))

    # Sum sigmoids and subtract 0.5 to center at floor(x)
    return jnp.sum(sigmoid_terms, axis=-1) + n_min - 1


def differentiable_mod(x, n: float, k=50.0, n_min=None, n_max=None):
    """
    Computes a differentiable version of the modulus operation.

    Args:
        x: A scalar or JAX array of real values.
        n: The modulus base (should be a positive integer).

    Returns:
        A smooth approximation to x % n.
    """
    # Compute the smooth floor and then use it to compute the mod
    smooth_floor_val = smooth_floor(x / n, k=k, n_min=n_min, n_max=n_max)
    return x - n * smooth_floor_val


class AggregationMode(StrEnum):
    """Aggregation modes for the processor."""

    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    MIN = "min"
    MOD_SUM = "mod_sum"
    ATTENTION = "attention"


class AggregationFunction(nnx.Module):
    """Base class for aggregation functions."""

    def __call__(self, msgs: Array, adj_mat: Array, z: Array, rng_key=None) -> Array:
        """Aggregate messages."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class SumAggregationFunction(AggregationFunction):
    """Sum aggregation function."""

    def __call__(self, msgs: Array, adj_mat: Array, z: Array, rng_key=None) -> Array:
        return jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)


class MaxAggregationFunction(AggregationFunction):
    """Max aggregation function."""

    def __call__(self, msgs: Array, adj_mat: Array, z: Array, rng_key=None) -> Array:
        maxarg = jnp.where(jnp.expand_dims(adj_mat, -1), msgs, -BIG_NUMBER)
        return jnp.max(maxarg, axis=1)


class MinAggregationFunction(AggregationFunction):
    """Min aggregation function."""

    def __call__(self, msgs: Array, adj_mat: Array, z: Array, rng_key=None) -> Array:
        minarg = jnp.where(jnp.expand_dims(adj_mat, -1), msgs, BIG_NUMBER)
        return jnp.min(minarg, axis=1)


class MeanAggregationFunction(AggregationFunction):
    """Mean aggregation function."""

    def __call__(self, msgs: Array, adj_mat: Array, z: Array, rng_key=None) -> Array:
        msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
        msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
        return msgs


class ModSumAggregationFunction(AggregationFunction):
    """Modulus sum aggregation function."""

    def __init__(
        self,
        n: float = 2,
        k: float = 50.0,
        n_min: float = -300,
        n_max: float = 300,
        n_is_learnable: bool = False,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.k = k
        if n_is_learnable and rngs is not None:
            self.n = nnx.Param(
                jax.nn.initializers.constant(n)(rngs.params(), ()),
                name="n",
            )
        else:
            self.n = n

    def __call__(self, msgs: Array, adj_mat: Array, z: Array, rng_key=None) -> Array:
        msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
        msgs = differentiable_mod(
            msgs, self.n, n_min=self.n_min, n_max=self.n_max, k=self.k
        )
        return msgs


class AttentionAggregationFunction(AggregationFunction):
    """Attention aggregation function."""

    def __init__(self, z_size: int, rngs: nnx.Rngs, softmax_temperature: float = 0.5):
        super().__init__()
        self.z_size = z_size
        self.sender_repr_f = nnx.Linear(z_size, z_size, rngs=rngs)
        self.receiver_repr_f = nnx.Linear(z_size, z_size, rngs=rngs)
        self.attention_weights = nnx.Linear(z_size, 1, rngs=rngs)
        self.softmax_temperature = softmax_temperature

    def __call__(self, msgs: Array, adj_mat: Array, z: Array, rng_key=None) -> Array:
        """Aggregate messages using attention weights."""
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        sender_repr = self.sender_repr_f(z)  # (B, N, H)
        receiver_repr = self.receiver_repr_f(z)  # (B, N, H)
        # Compute attention scores
        sender_repr = jnp.expand_dims(sender_repr, axis=1)  # (B, 1, N, H)
        receiver_repr = jnp.expand_dims(receiver_repr, axis=2)  # (B, N, 1, H)
        attention_scores = self.attention_weights(
            sender_repr + receiver_repr
        )  # (B, N, N, 1)
        attention_scores = jnp.squeeze(attention_scores, axis=-1)
        noise = jax.random.gumbel(rng_key, attention_scores.shape)
        attention_scores += noise  # Add noise for stability
        attention_scores /= self.softmax_temperature
        attention_weights: Array = jax.nn.softmax(
            attention_scores + (adj_mat - 1.0) * 1e9, axis=-1
        )
        msgs = msgs * jnp.expand_dims(attention_weights, -1)  # (B, N, N, H)
        return jnp.sum(msgs, axis=1)  # (B, N, H)


def make_aggregation_function(
    mode: AggregationMode,
    z_size: int,
    rngs: nnx.Rngs,
    modulus_n: float,
    mod_steepness: float,
    softmax_temperature: float,
    n_is_learnable: bool,
) -> AggregationFunction:
    """Factory function to create aggregation functions."""
    if mode == AggregationMode.SUM:
        return SumAggregationFunction()
    elif mode == AggregationMode.MAX:
        return MaxAggregationFunction()
    elif mode == AggregationMode.MIN:
        return MinAggregationFunction()
    elif mode == AggregationMode.MEAN:
        return MeanAggregationFunction()
    elif mode == AggregationMode.MOD_SUM:
        return ModSumAggregationFunction(
            n=modulus_n, k=mod_steepness, rngs=rngs, n_is_learnable=n_is_learnable
        )
    elif mode == AggregationMode.ATTENTION:
        return AttentionAggregationFunction(
            z_size=z_size, rngs=rngs, softmax_temperature=softmax_temperature
        )
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


class MessageWeightMLP(nnx.Module):
    """MLP for computing message weights."""

    def __init__(self, in_size: int, out_size: int, rngs: nnx.Rngs):
        super().__init__()
        self.layer1 = nnx.Linear(in_size, out_size, rngs=rngs)
        # self.layer2 = nnx.Linear(
        #     out_size, out_size, rngs=rngs, bias_init=jax.nn.initializers.constant(50.0)
        # )
        self.layer2 = nnx.Linear(out_size, out_size, rngs=rngs)

    def __call__(self, z: Array) -> Array:
        """Compute message weights."""
        return self.layer2(jax.nn.relu(self.layer1(z)))  # Apply MLP to compute weights


class PGN(Processor):
    """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

    def __init__(
        self,
        out_size: int,
        rngs: nnx.Rngs,
        mid_size: Optional[int] = None,
        mid_act: Optional[_Fn] = None,
        activation: Optional[_Fn] = jax.nn.relu,
        reduction_modes: list[AggregationMode] = [AggregationMode.MAX],
        msgs_mlp_sizes: Optional[List[int]] = None,
        use_ln: bool = False,
        use_triplets: bool = False,
        nb_triplet_fts: int = 8,
        gated: bool = False,
        differential_messages: bool = False,
        aggregation_weight_softmax: bool = False,
        constant_aggregation_weight_init: float | None = None,
        modulus_n: float = 2.0,
        softmax_temperature: float = 0.5,
        mod_steepness: float = 50.0,
        msg_weight_gumbel: bool = False,  # If True, use Gumbel softmax for message weights
        msg_weight_softmax_temperature: float = 1.0,  # Temperature for Gumbel softmax
        n_is_learnable: bool = False,
        per_node_agg_weights: bool = False,
        name: str = "mpnn_aggr",
    ):
        super().__init__(name=name)
        if mid_size is None:
            self.mid_size = out_size
        else:
            self.mid_size = mid_size
        self.out_size = out_size
        self.mid_act = mid_act
        self.activation = activation
        self.reduction_modes = reduction_modes
        self._msgs_mlp_sizes = msgs_mlp_sizes
        self.use_ln = use_ln
        self.use_triplets = use_triplets
        self.nb_triplet_fts = nb_triplet_fts
        self.gated = gated
        self.aggregation_weight_softmax = aggregation_weight_softmax
        self.msg_weight_gumbel = msg_weight_gumbel
        self.msg_weight_softmax_temperature = msg_weight_softmax_temperature
        self.per_node_agg_weights = per_node_agg_weights
        self.mean_message_weights = [0 for _ in self.reduction_modes]

        hidden_size = self.mid_size
        edge_fts_size = self.mid_size
        node_fts_size = self.mid_size
        graph_fts_size = self.mid_size
        z_size = node_fts_size + hidden_size

        message_module_constructor = (
            DifferentialMessageModule if differential_messages else SingleMessageModule
        )
        self.message_modules: List[MessageModule] = [
            message_module_constructor(
                z_size=z_size,
                edge_fts_size=edge_fts_size,
                graph_fts_size=graph_fts_size,
                mid_size=self.mid_size,
                rngs=rngs,
                msg_mlp_sizes=self._msgs_mlp_sizes,
                mid_act=self.mid_act,
            )
            for _ in range(len(self.reduction_modes))
        ]

        if not per_node_agg_weights:
            if constant_aggregation_weight_init is not None:
                # Initialize message weights to a constant value
                self.message_weights = nnx.Param(
                    jax.nn.initializers.constant(constant_aggregation_weight_init)(
                        rngs.params(), (len(self.reduction_modes),)
                    ),
                    name="message_weights",
                )
            else:
                self.message_weights = nnx.Param(
                    jax.random.normal(rngs.params(), (len(self.reduction_modes),)),
                    name="message_weights",
                )
        else:
            self.message_weight_mlp = MessageWeightMLP(
                in_size=z_size,
                out_size=len(self.reduction_modes),
                rngs=rngs,
            )

        self.aggregation_modules: List[AggregationFunction] = [
            make_aggregation_function(
                mode=reduction_mode,
                z_size=z_size,
                rngs=rngs,
                modulus_n=modulus_n,
                mod_steepness=mod_steepness,
                softmax_temperature=softmax_temperature,
                n_is_learnable=n_is_learnable,
            )
            for reduction_mode in self.reduction_modes
        ]

        if self.use_triplets:
            self.triplet_module = TripletMessageModule(
                nb_triplet_fts=nb_triplet_fts,
                z_size=z_size,
                edge_fts_size=edge_fts_size,
                graph_fts_size=graph_fts_size,
                rngs=rngs,
            )
            self.o3 = nnx.Linear(nb_triplet_fts, self.out_size, rngs=rngs)

        self.o1 = nnx.Linear(z_size, self.out_size, rngs=rngs)
        self.o2 = nnx.Linear(self.out_size, self.out_size, rngs=rngs)

        if self.use_ln:
            self.ln = nnx.LayerNorm(
                num_features=self.out_size,
                # axis=-1,
                feature_axes=-1,  # maybe this should be reduction_axes?
                use_bias=True,
                use_scale=True,
                rngs=rngs,
            )

        if self.gated:
            self.gate1 = nnx.Linear(z_size, self.out_size, rngs=rngs)
            self.gate2 = nnx.Linear(self.out_size, self.out_size, rngs=rngs)
            self.gate3 = nnx.Linear(
                self.out_size,
                self.out_size,
                rngs=rngs,
                bias_init=jax.nn.initializers.constant(-3.0),
            )  # Initialise bias to -3

    def message(self, z: Array, edge_fts: Array, graph_fts: Array) -> List[Array]:
        """Message function.
        z: Node features. (B, N, Z)
        edge_fts: Edge features. (B, N, N, H)
        graph_fts: Graph features. (B, H)
        Returns:
            msgs: Messages. (B, N, N, H)
        """
        msgs = [
            message_module(z, edge_fts, graph_fts)
            for message_module in self.message_modules
        ]

        return msgs  # [(B, N, N, H)]

    def aggregate(
        self, msgs: list[Array], adj_mat: Array, z: Array, rng_key=None
    ) -> Array:
        """Message aggregation function.
        msgs: Messages. (B, N, N, H)
        adj_mat: Graph adjacency matrix. (B, N, N)
        Returns:
            msgs: Aggregated messages. (B, N, H)
        """
        msgs_stacked = jnp.stack(
            [
                aggregation_module(msg, adj_mat, z=z, rng_key=rng_key)  # (B, N, H)
                for msg, aggregation_module in zip(msgs, self.aggregation_modules)
            ],
            axis=3,
        )  # (B, N, H, R)
        # Multiply each message by its corresponding weight

        if not self.per_node_agg_weights:
            weights = self.message_weights[...]  # (R,)
            if self.aggregation_weight_softmax:
                # Apply softmax to the weights
                if rng_key is not None and self.msg_weight_gumbel:
                    weights += jax.random.gumbel(key=rng_key, shape=weights.shape)
                weights = jax.nn.softmax(weights / self.msg_weight_softmax_temperature)
            weights = weights[None, None, None, :]  # (1, 1, 1, R)
        else:
            # Per-node aggregation weights
            weights = self.message_weight_mlp(z)  # (B, N, R)
            if self.aggregation_weight_softmax:
                # Apply softmax to the weights
                if rng_key is not None and self.msg_weight_gumbel:
                    weights += jax.random.gumbel(key=rng_key, shape=weights.shape)
                weights = jax.nn.softmax(
                    weights / self.msg_weight_softmax_temperature, axis=-1
                )  # (B, N, R)
            weights = jnp.expand_dims(weights, axis=-2)  # (B, N, 1, R)

        mean_message_weights = jax.lax.stop_gradient(
            jnp.mean(weights, axis=(0, 1, 2))
        )  # (R,)

        def set_mean_message_weights(mean_message_weights: Array):
            """Set the mean message weights."""
            self.mean_message_weights = mean_message_weights.tolist()

        jax.debug.callback(set_mean_message_weights, mean_message_weights)
        weighted_msgs = msgs_stacked * weights  # (B, N, H, R)
        # Aggregate messages across the reduction modes
        msgs_aggregated = jnp.sum(weighted_msgs, axis=-1)  # (B, N, H)
        return msgs_aggregated

    def update(self, z: Array, msgs: Array):
        h_1 = self.o1(z)  # (B, N, H)
        h_2 = self.o2(msgs)  # (B, N, H)
        ret = h_1 + h_2  # (B, N, H)
        if self.activation is not None:
            ret = self.activation(ret)
        return ret  # (B, N, H)

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
        self,
        node_fts: Array,
        edge_fts: Array,
        graph_fts: Array,
        adj_mat: Array,
        hidden: Array,
        rng_key: Optional[Array] = None,
        **unused_kwargs,
    ) -> Tuple[Array, Optional[Array]]:
        """MPNN inference step.
        Args:
          node_fts: Node features. (B, N, H)
          edge_fts: Edge features. (B, N, N, H)
          graph_fts: Graph features. (B, H)
          adj_mat: Graph adjacency matrix. (B, N, N)
          hidden: Hidden features. (B, N, H)
          **kwargs: Extra kwargs.
        """

        b, n, h = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        # Z = 2H
        z = jnp.concatenate([node_fts, hidden], axis=-1)  # (B, N, Z)

        tri_msgs = None

        if self.use_triplets:
            # Triplet messages, as done by Dudzik and Velickovic (2022)
            triplets = self.triplet_module(z, edge_fts, graph_fts)
            tri_msgs = self.o3(jnp.max(triplets, axis=1))  # (B, N, N, H)

            if self.activation is not None:
                tri_msgs = self.activation(tri_msgs)

        msgs = self.message(z, edge_fts, graph_fts)  # (B, N, N, H)

        # Message Aggregation
        agg_msgs = self.aggregate(msgs, adj_mat, z=z, rng_key=rng_key)  # (B, N, H)

        # Updated node features
        ret = self.update(z, agg_msgs)  # (B, N, H)

        if self.use_ln:
            ret = self.ln(ret)

        if self.gated:
            gate = jax.nn.sigmoid(
                self.gate3(jax.nn.relu(self.gate1(z) + self.gate2(agg_msgs)))
            )
            ret = ret * gate + hidden * (1 - gate)

        return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars

    @property
    def using_triplets(self) -> bool:
        """Whether to use triplet messages."""
        return self.use_triplets


class DeepSets(PGN):
    """Deep Sets (Zaheer et al., NeurIPS 2017)."""

    def __call__(
        self,
        node_fts: Array,
        edge_fts: Array,
        graph_fts: Array,
        adj_mat: Array,
        hidden: Array,
        **unused_kwargs,
    ) -> Tuple[Array, Optional[Array]]:
        assert adj_mat.ndim == 3
        adj_mat = jnp.ones_like(adj_mat) * jnp.eye(adj_mat.shape[-1])
        return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class MPNN(PGN):
    """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

    def __call__(
        self,
        node_fts: Array,
        edge_fts: Array,
        graph_fts: Array,
        adj_mat: Array,
        hidden: Array,
        **unused_kwargs,
    ) -> Tuple[Array, Optional[Array]]:
        adj_mat = jnp.ones_like(adj_mat)
        return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class PGNMask(PGN):
    """Masked Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

    @property
    def inf_bias(self):
        return True

    @property
    def inf_bias_edge(self):
        return True


# class MemNetMasked(Processor):
#     """Implementation of End-to-End Memory Networks.

#     Inspired by the description in https://arxiv.org/abs/1503.08895.
#     """

#     def __init__(
#         self,
#         vocab_size: int,
#         sentence_size: int,
#         linear_output_size: int,
#         embedding_size: int = 16,
#         memory_size: Optional[int] = 128,
#         num_hops: int = 1,
#         nonlin: Callable[[Any], Any] = jax.nn.relu,
#         apply_embeddings: bool = True,
#         init_func: hk.initializers.Initializer = jnp.zeros,
#         use_ln: bool = False,
#         name: str = "memnet",
#     ) -> None:
#         """Constructor.

#         Args:
#           vocab_size: the number of words in the dictionary (each story, query and
#             answer come contain symbols coming from this dictionary).
#           sentence_size: the dimensionality of each memory.
#           linear_output_size: the dimensionality of the output of the last layer
#             of the model.
#           embedding_size: the dimensionality of the latent space to where all
#             memories are projected.
#           memory_size: the number of memories provided.
#           num_hops: the number of layers in the model.
#           nonlin: non-linear transformation applied at the end of each layer.
#           apply_embeddings: flag whether to aply embeddings.
#           init_func: initialization function for the biases.
#           use_ln: whether to use layer normalisation in the model.
#           name: the name of the model.
#         """
#         super().__init__(name=name)
#         self._vocab_size = vocab_size
#         self._embedding_size = embedding_size
#         self._sentence_size = sentence_size
#         self._memory_size = memory_size
#         self._linear_output_size = linear_output_size
#         self._num_hops = num_hops
#         self._nonlin = nonlin
#         self._apply_embeddings = apply_embeddings
#         self._init_func = init_func
#         self._use_ln = use_ln
#         # Encoding part: i.e. "I" of the paper.
#         self._encodings = _position_encoding(sentence_size, embedding_size)

#     def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
#         self,
#         node_fts: _Array,
#         edge_fts: _Array,
#         graph_fts: _Array,
#         adj_mat: _Array,
#         hidden: _Array,
#         **unused_kwargs,
#     ) -> _Array:
#         """MemNet inference step."""

#         del hidden
#         node_and_graph_fts = jnp.concatenate([node_fts, graph_fts[:, None]], axis=1)
#         edge_fts_padded = jnp.pad(
#             edge_fts * adj_mat[..., None], ((0, 0), (0, 1), (0, 1), (0, 0))
#         )
#         nxt_hidden = jax.vmap(self._apply, (1), 1)(node_and_graph_fts, edge_fts_padded)

#         # Broadcast hidden state corresponding to graph features across the nodes.
#         nxt_hidden = nxt_hidden[:, :-1] + nxt_hidden[:, -1:]
#         return nxt_hidden, None  # pytype: disable=bad-return-type  # numpy-scalars

#     def _apply(self, queries: _Array, stories: _Array) -> _Array:
#         """Apply Memory Network to the queries and stories.

#         Args:
#           queries: Tensor of shape [batch_size, sentence_size].
#           stories: Tensor of shape [batch_size, memory_size, sentence_size].

#         Returns:
#           Tensor of shape [batch_size, vocab_size].
#         """
#         if self._apply_embeddings:
#             query_biases = hk.get_parameter(
#                 "query_biases",
#                 shape=[self._vocab_size - 1, self._embedding_size],
#                 init=self._init_func,
#             )
#             stories_biases = hk.get_parameter(
#                 "stories_biases",
#                 shape=[self._vocab_size - 1, self._embedding_size],
#                 init=self._init_func,
#             )
#             memory_biases = hk.get_parameter(
#                 "memory_contents",
#                 shape=[self._memory_size, self._embedding_size],
#                 init=self._init_func,
#             )
#             output_biases = hk.get_parameter(
#                 "output_biases",
#                 shape=[self._vocab_size - 1, self._embedding_size],
#                 init=self._init_func,
#             )

#             nil_word_slot = jnp.zeros([1, self._embedding_size])

#         # This is "A" in the paper.
#         if self._apply_embeddings:
#             stories_biases = jnp.concatenate([stories_biases, nil_word_slot], axis=0)
#             memory_embeddings = jnp.take(
#                 stories_biases, stories.reshape([-1]).astype(jnp.int32), axis=0
#             ).reshape(list(stories.shape) + [self._embedding_size])
#             memory_embeddings = jnp.pad(
#                 memory_embeddings,
#                 (
#                     (0, 0),
#                     (0, self._memory_size - jnp.shape(memory_embeddings)[1]),
#                     (0, 0),
#                     (0, 0),
#                 ),
#             )
#             memory = jnp.sum(memory_embeddings * self._encodings, 2) + memory_biases
#         else:
#             memory = stories

#         # This is "B" in the paper. Also, when there are no queries (only
#         # sentences), then there these lines are substituted by
#         # query_embeddings = 0.1.
#         if self._apply_embeddings:
#             query_biases = jnp.concatenate([query_biases, nil_word_slot], axis=0)
#             query_embeddings = jnp.take(
#                 query_biases, queries.reshape([-1]).astype(jnp.int32), axis=0
#             ).reshape(list(queries.shape) + [self._embedding_size])
#             # This is "u" in the paper.
#             query_input_embedding = jnp.sum(query_embeddings * self._encodings, 1)
#         else:
#             query_input_embedding = queries

#         # This is "C" in the paper.
#         if self._apply_embeddings:
#             output_biases = jnp.concatenate([output_biases, nil_word_slot], axis=0)
#             output_embeddings = jnp.take(
#                 output_biases, stories.reshape([-1]).astype(jnp.int32), axis=0
#             ).reshape(list(stories.shape) + [self._embedding_size])
#             output_embeddings = jnp.pad(
#                 output_embeddings,
#                 (
#                     (0, 0),
#                     (0, self._memory_size - jnp.shape(output_embeddings)[1]),
#                     (0, 0),
#                     (0, 0),
#                 ),
#             )
#             output = jnp.sum(output_embeddings * self._encodings, 2)
#         else:
#             output = stories

#         intermediate_linear = hk.Linear(self._embedding_size, with_bias=False)

#         # Output_linear is "H".
#         output_linear = hk.Linear(self._linear_output_size, with_bias=False)

#         for hop_number in range(self._num_hops):
#             query_input_embedding_transposed = jnp.transpose(
#                 jnp.expand_dims(query_input_embedding, -1), [0, 2, 1]
#             )

#             # Calculate probabilities.
#             probs = jax.nn.softmax(
#                 jnp.sum(memory * query_input_embedding_transposed, 2)
#             )

#             # Calculate output of the layer by multiplying by C.
#             transposed_probs = jnp.transpose(jnp.expand_dims(probs, -1), [0, 2, 1])
#             transposed_output_embeddings = jnp.transpose(output, [0, 2, 1])

#             # This is "o" in the paper.
#             layer_output = jnp.sum(transposed_output_embeddings * transposed_probs, 2)

#             # Finally the answer
#             if hop_number == self._num_hops - 1:
#                 # Please note that in the TF version we apply the final linear layer
#                 # in all hops and this results in shape mismatches.
#                 output_layer = output_linear(query_input_embedding + layer_output)
#             else:
#                 output_layer = intermediate_linear(query_input_embedding + layer_output)

#             query_input_embedding = output_layer
#             if self._nonlin:
#                 output_layer = self._nonlin(output_layer)

#         # This linear here is "W".
#         ret = hk.Linear(self._vocab_size, with_bias=False)(output_layer)

#         if self._use_ln:
#             ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
#             ret = ln(ret)

#         return ret


# class MemNetFull(MemNetMasked):
#     """Memory Networks with full adjacency matrix."""

#     def __call__(
#         self,
#         node_fts: _Array,
#         edge_fts: _Array,
#         graph_fts: _Array,
#         adj_mat: _Array,
#         hidden: _Array,
#         **unused_kwargs,
#     ) -> _Array:
#         adj_mat = jnp.ones_like(adj_mat)
#         return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


ProcessorFactory = Callable[[int, nnx.Rngs], Processor]


class ProcessorKind(StrEnum):
    """Enum for processor kinds."""

    DEEP_SETS = "deepsets"
    GAT = "gat"
    GAT_FULL = "gat_full"
    GATV2 = "gatv2"
    GATV2_FULL = "gatv2_full"
    MPNN = "mpnn"
    PGN = "pgn"
    PGN_MASK = "pgn_mask"
    TRIPLET_MPNN = "triplet_mpnn"
    TRIPLET_PGN = "triplet_pgn"
    TRIPLET_PGN_MASK = "triplet_pgn_mask"
    GPBN = "gpbn"
    GPBN_MASK = "gpbn_mask"
    GMPNN = "gmpnn"
    TRIPLET_GPBN = "triplet_gpbn"


def get_processor_factory(
    kind: ProcessorKind,
    use_ln: bool,
    nb_triplet_fts: int,
    nb_heads: int = 4,
    reduction: list[AggregationMode] = [AggregationMode.MAX],
) -> ProcessorFactory:
    """Returns a processor factory.

    Args:
      kind: One of the available types of processor.
      use_ln: Whether the processor passes the output through a layernorm layer.
      nb_triplet_fts: How many triplet features to compute.
      nb_heads: Number of attention heads for GAT processors.
    Returns:
      A callable that takes an `out_size` parameter (equal to the hidden
      dimension of the network) and returns a processor instance.
    """

    def _factory(out_size: int, rngs: nnx.Rngs) -> Processor:
        if kind == ProcessorKind.DEEP_SETS:
            processor = DeepSets(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=False,
                nb_triplet_fts=0,
                rngs=rngs,
            )
        elif kind == ProcessorKind.GAT:
            processor = GAT(
                out_size=out_size,
                nb_heads=nb_heads,
                use_ln=use_ln,
                rngs=rngs,
            )
        elif kind == ProcessorKind.GAT_FULL:
            processor = GATFull(
                out_size=out_size, nb_heads=nb_heads, use_ln=use_ln, rngs=rngs
            )
        elif kind == ProcessorKind.GATV2:
            processor = GATv2(
                out_size=out_size, nb_heads=nb_heads, use_ln=use_ln, rngs=rngs
            )
        elif kind == ProcessorKind.GATV2_FULL:
            processor = GATv2Full(
                out_size=out_size, nb_heads=nb_heads, use_ln=use_ln, rngs=rngs
            )
        # elif kind == "memnet_full":
        #     processor = MemNetFull(
        #         vocab_size=out_size,
        #         sentence_size=out_size,
        #         linear_output_size=out_size,
        #     )
        # elif kind == "memnet_masked":
        #     processor = MemNetMasked(
        #         vocab_size=out_size,
        #         sentence_size=out_size,
        #         linear_output_size=out_size,
        #     )
        elif kind == ProcessorKind.MPNN:
            processor = MPNN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=False,
                nb_triplet_fts=0,
                rngs=rngs,
                reduction_modes=reduction,
            )
        elif kind == ProcessorKind.PGN:
            processor = PGN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=False,
                nb_triplet_fts=0,
                rngs=rngs,
            )
        elif kind == ProcessorKind.PGN_MASK:
            processor = PGNMask(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=False,
                nb_triplet_fts=0,
                rngs=rngs,
            )
        elif kind == ProcessorKind.TRIPLET_MPNN:
            processor = MPNN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=True,
                nb_triplet_fts=nb_triplet_fts,
                rngs=rngs,
            )
        elif kind == ProcessorKind.TRIPLET_PGN:
            processor = PGN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=True,
                nb_triplet_fts=nb_triplet_fts,
                rngs=rngs,
            )
        elif kind == ProcessorKind.TRIPLET_PGN_MASK:
            processor = PGNMask(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=True,
                nb_triplet_fts=nb_triplet_fts,
                rngs=rngs,
            )
        elif kind == ProcessorKind.GPBN:
            processor = PGN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=False,
                nb_triplet_fts=nb_triplet_fts,
                gated=True,
                rngs=rngs,
            )
        elif kind == ProcessorKind.GPBN_MASK:
            processor = PGNMask(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=False,
                nb_triplet_fts=nb_triplet_fts,
                gated=True,
                rngs=rngs,
            )
        elif kind == ProcessorKind.GMPNN:
            processor = MPNN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=False,
                nb_triplet_fts=nb_triplet_fts,
                gated=True,
                rngs=rngs,
            )
        elif kind == ProcessorKind.TRIPLET_GPBN:
            processor = PGN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=True,
                nb_triplet_fts=nb_triplet_fts,
                gated=True,
                rngs=rngs,
            )
        elif kind == ProcessorKind.TRIPLET_GPBN_MASK:
            processor = PGNMask(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=True,
                nb_triplet_fts=nb_triplet_fts,
                gated=True,
                rngs=rngs,
            )
        elif kind == ProcessorKind.TRIPLET_GMPNN:
            processor = MPNN(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,
                use_triplets=True,
                nb_triplet_fts=nb_triplet_fts,
                gated=True,
                rngs=rngs,
            )
        else:
            raise ValueError("Unexpected processor kind " + kind)

        return processor

    return _factory


# def _position_encoding(sentence_size: int, embedding_size: int) -> np.ndarray:
#     """Position Encoding described in section 4.1 [1]."""
#     encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
#     ls = sentence_size + 1
#     le = embedding_size + 1
#     for i in range(1, le):
#         for j in range(1, ls):
#             encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
#     encoding = 1 + 4 * encoding / embedding_size / sentence_size
#     return np.transpose(encoding)
