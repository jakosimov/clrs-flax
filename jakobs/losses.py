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
"""Utilities for calculating losses."""

from typing import List, Tuple
import chex
from clrs._src import probing
from clrs._src import specs

import haiku as hk
import jax
import jax.numpy as jnp
from jax import Array

_DataPoint = probing.DataPoint
_OutputClass = specs.OutputClass
_Type = specs.Type

EPS = 1e-12


def output_loss(truth: _DataPoint, pred: Array, nb_nodes: int) -> float:
    """Output loss for full-sample training."""

    if truth.type_ == _Type.SCALAR:
        total_loss = jnp.mean((pred - truth.data) ** 2)

    elif truth.type_ == _Type.MASK:
        loss = (
            jnp.maximum(pred, 0)
            - pred * truth.data
            + jnp.log1p(jnp.exp(-jnp.abs(pred)))
        )
        mask = (truth.data != _OutputClass.MASKED).astype(jnp.float32)
        total_loss = jnp.sum(loss * mask) / jnp.sum(mask)

    elif truth.type_ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
        masked_truth = truth.data * (truth.data != _OutputClass.MASKED).astype(
            jnp.float32
        )
        total_loss = -jnp.sum(masked_truth * jax.nn.log_softmax(pred)) / jnp.sum(
            truth.data == _OutputClass.POSITIVE
        )

    elif truth.type_ == _Type.POINTER:
        total_loss = jnp.mean(
            -jnp.sum(
                hk.one_hot(truth.data, nb_nodes) * jax.nn.log_softmax(pred), axis=-1
            )
        )

    elif truth.type_ == _Type.PERMUTATION_POINTER:
        # Predictions are NxN logits aiming to represent a doubly stochastic matrix.
        # Compute the cross entropy between doubly stochastic pred and truth_data
        total_loss = jnp.mean(-jnp.sum(truth.data * pred, axis=-1))

    return total_loss  # pytype: disable=bad-return-type  # jnp-type


def hint_loss(
    truth: _DataPoint,
    preds: List[Array],
    lengths: Array,
    nb_nodes: int,
    verbose: bool = False,
):
    """Hint loss for full-sample training."""
    total_loss = 0.0
    verbose_loss = {}
    length = truth.data.shape[0] - 1

    loss, mask = _hint_loss(
        truth_data=truth.data[1:],
        truth_type=truth.type_,
        pred=jnp.stack(preds),
        nb_nodes=nb_nodes,
    )
    mask *= _is_not_done_broadcast(lengths, jnp.arange(length)[:, None], loss)
    loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), EPS)
    if verbose:
        verbose_loss["loss_" + truth.name] = loss
    else:
        total_loss += loss

    return verbose_loss if verbose else total_loss


def _hint_loss(
    truth_data: Array,
    truth_type: str,
    pred: Array,
    nb_nodes: int,
) -> Tuple[Array, Array]:
    """Hint loss helper."""
    mask = None
    if truth_type == _Type.SCALAR:
        loss = (pred - truth_data) ** 2

    elif truth_type == _Type.MASK:
        loss = (
            jnp.maximum(pred, 0)
            - pred * truth_data
            + jnp.log1p(jnp.exp(-jnp.abs(pred)))
        )
        mask = (truth_data != _OutputClass.MASKED).astype(
            jnp.float32
        )  # pytype: disable=attribute-error  # numpy-scalars

    elif truth_type == _Type.MASK_ONE:
        loss = -jnp.sum(truth_data * jax.nn.log_softmax(pred), axis=-1, keepdims=True)

    elif truth_type == _Type.CATEGORICAL:
        loss = -jnp.sum(truth_data * jax.nn.log_softmax(pred), axis=-1)
        mask = jnp.any(truth_data == _OutputClass.POSITIVE, axis=-1).astype(jnp.float32)

    elif truth_type == _Type.POINTER:
        loss = -jnp.sum(
            hk.one_hot(truth_data, nb_nodes) * jax.nn.log_softmax(pred), axis=-1
        )

    elif truth_type == _Type.PERMUTATION_POINTER:
        # Predictions are NxN logits aiming to represent a doubly stochastic matrix.
        # Compute the cross entropy between doubly stochastic pred and truth_data
        loss = -jnp.sum(truth_data * pred, axis=-1)

    if mask is None:
        mask = jnp.ones_like(loss)
    return loss, mask


def _is_not_done_broadcast(lengths, i, tensor):
    is_not_done = (lengths > i + 1) * 1.0
    while len(is_not_done.shape) < len(
        tensor.shape
    ):  # pytype: disable=attribute-error  # numpy-scalars
        is_not_done = jnp.expand_dims(is_not_done, -1)
    return is_not_done
