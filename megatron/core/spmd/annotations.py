"""Megatron-specific helpers for working with spmd_types.

These helpers encode Megatron's conventions for how parameters are distributed
and where gradient reductions occur. Other codebases will need similar helpers,
but the specific types will differ.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import spmd_types as spmd
import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher


def _tensor_parallel_type(tensor: torch.Tensor) -> spmd.SpmdType:
    """Infer a parameter's TP type from Megatron's distribution metadata.

    ``megatron.core.tensor_parallel.layers.set_tensor_model_parallel_attributes``
    marks physically sharded parameters with ``tensor_model_parallel`` and records
    their global shard dimension in ``partition_dim``. Column-parallel weights use
    dimension 0, while row-parallel weights use dimension 1.

    ``sequence_parallel`` means something different when attached to a
    parameter: the parameter value is replicated, but each TP rank accumulates
    a gradient from its local sequence shard. The
    ``_allreduce_non_tensor_model_parallel_grads`` helper in
    ``megatron.core.distributed.finalize_model_grads`` reads this attribute and
    sums those partial gradients across TP, which is exactly the R (replicated
    value, partial gradient) contract.

    Other trainable parameters are invariant across TP. Non-trainable tensors
    are treated as replicated because only their forward value matters.
    """
    if getattr(tensor, "tensor_model_parallel", False):
        return spmd.S(getattr(tensor, "partition_dim"))
    if getattr(tensor, "sequence_parallel", False):
        return spmd.R
    return spmd.I if tensor.requires_grad else spmd.R


def _expert_tensor_parallel_type(tensor: torch.Tensor) -> spmd.SpmdType:
    """Infer an expert parameter's type on the expert tensor-parallel axis.

    This follows the same conventions as ``_tensor_parallel_type`` with one
    exception: ``TEGroupedLinear`` in
    ``megatron.core.extensions.transformer_engine`` deliberately stamps
    ``partition_dim`` on expert weights *without* setting
    ``tensor_model_parallel`` (which would change Megatron's num-zeros gradient
    accounting), so here the presence of ``partition_dim`` alone marks a shard.
    """
    if getattr(tensor, "tensor_model_parallel", False) or hasattr(tensor, "partition_dim"):
        return spmd.S(getattr(tensor, "partition_dim"))
    if getattr(tensor, "sequence_parallel", False):
        return spmd.R
    return spmd.I if tensor.requires_grad else spmd.R


def annotate_tensor(
    tensor: torch.Tensor, *, tensor_parallel_type: spmd.SpmdType | None = None
) -> None:
    """Annotate a tensor that has no gradient to reduce: a buffer or frozen parameter.

    Such a tensor is simply replicated on every axis the model does not shard.
    """
    spmd.assert_type(
        tensor,
        {"TP": tensor_parallel_type or _tensor_parallel_type(tensor), "DP": spmd.R, "CP": spmd.R},
    )


def _annotate_dense_parameter(tensor: torch.Tensor) -> None:
    """Annotate a trainable parameter on the axis the model shards it over.

    Only TP is the model's to declare. Whether the parameter is replicated across
    data parallelism, and that its gradient gets summed there, is a promise made
    by whichever wrapper performs that reduction (see
    ``DistributedDataParallel``), so it is left for that wrapper to assert.
    """
    spmd.assert_type(tensor, {"TP": _tensor_parallel_type(tensor)})


def model_parallel_mesh() -> dict[str, spmd.MeshAxis]:
    """Return the mesh Megatron's dense layers are typed against.

    Megatron does not build a DeviceMesh; it hands out process groups from
    ``parallel_state``. The three that matter for typing dense computation are
    tensor parallel, data parallel (without context parallel folded in), and
    context parallel.
    """
    return {
        "TP": spmd.MeshAxis.of(parallel_state.get_tensor_model_parallel_group()),
        "DP": spmd.MeshAxis.of(parallel_state.get_data_parallel_group(with_context_parallel=False)),
        "CP": spmd.MeshAxis.of(parallel_state.get_context_parallel_group()),
    }


def expert_parallel_mesh() -> dict[str, spmd.MeshAxis]:
    """Return the expert factorization of the same ranks as ``model_parallel_mesh``.

    Megatron routes MoE experts with a separate set of process groups: experts
    are distributed over EP, each expert's weights are optionally sharded over
    ETP, and the remaining ranks (EDP) hold replicas. These groups tile the
    same ranks as TP x DP x CP but cut them differently, so tensors crossing
    between the two regions must be explicitly reinterpreted.
    """
    return {
        "ETP": spmd.MeshAxis.of(parallel_state.get_expert_tensor_parallel_group()),
        "EDP": spmd.MeshAxis.of(parallel_state.get_expert_data_parallel_group()),
        "EP": spmd.MeshAxis.of(parallel_state.get_expert_model_parallel_group()),
    }


def _is_expert_parallel_parameter(tensor: torch.Tensor) -> bool:
    """Whether Megatron reduces this parameter's gradient over EDP rather than DP.

    Megatron has no positive marker for this. Expert modules clear the
    ``allreduce`` attribute when expert parallelism is on, meaning "do not
    all-reduce this gradient over the regular data-parallel group", and
    ``DistributedDataParallel``, the optimizer, and Megatron-FSDP all recover
    expert-parallel membership from it with this same expression.
    """
    return not getattr(tensor, "allreduce", True)


def _annotate_expert_parameter(tensor: torch.Tensor) -> None:
    """Annotate an expert parameter on the axes the model shards it over.

    Each EP rank owns different experts, so on the EP axis the parameter is
    varying; within one expert the weight may be sharded over ETP. Replication
    over EDP is, as for dense parameters, the gradient reducer's promise.
    """
    spmd.assert_type(
        tensor,
        {
            parallel_state.get_expert_tensor_parallel_group(): _expert_tensor_parallel_type(tensor),
            parallel_state.get_expert_model_parallel_group(): spmd.V,
        },
    )


@contextmanager
def expert_parallel_region() -> Iterator[None]:
    """Type-check the enclosed expert computation on the ETP x EDP x EP mesh.

    Tensors produced inside the region carry expert-mesh types. Pass any that
    flow back out through ``leave_expert_parallel_region``.
    """
    with spmd.set_current_mesh(expert_parallel_mesh()):
        yield


def leave_expert_parallel_region(tensor: torch.Tensor) -> torch.Tensor:
    """Re-express an ``expert_parallel_region`` result on the enclosing mesh.

    Call this after the ``with`` block has exited, so the enclosing mesh is
    current again. Both meshes cover the same ranks, so this only retags the
    tensor; no data moves.
    """
    if not spmd.is_type_checking():
        return tensor
    return spmd.reinterpret_mesh(tensor, spmd.current_mesh())


def _annotate_vocab_bounds(module: VocabParallelEmbedding) -> None:
    """Type the per-rank vocabulary bounds that the embedding masks against.

    They are Python ints, so ordinary propagation cannot see that comparing
    token ids against them produces rank-varying results. Wrapping them as
    typed scalars makes the variation visible at its source; outside a
    type-checking run a ``Scalar`` behaves as its plain value.
    """
    bound_type = {"TP": spmd.V, "DP": spmd.R, "CP": spmd.R}
    module.vocab_start_index = spmd.Scalar(module.vocab_start_index, bound_type)
    module.vocab_end_index = spmd.Scalar(module.vocab_end_index, bound_type)


def _annotate_dispatcher_state(dispatcher: MoEFlexTokenDispatcher) -> None:
    """Type the per-rank overflow flag a HybridEP dispatcher accumulates across steps.

    Each rank ORs its own dispatch overflow into this flag, so it varies
    everywhere. Paged stashing all-reduces it before acting on it.
    """
    over_budget = dispatcher.check_over_budget()
    if over_budget is not None:
        spmd.assert_type(over_budget, {"TP": spmd.V, "DP": spmd.V, "CP": spmd.V})


def annotate_model(model: nn.Module) -> None:
    """Annotate model-owned tensors, and the few typed scalars, from Megatron's conventions.

    The model declares what it knows: how each tensor is sharded. The
    data-parallel axis of a trainable parameter is declared by the wrapper that
    reduces its gradient, so ``model`` should already be wrapped in
    ``DistributedDataParallel``; a bare model leaves that axis untyped and strict
    checking will say so at the first use.
    """
    for module in model.modules():
        if isinstance(module, VocabParallelEmbedding):
            _annotate_vocab_bounds(module)
        # Token dispatchers are plain objects owned by the MoE layer, not submodules.
        dispatcher = getattr(module, "token_dispatcher", None)
        if isinstance(dispatcher, MoEFlexTokenDispatcher):
            _annotate_dispatcher_state(dispatcher)
        for tensor in module.parameters(recurse=False):
            if not tensor.requires_grad:
                annotate_tensor(tensor)
            elif _is_expert_parallel_parameter(tensor):
                _annotate_expert_parameter(tensor)
            else:
                _annotate_dense_parameter(tensor)
        # DistributedDataParallel shadows ``buffers`` with its list of gradient
        # buffers, so call the nn.Module method explicitly.
        for tensor in nn.Module.buffers(module, recurse=False):
            annotate_tensor(tensor)
        # A few production modules, including YarnRotaryEmbedding, own replicated
        # constant tensors without registering them as buffers.
        for value in vars(module).values():
            if isinstance(value, torch.Tensor) and not value.requires_grad:
                annotate_tensor(value, tensor_parallel_type=spmd.R)
