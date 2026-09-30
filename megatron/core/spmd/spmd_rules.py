"""SPMD contracts for Megatron operators and custom autograd functions."""

from __future__ import annotations

import inspect

import spmd_types as spmd
import torch
from spmd_types import rules

from megatron.core import utils as megatron_utils
from megatron.core.extensions import transformer_engine as te_extension
from megatron.core.fusions import fused_bias_swiglu, fused_mla_yarn_rope_apply
from megatron.core.tensor_parallel import layers as tp_layers
from megatron.core.tensor_parallel import mappings as tp_mappings
from megatron.core.transformer import multi_token_prediction as mtp
from megatron.core.transformer.moe import fused_a2a, moe_utils


def rule_for(function):
    """Attach a type rule to an autograd function without changing its behavior."""

    def decorate(rule):
        if function is not None:
            function.spmd_typecheck = staticmethod(rule)
        return rule

    return decorate


@rule_for(tp_mappings._CopyToModelParallelRegion)
def _copy_to_model_parallel_region(*, input_, group):
    """Identity forward, all-reduce backward."""
    return rules.convert(input_, group, src=spmd.I, dst=spmd.R)


def _summed_type(x, group):
    """The type a reduction over ``group`` sums: a pending ``P``, else ``V``.

    ``rules.einsum`` makes a sharded contraction ``P`` on global mesh axes; on
    local ones it ignores shard dimensions and the product stays ``V``.
    """
    axis = spmd.normalize_axis(group)
    return spmd.P if spmd.get_local_type(x).get(axis) is spmd.P else spmd.V


@rule_for(tp_mappings._ReduceFromModelParallelRegion)
def _reduce_from_model_parallel_region(output, *, input_, group):
    """All-reduce forward, identity backward."""
    # I adds a gradient promise on top of R. A result with no gradient, such
    # as router token counts, is typed R so it can mix freely with V.
    dst = spmd.I if output.requires_grad else spmd.R
    rules.all_reduce(input_, group, src=_summed_type(input_, group), dst=dst, out=output)


@rule_for(tp_mappings._ScatterToModelParallelRegion)
def _scatter_to_model_parallel_region(*, input_, group):
    """Last-dimension split forward, gather backward."""
    return rules.convert(input_, group, src=spmd.I, dst=spmd.S(-1))


@rule_for(tp_mappings._GatherFromModelParallelRegion)
def _gather_from_model_parallel_region(*, input_, group):
    """Last-dimension gather forward, split backward."""
    return rules.all_gather(input_, group, src=spmd.S(-1), dst=spmd.I)


@rule_for(tp_mappings._ScatterToSequenceParallelRegion)
def _scatter_to_sequence_parallel_region(*, input_, group):
    """Sequence split forward, gather backward."""
    return rules.convert(input_, group, src=spmd.I, dst=spmd.S(0))


@rule_for(tp_mappings._GatherFromSequenceParallelRegion)
def _gather_from_sequence_parallel_region(*, input_, group, tensor_parallel_output_grad):
    """Sequence gather forward; the backward reduce-scatters or splits."""
    dst = spmd.R if tensor_parallel_output_grad else spmd.I
    return rules.all_gather(input_, group, src=spmd.S(0), dst=dst)


@rule_for(tp_mappings._ReduceScatterToSequenceParallelRegion)
def _reduce_scatter_to_sequence_parallel_region(*, input_, group):
    """Sequence reduce-scatter forward, gather backward."""
    return rules.reduce_scatter(input_, group, src=_summed_type(input_, group), dst=spmd.S(0))


@rule_for(tp_mappings._AllGatherFromTensorParallelRegion)
def _all_gather_from_tensor_parallel_region(*, input_, group):
    """Last-dimension all-gather forward, reduce-scatter backward."""
    return rules.all_gather(input_, group, src=spmd.S(-1), dst=spmd.R)


@rule_for(tp_mappings._ReduceScatterToTensorParallelRegion)
def _reduce_scatter_to_tensor_parallel_region(*, input_, group):
    """Last-dimension reduce-scatter forward, all-gather backward."""
    return rules.reduce_scatter(input_, group, src=_summed_type(input_, group), dst=spmd.S(-1))


@rule_for(tp_mappings._AllToAll)
def _all_to_all(*, group, input):
    """Self-adjoint token exchange."""
    return rules.all_to_all(input, group, src=spmd.S(0), dst=spmd.S(0))


@rule_for(tp_layers.LinearWithFrozenWeight)
def _linear_with_frozen_weight(*, input, weight, bias, allreduce_dgrad, tp_group):
    if allreduce_dgrad:
        input = rules.convert(input, tp_group, src=spmd.I, dst=spmd.R)
    output = rules.einsum("...k,nk->...n", input, weight)
    if bias is not None:
        output = rules.einsum("...n,n->...n", output, bias)
    return output


@rule_for(tp_layers.LinearWithGradAccumulationAndAsyncCommunication)
def _linear_with_grad_accumulation(
    *, input, weight, bias, allreduce_dgrad, sequence_parallel, tp_group
):
    """Trainable linear, with the sequence-parallel all-gather done inside."""
    if allreduce_dgrad:
        input = rules.convert(input, tp_group, src=spmd.I, dst=spmd.R)
    elif sequence_parallel:
        input = rules.all_gather(input, tp_group, src=spmd.S(0), dst=spmd.R)
    output = rules.einsum("...k,nk->...n", input, weight)
    if bias is not None:
        output = rules.einsum("...n,n->...n", output, bias)
    return output


@rule_for(moe_utils.MoEAuxLossAutoScaler)
def _moe_aux_loss_autoscaler(out, *, output):
    """The forward is the identity on ``output``.

    ``aux_loss`` only contributes a gradient in backward, so it must not
    influence the result's type; the generic local rule would let it.
    """
    rules.output(out, output)


@rule_for(mtp.MTPLossAutoScaler)
def _mtp_loss_autoscaler(out, *, output):
    """The forward is the identity on ``output``.

    ``mtp_loss`` only contributes a gradient in backward, so it must not
    influence the result's type; the generic local rule would let it.
    """
    rules.output(out, output)


@rule_for(fused_a2a.HybridEPDispatch)
def _hybrid_ep_dispatch(outputs, *, x, group):
    """Tokens are exchanged across ``group``, so every result varies along it.

    The group folds expert and expert-tensor parallelism, which the
    type-checking mesh may keep as separate axes, so the claim is made on each
    mesh axis the group contains. Other axes keep the input's type.
    """
    dispatched_hidden, dispatched_probs, _, tokens_per_expert, handle = outputs
    group_axis = spmd.MeshAxis.of(group)
    varying = {axis: spmd.V for axis in spmd.current_mesh() if axis <= group_axis}
    # The dispatcher reads the handle's overflow flag and combine reads its token
    # bookkeeping; the real and fake handles differ, so type every tensor in it.
    handle_tensors = [t for t in handle if isinstance(t, torch.Tensor)]
    for tensor in (dispatched_hidden, dispatched_probs, tokens_per_expert, *handle_tensors):
        spmd.assert_local_type_like(tensor, x, varying)


@rule_for(fused_a2a.HybridEPCombine)
def _hybrid_ep_combine(out, *, x):
    """Tokens return to their owners; the axes they crossed already vary in ``x``."""
    rules.output(out, x)


def _named_non_tensor_args(unpacker, values):
    """Name the values immediately unpacked into an implementation's locals."""
    first_local = len(inspect.signature(unpacker).parameters)
    names = unpacker.__code__.co_varnames[first_local : first_local + len(values)]
    if len(names) != len(values):
        raise NotImplementedError(
            f"Could not identify all non-tensor arguments for {unpacker.__qualname__}"
        )
    return dict(zip(names, values))


def _te_linear_input(inp, options):
    """The GEMM input after TE's input-side collective."""
    tp_group = options["tp_group"]
    if options["parallel_mode"] == "column":
        if options["sequence_parallel"]:
            return rules.all_gather(inp, tp_group, src=spmd.S(0), dst=spmd.R)
        return rules.convert(inp, tp_group, src=spmd.I, dst=spmd.R)
    return inp


def _te_linear_output(inp, weight, options):
    """TE's GEMM and output-side collective, given the GEMM input."""
    tp_group = options["tp_group"]
    parallel_mode = options["parallel_mode"]
    if parallel_mode not in ("column", "row", None):
        # Megatron passes None when it performs TP communication itself.
        raise NotImplementedError(f"Unsupported Transformer Engine linear mode {parallel_mode!r}")
    out = rules.einsum("...k,nk->...n", inp, weight)
    if parallel_mode == "row":
        src = _summed_type(out, tp_group)
        if options["sequence_parallel"]:
            return rules.reduce_scatter(out, tp_group, src=src, dst=spmd.S(0))
        return rules.all_reduce(out, tp_group, src=src, dst=spmd.I)
    return out


_te_extension = te_extension if te_extension.HAVE_TE else None


@rule_for(getattr(_te_extension, "_Linear", None))
def _te_linear(*, weight, inp, non_tensor_args):
    """Returns ``(out, weight_workspace)``; the workspace is ``weight`` recast."""
    from transformer_engine.pytorch.module import linear as te_linear

    options = _named_non_tensor_args(te_linear._linear_forward_impl, non_tensor_args)
    return _te_linear_output(_te_linear_input(inp, options), weight, options), weight


@rule_for(getattr(_te_extension, "_LayerNormLinear", None))
def _te_layernorm_linear(*, inp, ln_weight, weight, non_tensor_args):
    """Returns ``(out, ln_out, weight_workspace)``."""
    from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear

    options = _named_non_tensor_args(_LayerNormLinear.forward, non_tensor_args)
    # LayerNorm reads the whole hidden dim, per token.
    ln_out = rules.einsum("..._,_->..._", inp, ln_weight)
    gemm_input = _te_linear_input(ln_out, options)
    out = _te_linear_output(gemm_input, weight, options)
    if options["return_layernorm_output_gathered"]:
        ln_out = gemm_input
    return out, ln_out, weight


@rule_for(getattr(_te_extension, "CrossEntropyFunction", None))
def _te_parallel_cross_entropy(*, inp, target, reduce_loss, dist_process_group):
    """Cross-entropy, vocab-parallel over ``dist_process_group`` when it is set.

    The kernel all-gathers the softmax statistics, and each rank's logit
    gradient covers only its own vocab slice, which is the all-gather to ``I``.
    """
    if dist_process_group is not None:
        inp = rules.all_gather(inp, dist_process_group, src=spmd.S(-1), dst=spmd.I)
    if reduce_loss:
        return rules.einsum(f"{'_' * inp.ndim},{'_' * target.ndim}->", inp, target)
    return rules.einsum("..._,...->...", inp, target)


LOCAL_AUTOGRAD_FUNCTIONS = [
    megatron_utils.MakeViewlessTensor,
    fused_bias_swiglu.SwiGLUFunction,
    fused_mla_yarn_rope_apply.ApplyMLARotaryEmbQ,
    fused_mla_yarn_rope_apply.ApplyMLARotaryEmbKV,
    moe_utils.RouterGatingLinearFunction,
]

if te_extension.HAVE_TE:
    LOCAL_AUTOGRAD_FUNCTIONS.extend(
        function
        for name in (
            "_OperationFuserAutogradFunction",
            "FusedAttnFunc",
            "FusedAuxLoss",
            "FusedTopkScoreFunction",
            "FusedComputeScoresForMoEAuxLoss",
        )
        if (function := getattr(te_extension, name, None)) is not None
    )

for function in LOCAL_AUTOGRAD_FUNCTIONS:
    spmd.register_local_autograd_function(function)
