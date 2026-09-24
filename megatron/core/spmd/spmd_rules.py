"""SPMD contracts for Megatron operators and custom autograd functions."""

from __future__ import annotations

import inspect

import spmd_types as spmd

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
def _copy_to_model_parallel_region(output, *, input_, group):
    """Type the identity forward and all-reduce backward."""
    spmd.assert_type(input_, {group: spmd.I})
    spmd.assert_local_type_like(output, input_, {group: spmd.R})


@rule_for(tp_mappings._ReduceFromModelParallelRegion)
def _reduce_from_model_parallel_region(output, *, input_, group):
    """Type the all-reduce forward and identity backward."""
    spmd.assert_type(input_, {group: spmd.V})
    # I adds a gradient promise on top of R. A result with no gradient, such
    # as router token counts, is typed R so it can mix freely with V.
    output_type = spmd.I if output.requires_grad else spmd.R
    spmd.assert_local_type_like(output, input_, {group: output_type})


@rule_for(tp_mappings._ScatterToModelParallelRegion)
def _scatter_to_model_parallel_region(output, *, input_, group):
    """Type the last-dimension split and gather backward."""
    spmd.assert_type(input_, {group: spmd.I})
    spmd.assert_local_type_like(output, input_, {group: spmd.S(-1)})


@rule_for(tp_mappings._GatherFromModelParallelRegion)
def _gather_from_model_parallel_region(output, *, input_, group):
    """Type the last-dimension gather and split backward."""
    spmd.assert_type(input_, {group: spmd.S(-1)})
    spmd.assert_local_type_like(output, input_, {group: spmd.I})


@rule_for(tp_mappings._ScatterToSequenceParallelRegion)
def _scatter_to_sequence_parallel_region(output, *, input_, group):
    """Type the sequence split and gather backward."""
    spmd.assert_type(input_, {group: spmd.I})
    spmd.assert_local_type_like(output, input_, {group: spmd.S(0)})


@rule_for(tp_mappings._GatherFromSequenceParallelRegion)
def _gather_from_sequence_parallel_region(output, *, input_, group, tensor_parallel_output_grad):
    """Type the sequence gather and its selectable gradient collective."""
    spmd.assert_type(input_, {group: spmd.S(0)})
    output_type = spmd.R if tensor_parallel_output_grad else spmd.I
    spmd.assert_local_type_like(output, input_, {group: output_type})


@rule_for(tp_mappings._ReduceScatterToSequenceParallelRegion)
def _reduce_scatter_to_sequence_parallel_region(output, *, input_, group):
    """Type the sequence reduce-scatter and gather backward."""
    spmd.assert_type(input_, {group: spmd.V})
    spmd.assert_local_type_like(output, input_, {group: spmd.S(0)})


@rule_for(tp_mappings._AllGatherFromTensorParallelRegion)
def _all_gather_from_tensor_parallel_region(output, *, input_, group):
    """Type the last-dimension all-gather and reduce-scatter backward."""
    spmd.assert_type(input_, {group: spmd.S(-1)})
    spmd.assert_local_type_like(output, input_, {group: spmd.R})


@rule_for(tp_mappings._ReduceScatterToTensorParallelRegion)
def _reduce_scatter_to_tensor_parallel_region(output, *, input_, group):
    """Type the last-dimension reduce-scatter and all-gather backward."""
    spmd.assert_type(input_, {group: spmd.V})
    spmd.assert_local_type_like(output, input_, {group: spmd.S(-1)})


@rule_for(tp_mappings._AllToAll)
def _all_to_all(output, *, group, input):
    """Type the self-adjoint token exchange."""
    spmd.assert_type(input, {group: spmd.S(0)})
    spmd.assert_local_type_like(output, input, {group: spmd.S(0)})


@rule_for(tp_layers.LinearWithFrozenWeight)
def _linear_with_frozen_weight(output, *, input, weight, bias, allreduce_dgrad, tp_group):
    """Type the tensor-parallel contract of the completed frozen linear."""
    tp_axis = spmd.normalize_axis(tp_group)
    spmd.assert_type(weight, {tp_axis: spmd.V})
    if bias is not None:
        spmd.assert_type(bias, {tp_axis: spmd.V})
    if allreduce_dgrad:
        input_type = spmd.I
    elif spmd.get_axis_local_type(input, tp_axis) is spmd.R:
        # Sequence parallelism gathers S(0) to R before calling the frozen
        # linear. Other no-reduction paths consume V/S inputs.
        input_type = spmd.R
    else:
        input_type = spmd.V
    spmd.assert_type(input, {tp_axis: input_type})
    spmd.assert_local_type_like(output, input, {tp_axis: spmd.V})


@rule_for(tp_layers.LinearWithGradAccumulationAndAsyncCommunication)
def _linear_with_grad_accumulation(
    output, *, input, weight, bias, allreduce_dgrad, sequence_parallel, tp_group
):
    """Type the tensor-parallel contract of the completed trainable linear."""
    tp_axis = spmd.normalize_axis(tp_group)
    spmd.assert_type(weight, {tp_axis: spmd.V})
    if bias is not None:
        spmd.assert_type(bias, {tp_axis: spmd.V})
    if allreduce_dgrad:
        input_type = spmd.I
    elif sequence_parallel:
        input_type = spmd.S(0)
    else:
        input_type = spmd.V
    spmd.assert_type(input, {tp_axis: input_type})
    spmd.assert_local_type_like(output, input, {tp_axis: spmd.V})


@rule_for(moe_utils.MoEAuxLossAutoScaler)
def _moe_aux_loss_autoscaler(result, *, output):
    """The forward is the identity on ``output``.

    ``aux_loss`` only contributes a gradient in backward, so it must not
    influence the result's type; the generic local rule would let it.
    """
    spmd.assert_type_like(result, output)


@rule_for(mtp.MTPLossAutoScaler)
def _mtp_loss_autoscaler(result, *, output):
    """The forward is the identity on ``output``.

    ``mtp_loss`` only contributes a gradient in backward, so it must not
    influence the result's type; the generic local rule would let it.
    """
    spmd.assert_type_like(result, output)


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
    # The handle's last entry is this rank's overflow flag, which the dispatcher reads.
    for tensor in (dispatched_hidden, dispatched_probs, tokens_per_expert, handle[-1]):
        spmd.assert_local_type_like(tensor, x, varying)


@rule_for(fused_a2a.HybridEPCombine)
def _hybrid_ep_combine(output, *, x):
    """Tokens return to their owners; the axes they crossed already vary in ``x``."""
    spmd.assert_local_type_like(output, x)


def _named_non_tensor_args(unpacker, values):
    """Name the values immediately unpacked into an implementation's locals."""
    first_local = len(inspect.signature(unpacker).parameters)
    names = unpacker.__code__.co_varnames[first_local : first_local + len(values)]
    if len(names) != len(values):
        raise NotImplementedError(
            f"Could not identify all non-tensor arguments for {unpacker.__qualname__}"
        )
    return dict(zip(names, values))


def _assert_te_linear_types(out, inp, options):
    """State the TP contract shared by Transformer Engine linear functions."""
    tp_group = options["tp_group"]
    parallel_mode = options["parallel_mode"]
    if parallel_mode is None:
        # Megatron passes None when it performs TP communication itself.
        spmd.assert_local_type_like(out, inp)
    elif parallel_mode == "column":
        input_type = spmd.S(0) if options["sequence_parallel"] else spmd.I
        spmd.assert_type(inp, {tp_group: input_type})
        spmd.assert_local_type_like(out, inp, {tp_group: spmd.S(-1)})
    elif parallel_mode == "row":
        output_type = spmd.S(0) if options["sequence_parallel"] else spmd.I
        spmd.assert_type(inp, {tp_group: spmd.S(-1)})
        spmd.assert_local_type_like(out, inp, {tp_group: output_type})
    else:
        raise NotImplementedError(f"Unsupported Transformer Engine linear mode {parallel_mode!r}")


_te_extension = te_extension if te_extension.HAVE_TE else None


@rule_for(getattr(_te_extension, "_Linear", None))
def _te_linear(output, *, weight, inp, non_tensor_args):
    from transformer_engine.pytorch.module import linear as te_linear

    out, weight_workspace = output
    options = _named_non_tensor_args(te_linear._linear_forward_impl, non_tensor_args)
    _assert_te_linear_types(out, inp, options)
    if weight_workspace is not None:
        spmd.assert_local_type_like(weight_workspace, weight)


@rule_for(getattr(_te_extension, "_LayerNormLinear", None))
def _te_layernorm_linear(output, *, inp, weight, non_tensor_args):
    from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear

    out, ln_out, weight_workspace = output
    options = _named_non_tensor_args(_LayerNormLinear.forward, non_tensor_args)
    _assert_te_linear_types(out, inp, options)
    if ln_out is not None:
        if options["return_layernorm_output_gathered"]:
            spmd.assert_local_type_like(ln_out, inp, {options["tp_group"]: spmd.R})
        else:
            spmd.assert_local_type_like(ln_out, inp)
    if weight_workspace is not None:
        spmd.assert_local_type_like(weight_workspace, weight)


@rule_for(getattr(_te_extension, "CrossEntropyFunction", None))
def _te_parallel_cross_entropy(output, *, inp, target, dist_process_group):
    if dist_process_group is None:
        # No collective: logits, targets, and loss vary together.
        spmd.assert_local_type_like(inp, target)
        spmd.assert_local_type_like(output, target)
        return
    spmd.assert_type(inp, {dist_process_group: spmd.S(-1)})
    spmd.assert_type(target, {dist_process_group: spmd.I})
    spmd.assert_local_type_like(output, target)


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
