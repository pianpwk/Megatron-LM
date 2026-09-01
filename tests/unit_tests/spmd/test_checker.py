from contextlib import contextmanager
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.testing._internal.distributed.fake_pg import FakeStore

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.spmd import annotate_model, typecheck
from megatron.core.tensor_parallel import mappings
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import CudaGraphScope, LayerType
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig

MLPERF_DEEPSEEK_V3_VOCAB_SIZE = 129280
MLPERF_DEEPSEEK_V3_SEQUENCE_LENGTH = 4096
MLPERF_DEEPSEEK_V3_PIPELINE_LAYOUT = "Et*4|(t*4|)*14tmL"


def _make_mlperf_deepseek_v3_config() -> MLATransformerConfig:
    """The MLPerf Training v6.0 NVIDIA GB200 DeepSeek-V3 configuration.

    This mirrors ``config_GB200_64x4x480xtp2pp4ep32cp1_mxfp8_full_cg.sh``.
    Launcher, optimizer, and data-loader flags that do not belong to
    ``MLATransformerConfig`` are intentionally outside this model-level test.
    """
    return MLATransformerConfig(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=4,
        sequence_parallel=True,
        context_parallel_size=1,
        expert_model_parallel_size=32,
        expert_tensor_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        deallocate_pipeline_outputs=False,
        microbatch_group_size_per_vp_stage=4,
        gradient_accumulation_fusion=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
        num_layers=61,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        pipeline_model_parallel_layout=MLPERF_DEEPSEEK_V3_PIPELINE_LAYOUT,
        hidden_size=7168,
        num_attention_heads=128,
        num_query_groups=128,
        ffn_hidden_size=18432,
        kv_channels=64,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-6,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        num_moe_experts=256,
        qk_layernorm=True,
        init_method_std=0.006,
        bias_activation_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=True,
        apply_rope_fusion=True,
        use_transformer_engine_op_fuser=True,
        fused_residual_rmsnorm=True,
        # The production recipe recomputes the dense MLP. Type checking does not
        # run out of memory without it, and with checkpointing off the checker
        # sees the MLP as ordinary operations instead of an opaque autograd
        # boundary whose output type it would otherwise have to take on trust.
        recompute_granularity=None,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param=True,
        fp8_wgrad=True,
        fp8_output_proj=True,
        fp8_dot_product_attention=True,
        high_priority_a2a_comm_stream=False,
        moe_shared_expert_intermediate_size=2048,
        moe_shared_expert_overlap=False,
        moe_layer_freq=[0] * 3 + [1] * 58,
        moe_ffn_hidden_size=2048,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_topk=8,
        moe_router_num_groups=8,
        moe_router_group_topk=4,
        moe_router_pre_softmax=True,
        moe_router_topk_scaling_factor=2.5,
        moe_router_score_function="sigmoid",
        moe_router_dtype="bf16",
        moe_router_enable_expert_bias=True,
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        moe_single_grouped_bias=False,
        moe_aux_loss_coeff=0.01,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_permute_fusion=True,
        moe_router_fusion=True,
        moe_hybridep_num_sms=32,
        moe_hybridep_num_sms_preprocessing=32,
        moe_mlp_glu_interleave_size=32,
        moe_expert_rank_capacity_factor=4,
        moe_paged_stash=True,
        moe_paged_stash_buffer_size_factor_cuda=1.5,
        moe_paged_stash_buffer_size_factor_cpu=1.0,
        # These tests call the model forward directly rather than through the
        # pipeline schedule, so the overlap features that need the schedule
        # driver are disabled.
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
        use_te_activation_func=False,
        cuda_graph_impl="local",
        cuda_graph_scope=[CudaGraphScope.full_iteration],
        cuda_graph_warmup_steps=2,
        use_te_rng_tracker=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        rotary_base=10000.0,
        rotary_scaling_factor=40,
        original_max_position_embeddings=4096,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="transformer_engine",
    )


def _build_mlperf_deepseek_v3_model(config: MLATransformerConfig, vp_stage: int) -> GPTModel:
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=True,
        normalization="RMSNorm",
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )
    mtp_spec = get_gpt_mtp_block_spec(
        config, block_spec, use_transformer_engine=True, vp_stage=vp_stage, pp_rank=pp_rank
    )
    local_layout = config.pipeline_model_parallel_layout.layout[pp_rank][vp_stage]
    return GPTModel(
        config=config,
        transformer_layer_spec=block_spec,
        vocab_size=MLPERF_DEEPSEEK_V3_VOCAB_SIZE,
        max_sequence_length=MLPERF_DEEPSEEK_V3_SEQUENCE_LENGTH,
        pre_process=LayerType.embedding in local_layout,
        post_process=LayerType.loss in local_layout,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        scatter_embedding_sequence_parallel=True,
        mtp_block_spec=mtp_spec,
        vp_stage=vp_stage,
    )


@pytest.fixture
def mlperf_environment(monkeypatch):
    """The part of the MLPerf recipe that lives in environment variables.

    ``config_GB200_64x4x480xtp2pp4ep32cp1_mxfp8_full_cg.sh`` exports
    ``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`` to select TE's CuTe DSL grouped-MLP kernel.
    The unit-test harness also forces TE's unfused attention through
    ``NVTE_*_ATTN``; the recipe leaves ``attention_backend=auto`` and lets
    ``GPTModel`` set those variables itself, so they are cleared here.
    """
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    for variable in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(variable, raising=False)


def _wrap_in_ddp(config: MLATransformerConfig, model: GPTModel) -> DistributedDataParallel:
    """Wrap the model the way the recipe trains it.

    MLPerf runs Megatron's DistributedDataParallel with the distributed optimizer.
    Besides installing the ``main_grad`` buffers that fused weight-gradient
    accumulation writes into, DDP is the component that promises parameters are
    replicated across data parallelism, since it performs that gradient
    reduction. The MXFP8 parameter-gather settings match ``config_common_mxfp8.sh``;
    the recipe's communication overlap flags are left off because they add
    parameter all-gathers that only make sense across optimizer steps.
    """
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        fp8_param_gather=True,
        reuse_grad_buf_for_mxfp8_param_ag=True,
    )
    return DistributedDataParallel(config, ddp_config, model)


@contextmanager
def _fake_distributed(rank: int, world_size: int, **model_parallel_kwargs):
    """Initialize a fake process group and, optionally, Megatron's parallel state.

    The fake backend lets a single process impersonate any rank of a large
    job. Collectives return placeholder data, which is all SPMD type checking
    needs to follow how types flow through them.
    """
    torch.distributed.init_process_group(
        backend="fake", store=FakeStore(), rank=rank, world_size=world_size
    )
    try:
        if model_parallel_kwargs:
            parallel_state.initialize_model_parallel(
                create_gloo_process_groups=False, **model_parallel_kwargs
            )
        yield
    finally:
        if model_parallel_kwargs:
            parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_native_mapping_spmd_contracts() -> None:
    with _fake_distributed(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    ):
        group = parallel_state.get_tensor_model_parallel_group()
        cases = (
            (
                (4, 8),
                spmd.I,
                spmd.R,
                lambda x: mappings.copy_to_tensor_model_parallel_region(x, group),
            ),
            (
                (4, 8),
                spmd.V,
                spmd.I,
                lambda x: mappings.reduce_from_tensor_model_parallel_region(x, group),
            ),
            (
                (4, 8),
                spmd.I,
                spmd.S(-1),
                lambda x: mappings.scatter_to_tensor_model_parallel_region(x, group),
            ),
            (
                (4, 4),
                spmd.S(-1),
                spmd.I,
                lambda x: mappings.gather_from_tensor_model_parallel_region(x, group),
            ),
            (
                (8, 4),
                spmd.I,
                spmd.S(0),
                lambda x: mappings.scatter_to_sequence_parallel_region(x, group),
            ),
            (
                (4, 4),
                spmd.S(0),
                spmd.R,
                lambda x: mappings.gather_from_sequence_parallel_region(x, True, group),
            ),
            (
                (4, 4),
                spmd.S(0),
                spmd.I,
                lambda x: mappings.gather_from_sequence_parallel_region(x, False, group),
            ),
            (
                (8, 4),
                spmd.V,
                spmd.S(0),
                lambda x: mappings.reduce_scatter_to_sequence_parallel_region(x, group),
            ),
            (
                (4, 4),
                spmd.S(-1),
                spmd.R,
                lambda x: mappings.all_gather_last_dim_from_tensor_parallel_region(x, group),
            ),
            (
                (4, 8),
                spmd.V,
                spmd.S(-1),
                lambda x: mappings.reduce_scatter_last_dim_to_tensor_parallel_region(x, group),
            ),
            ((8, 4), spmd.S(0), spmd.S(0), lambda x: mappings.all_to_all(group, x)),
        )

        inputs = [torch.randn(shape, device="cuda", requires_grad=True) for shape, _, _, _ in cases]
        counts = torch.ones((4, 8), device="cuda")
        with typecheck():
            for input_, (_, input_type, output_type, operation) in zip(inputs, cases):
                spmd.assert_type(input_, {"TP": input_type})
                output = operation(input_)
                expected_local_type = spmd.V if isinstance(output_type, spmd.Shard) else output_type
                assert spmd.get_axis_local_type(output, "TP") == expected_local_type
                if isinstance(output_type, spmd.Shard):
                    placements = [None] * output.ndim
                    placements[output_type.dim % output.ndim] = spmd.normalize_axis("TP")
                    assert spmd.get_partition_spec(output) == spmd.PartitionSpec(*placements)

            # Without a gradient there is no invariance promise to make, so the
            # all-reduce result is typed R rather than I.
            spmd.assert_type(counts, {"TP": spmd.V})
            reduced = mappings.reduce_from_tensor_model_parallel_region(counts, group)
            assert spmd.get_axis_local_type(reduced, "TP") is spmd.R


def test_expert_parameters_use_expert_parallel_mesh() -> None:
    with _fake_distributed(
        rank=0,
        world_size=8,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=2,
    ):
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.empty(4, 4))
        module.bias = torch.nn.Parameter(torch.empty(4))
        module.weight.allreduce = False
        module.weight.tensor_model_parallel = True
        module.weight.partition_dim = 1
        module.bias.allreduce = False

        with typecheck():
            annotate_model(module)

        expert_tp = parallel_state.get_expert_tensor_parallel_group()
        expert_dp = parallel_state.get_expert_data_parallel_group()
        expert_parallel = parallel_state.get_expert_model_parallel_group()
        assert spmd.get_axis_local_type(module.weight, expert_tp) is spmd.V
        assert spmd.get_axis_local_type(module.weight, expert_parallel) is spmd.V
        # Replication over EDP is asserted by the gradient reducer, not the model.
        assert spmd.maybe_get_axis_local_type(module.weight, expert_dp) is None
        assert spmd.get_partition_spec(module.weight) == spmd.PartitionSpec(
            None, spmd.MeshAxis.of(expert_tp)
        )
        assert spmd.get_axis_local_type(module.bias, expert_tp) is spmd.I
        assert spmd.get_axis_local_type(module.bias, expert_parallel) is spmd.V


def test_ddp_promises_replication_on_each_folded_axis() -> None:
    """DDP reduces over the folded DP x CP group but the mesh keeps DP and CP apart."""
    with _fake_distributed(
        rank=0, world_size=8, tensor_model_parallel_size=2, context_parallel_size=2
    ):
        config = TransformerConfig(
            num_layers=1, hidden_size=8, num_attention_heads=1, use_cpu_initialization=True
        )
        module = torch.nn.Linear(8, 8, bias=False).cuda()
        model = DistributedDataParallel(config, DistributedDataParallelConfig(), module)
        inputs = torch.randn(4, 8, device="cuda")
        with typecheck():
            annotate_model(model)
            spmd.assert_type(inputs, {"TP": spmd.I, "DP": spmd.V, "CP": spmd.V})
            model(inputs)

        weight = module.weight
        assert (
            spmd.get_axis_local_type(weight, parallel_state.get_tensor_model_parallel_group())
            is spmd.I
        )
        assert spmd.get_axis_local_type(weight, parallel_state.get_data_parallel_group()) is spmd.R
        assert (
            spmd.get_axis_local_type(weight, parallel_state.get_context_parallel_group()) is spmd.R
        )


def test_mlperf_deepseek_v3_te_config_typechecks(mlperf_environment) -> None:
    """Check the MLPerf MoE/MTP/output chunk with real CUDA/TE kernels."""
    with _fake_distributed(
        rank=192,
        world_size=256,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=4,
        context_parallel_size=1,
        expert_model_parallel_size=32,
        expert_tensor_parallel_size=1,
    ):
        model_parallel_cuda_manual_seed(123)
        parallel_state.set_virtual_pipeline_model_parallel_rank(3)
        config = _make_mlperf_deepseek_v3_config()
        model = _wrap_in_ddp(config, _build_mlperf_deepseek_v3_model(config, vp_stage=3).cuda())
        hidden_states = torch.randn(
            MLPERF_DEEPSEEK_V3_SEQUENCE_LENGTH // config.tensor_model_parallel_size,
            1,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        input_ids = torch.zeros(
            (1, MLPERF_DEEPSEEK_V3_SEQUENCE_LENGTH), device="cuda", dtype=torch.long
        )
        labels = torch.zeros_like(input_ids)
        position_ids = torch.arange(MLPERF_DEEPSEEK_V3_SEQUENCE_LENGTH, device="cuda").unsqueeze(0)
        model.module.set_input_tensor(hidden_states)

        with (
            patch.object(model.module, "preprocess_for_paged_stash", return_value=None),
            patch.object(MTPLossLoggingHelper, "save_loss_to_tracker", return_value=None),
            typecheck(),
        ):
            annotate_model(model)
            spmd.assert_type(hidden_states, {"TP": spmd.S(0), "DP": spmd.V, "CP": spmd.V})
            spmd.assert_type(input_ids, {"TP": spmd.R, "DP": spmd.V, "CP": spmd.V})
            spmd.assert_type(position_ids, {"TP": spmd.I, "DP": spmd.R, "CP": spmd.R})
            spmd.assert_type(labels, {"TP": spmd.I, "DP": spmd.V, "CP": spmd.V})
            loss = model(input_ids, position_ids, attention_mask=None, labels=labels)
            loss_for_backward = spmd.reinterpret(
                loss.sum(),
                parallel_state.get_data_parallel_group(with_context_parallel=False),
                src=spmd.V,
                dst=spmd.P,
            )
            loss_for_backward.backward()

        assert loss.numel() == MLPERF_DEEPSEEK_V3_SEQUENCE_LENGTH


def test_mlperf_deepseek_v3_dense_embedding_chunk_typechecks(mlperf_environment) -> None:
    """Check the MLPerf embedding and dense-layer chunk with real CUDA/TE kernels."""
    with _fake_distributed(
        rank=0,
        world_size=256,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=4,
        context_parallel_size=1,
        expert_model_parallel_size=32,
        expert_tensor_parallel_size=1,
    ):
        model_parallel_cuda_manual_seed(123)
        parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        config = _make_mlperf_deepseek_v3_config()
        model = _wrap_in_ddp(config, _build_mlperf_deepseek_v3_model(config, vp_stage=0).cuda())
        input_ids = torch.zeros(
            (1, MLPERF_DEEPSEEK_V3_SEQUENCE_LENGTH), device="cuda", dtype=torch.long
        )
        position_ids = torch.arange(MLPERF_DEEPSEEK_V3_SEQUENCE_LENGTH, device="cuda").unsqueeze(0)

        with (
            patch.object(model.module, "preprocess_for_paged_stash", return_value=None),
            typecheck(),
        ):
            annotate_model(model)
            spmd.assert_type(input_ids, {"TP": spmd.R, "DP": spmd.V, "CP": spmd.V})
            spmd.assert_type(position_ids, {"TP": spmd.I, "DP": spmd.R, "CP": spmd.R})
            output = model(input_ids, position_ids, attention_mask=None)
            loss = spmd.reinterpret(
                output.sum(),
                parallel_state.get_tensor_model_parallel_group(),
                src=spmd.V,
                dst=spmd.P,
            )
            loss = spmd.reinterpret(
                loss,
                parallel_state.get_data_parallel_group(with_context_parallel=False),
                src=spmd.V,
                dst=spmd.P,
            )
            loss.backward()

        assert output.shape == (
            MLPERF_DEEPSEEK_V3_SEQUENCE_LENGTH // config.tensor_model_parallel_size,
            1,
            config.hidden_size,
        )
