"""The Megatron SPMD type-checking context."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Literal

import spmd_types as spmd
import torch
from spmd_types.checker import typecheck as _typecheck

from megatron.core.spmd.annotations import model_parallel_mesh


def assert_ddp_parameters_replicated(ddp: Any) -> None:
    """Record the gradient-reduction replication promise made by Megatron DDP."""
    mesh = spmd.current_mesh()
    dense_reduction_axis = spmd.MeshAxis.of(ddp.dp_cp_group)
    for param in ddp.params_with_grad:
        if getattr(param, "allreduce", True):
            for axis in mesh:
                if axis <= dense_reduction_axis:
                    spmd.assert_type(param, {axis: spmd.R})
        else:
            spmd.assert_type(param, {ddp.expt_dp_group: spmd.R})


@contextmanager
def typecheck(strict_mode: Literal["permissive", "strict"] = "strict") -> Iterator[None]:
    """Type-check one Megatron execution without changing normal execution."""
    # Importing the rules lazily avoids touching Megatron autograd classes
    # during normal execution; Python's module cache registers them only once.
    from megatron.core.spmd import spmd_rules  # noqa: F401

    mesh = model_parallel_mesh()
    with (
        _typecheck(strict_mode=strict_mode),
        spmd.set_current_mesh(mesh),
        # Keep pre-decorated torch.compile functions visible to the type checker.
        torch.compiler.set_stance("force_eager"),
    ):
        yield
