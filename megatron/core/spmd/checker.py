"""The Megatron SPMD type-checking context."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Literal

import spmd_types as spmd
import torch
from spmd_types.checker import typecheck as _typecheck

from megatron.core.spmd.annotations import model_parallel_mesh


@contextmanager
def typecheck(strict_mode: Literal["permissive", "strict"] = "strict") -> Iterator[None]:
    """Type-check one Megatron execution without changing normal execution."""
    mesh = model_parallel_mesh()
    with (
        _typecheck(strict_mode=strict_mode),
        spmd.set_current_mesh(mesh),
        # Keep pre-decorated torch.compile functions visible to the type checker.
        torch.compiler.set_stance("force_eager"),
    ):
        yield
