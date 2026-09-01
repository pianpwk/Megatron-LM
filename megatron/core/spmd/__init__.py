"""Optional SPMD type-checking integration for Megatron Core."""

from .annotations import annotate_model, annotate_tensor
from .checker import typecheck

__all__ = ["annotate_model", "annotate_tensor", "typecheck"]
