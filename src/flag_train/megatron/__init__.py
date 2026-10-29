"""Megatron-related implementations."""

from .fast_gelu import fast_gelu
from .fused_gated_norm import fused_gated_norm
from .fused_indices_to_multihot import fused_indices_to_multihot
from .fused_pad_routing_map import fused_pad_routing_map
from .fused_proj_rms_backward import fused_proj_rms_backward
from .fused_proj_rms_forward import fused_proj_rms_forward
from .quick_gelu import quick_gelu
from .scaled_upper_triang_masked_softmax_forward import (
    scaled_upper_triang_masked_softmax_forward,
)

__all__ = [
    "fast_gelu",
    "fused_gated_norm",
    "fused_indices_to_multihot",
    "fused_pad_routing_map",
    "fused_proj_rms_backward",
    "fused_proj_rms_forward",
    "quick_gelu",
    "scaled_upper_triang_masked_softmax_forward",
]

# The vendor-operator registration for these names runs in flag_train/__init__.py
# once this package and its siblings are fully imported -- see the comment there.
