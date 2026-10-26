"""Megatron-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .fused_indices_to_multihot import fused_indices_to_multihot
from .bias_swiglu_impl import bias_swiglu_impl, weighted_bias_swiglu_impl
from .quick_gelu import quick_gelu

__all__ = [
    "bias_swiglu_impl",
    "fused_indices_to_multihot",
    "quick_gelu",
    "weighted_bias_swiglu_impl",
]

SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
