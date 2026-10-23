"""Megatron-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .bias_swiglu_impl import bias_swiglu_impl, weighted_bias_swiglu_impl
from .quick_gelu import quick_gelu

__all__ = [
    "bias_swiglu_impl",
    "quick_gelu",
    "weighted_bias_swiglu_impl",
]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
