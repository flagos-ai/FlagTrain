"""Megatron-related implementations."""

from flag_train.megatron.fused_gated_norm import fused_gated_norm
from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

__all__ = ["fused_gated_norm"]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
