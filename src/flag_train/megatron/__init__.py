"""Megatron-related implementations."""

from flag_train.megatron.fused_pad_routing_map import fused_pad_routing_map
from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

__all__ = ["fused_pad_routing_map"]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
