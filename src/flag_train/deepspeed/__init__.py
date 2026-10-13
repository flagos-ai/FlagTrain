"""DeepSpeed-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .blocked_flash import blocked_flash
from .lamb import lamb
from .loco_quantized_reduction import loco_quantized_reduction

__all__ = [
    "blocked_flash",
    "lamb",
    "loco_quantized_reduction",
]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
