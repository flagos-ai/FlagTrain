"""DeepSpeed-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .blocked_flash import blocked_flash
from .evoformer_attn import evoformer_attn
from .lamb import lamb

__all__ = [
    "blocked_flash",
    "evoformer_attn",
    "lamb",
]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
