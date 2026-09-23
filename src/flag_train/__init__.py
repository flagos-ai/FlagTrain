"""FlagTrain package skeleton; no training or operator implementation yet."""

from flag_train import runtime, testing
from flag_train.deepspeed.lamb import lamb

device = runtime.device.name
vendor_name = runtime.device.vendor_name


class use_train:
    """Placeholder for FlagTrain' aten-patching context manager.

    The real ``use_train`` swaps in the registered operator implementations for
    the duration of the block, which needs the operator registry that FlagTrain
    does not carry yet. Nothing in this repository enters it -- the benchmarks
    pass an explicit ``gems_op`` -- so it only has to exist and fail loudly if a
    caller assumes the FlagTrain behaviour.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "flag_train has no registered aten operator overrides yet; pass the "
            "operator under test explicitly instead of relying on use_train()."
        )


__all__ = [
    "device",
    "lamb",
    "use_train",
    "vendor_name",
]
