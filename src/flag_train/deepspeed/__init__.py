"""DeepSpeed-related implementations."""

from .blocked_flash import blocked_flash
from .cuda_wf6af16_linear import cuda_wf6af16_linear
from .evoformer_attn import evoformer_attn
from .lamb import lamb
<<<<<<< HEAD
=======
from .lion import multi_tensor_lion
from .loco_quantized_reduction import loco_quantized_reduction
>>>>>>> 8c3cddd ([KMCompiler][NVIDIA][Ascend][Hygon][Thead][MetaX][iluvatar]Add loco_quantized_reduction operator with triton kernel.)
from .mixed_gemm import mixed_gemm
from .mixed_moe_gemm import mixed_moe_gemm
from .multi_tensor_adam import multi_tensor_adam
from .multi_tensor_lion import multi_tensor_lion
from .packbits import packbits
from .unpackbits import unpackbits

__all__ = [
    "blocked_flash",
    "cuda_wf6af16_linear",
    "evoformer_attn",
    "lamb",
    "mixed_gemm",
    "mixed_moe_gemm",
    "multi_tensor_adam",
    "multi_tensor_lion",
    "packbits",
    "unpackbits",
<<<<<<< HEAD
=======
    "wf6af16_linear",
    "loco_quantized_reduction",
>>>>>>> 8c3cddd ([KMCompiler][NVIDIA][Ascend][Hygon][Thead][MetaX][iluvatar]Add loco_quantized_reduction operator with triton kernel.)
]

# The vendor-operator registration for these names runs in flag_train/__init__.py
# once this package and its siblings are fully imported -- see the comment there.
