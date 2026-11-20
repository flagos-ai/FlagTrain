"""DeepSpeed-related implementations."""

from .blocked_flash import blocked_flash
from .cuda_wf6af16_linear import cuda_wf6af16_linear
from .evoformer_attn import evoformer_attn
from .fp8_gemm_triton import fp8_gemm_triton
from .lamb import lamb
from .loco_quantized_reduction import loco_quantized_reduction
from .mixed_gemm import mixed_gemm
from .mixed_moe_gemm import mixed_moe_gemm
from .moe_gemm import moe_gemm
from .multi_tensor_adam import multi_tensor_adam
from .multi_tensor_lion import multi_tensor_lion
from .packbits import packbits
from .unpackbits import unpackbits

__all__ = [
    "blocked_flash",
    "cuda_wf6af16_linear",
    "evoformer_attn",
    "fp8_gemm_triton",
    "lamb",
    "loco_quantized_reduction",
    "mixed_gemm",
    "mixed_moe_gemm",
    "moe_gemm",
    "multi_tensor_adam",
    "multi_tensor_lion",
    "packbits",
    "unpackbits",
]

# The vendor-operator registration for these names runs in flag_train/__init__.py
# once this package and its siblings are fully imported -- see the comment there.
