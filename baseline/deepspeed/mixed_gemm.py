# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVIDIA baseline for mixed_gemm.

The baseline is DeepSpeed's CUTLASS mixed-precision GEMM with weight-only
integer quantization:

    deepspeed.ops.op_builder.InferenceCutlassBuilder().load().mixed_gemm
    (output, hidden_states, weight, scales, bias, num_bits, activation_raw)

Source: ``deepspeed/inference/v2/kernels/cutlass_ops/mixed_gemm/mixed_gemm.cu``.
The operator writes the preallocated ``output`` and returns nothing, so it is
``IS_INPLACE``. ``activation_raw`` is 6 (identity) here: the FlagTrain benchmark
keeps the activation out of the GEMM measurement, and both dtypes below are the
ones the kernel's ``ACT_DTYPE_SWITCH`` instantiates.

When DeepSpeed is unavailable -- or its prebuilt ``deepspeedft`` library does not
carry this GPU's compute capability, which is the case on sm_90 -- ``native()``
raises; the collector does not collect this operator's baseline and does not fall
back to a torch composition. ``_call_probe`` makes the second case visible instead
of leaving it to the timed run, because that library imports cleanly and fails
only when called.

The input construction mirrors the shapes, dtypes, quantization parameters and
weight footprint in ``FlagTrain/benchmark/deepspeed/test_mixed_gemm.py``: weights
are a uint8 buffer of ``[K, N]`` bytes at 8 bit and the same ``[K, N]`` logical
shape packed two nibbles per byte along N at 4 bit.

Note: the packed buffer is random rather than run through DeepSpeed's
``preprocess_weights_for_mixed_gemm``. That entry point is not importable from the
shipped wheel, and the collector measures latency and NCU metrics only and never
compares values. The buffer therefore has the real footprint and drives the same
weight bandwidth, but it is not a numerical oracle -- numerical correctness is
covered by the torch reference in the FlagTrain benchmark. The shape key carries
the bit width so the 4-bit and 8-bit variants of one (M, K, N) do not overwrite
each other.
"""

import torch

OP_NAME = "mixed_gemm"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True  # the operator writes the preallocated `output`

_ACTIVATION = 6  # identity

# benchmark/deepspeed/test_mixed_gemm.py::_SHAPES, (m, k, n, num_bits).
_SHAPES = [
    # Thin: few tokens, wide weight -- the decoding shape, where the weight
    # dominates traffic.
    (1, 4096, 4096, 4),
    (64, 4096, 4096, 4),
    (1, 4096, 4096, 8),
    (64, 4096, 4096, 8),
    # Square, walking past L2.
    (1024, 1024, 1024, 4),
    (2048, 2048, 2048, 4),
    (4096, 4096, 4096, 4),
    (1024, 1024, 1024, 8),
    (2048, 2048, 2048, 8),
    (4096, 4096, 4096, 8),
    # Wide and deep: many tokens, a long reduction, 16384 output columns.
    (8192, 4096, 4096, 4),
    (2048, 4096, 16384, 4),
    (8192, 4096, 4096, 8),
    (2048, 4096, 16384, 8),
]


def native():
    """DeepSpeed's mixed_gemm binding; raise when it cannot be loaded."""
    try:
        from deepspeed.ops.op_builder import InferenceCutlassBuilder

        kernel = InferenceCutlassBuilder().load().mixed_gemm
    except Exception as exc:
        raise RuntimeError(
            "mixed_gemm baseline requires DeepSpeed's CUTLASS mixed GEMM "
            "(InferenceCutlassBuilder). Install deepspeed and ensure its CUDA "
            f"extensions build on this machine: {exc!r}"
        ) from exc
    return _call_probe(kernel)


def _call_probe(kernel):
    """Run the kernel once on 128x128 so a missing sm target fails here.

    The shipped ``deepspeedft`` library is prebuilt, so a compute capability it
    does not cover still imports and fails only at the first call.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = k = n = 128
    hidden = torch.randn(m, k, dtype=DTYPES[0], device=device)
    weight = torch.randint(0, 256, (k, n // 2), dtype=torch.uint8, device=device)
    scales = torch.rand(n, dtype=DTYPES[0], device=device) + 0.1
    output = torch.empty(m, n, dtype=DTYPES[0], device=device)
    try:
        kernel(output, hidden, weight, scales, None, 4, _ACTIVATION)
    except Exception as exc:
        raise RuntimeError(
            "DeepSpeed's mixed_gemm loaded but does not run here. Its kernel ships "
            "prebuilt in the `dskernels` package, and `InferenceCutlassBuilder` links "
            "`-ldeepspeedft` rather than compiling it, so that package has to carry "
            f"this GPU's compute capability: {exc!r}"
        ) from exc
    return kernel


def grid():
    return [{"shape": (m, k, n), "num_bits": num_bits} for m, k, n, num_bits in _SHAPES]


def build_inputs(binding, dtype, device):
    m, k, n = binding["shape"]
    num_bits = binding["num_bits"]
    torch.manual_seed(0)
    hidden = torch.randn(m, k, dtype=dtype, device=device)
    # 4 bit packs two weights per byte along N; 8 bit is one byte per weight.
    packed_cols = n // 2 if num_bits == 4 else n
    weight = torch.randint(0, 256, (k, packed_cols), dtype=torch.uint8, device=device)
    scales = (torch.rand(n, device=device) + 0.1).to(dtype)
    output = torch.empty(m, n, dtype=dtype, device=device)
    args = (output, hidden, weight, scales, None, num_bits, _ACTIVATION)
    return args, {}


def key_shape(binding):
    m, k, n = binding["shape"]
    return f"{m}x{k}x{n}_bits{binding['num_bits']}"


def config(binding, dtype):
    m, k, n = binding["shape"]
    bits = binding["num_bits"]
    return {
        "inputs": {
            "output": {"shape": [m, n], "dtype": str(dtype)},
            "hidden_states": {"shape": [m, k], "dtype": str(dtype)},
            "weight": {
                "shape": [k, n // 2 if bits == 4 else n],
                "dtype": "torch.uint8",
            },
            "scales": {"shape": [n], "dtype": str(dtype)},
            "num_bits": {"scalar": bits},
            "activation_raw": {"scalar": _ACTIVATION},
        },
        "dims": {
            "M": m,
            "K": k,
            "N": n,
        },
    }
