#16bit moe_gemm_triton

import logging

import torch
import triton
import triton.language as tl

from flag_train import runtime
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

@libentry()
@triton.jit
def moe_gemm_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    row_ends_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_IM: tl.constexpr,
    STRIDE_IK: tl.constexpr,
    STRIDE_OM: tl.constexpr,
    STRIDE_ON: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tle.program_id(0)
    expert = tle.program_id(1)
    tiles_n = tl.cdiv(N, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile % tiles_n
    first_row = tl.load(row_ends_ptr + expert - 1, expert > 0, other=0)
    last_row = tl.load(row_ends_ptr + expert)
    if tile_m * BLOCK_M < last_row - first_row:
        rows = first_row + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        row_mask = (rows < last_row) & (rows < M)
        col_mask = cols < N
        expert_offset = expert.to(tl.int64)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for block_k in range(tl.cdiv(K, BLOCK_K)):
            ks = block_k * BLOCK_K + offsets_k
            inputs = tl.load(
                input_ptr + rows[:, None] * STRIDE_IM + ks[None, :] * STRIDE_IK,
                row_mask[:, None] & (ks[None, :] < K),
                other=0,
            )
            weight_offsets = (
                expert_offset * K * N
                + ks[:, None].to(tl.int64) * N
                + cols[None, :]
            )
            weights = tl.load(
                weight_ptr + weight_offsets,
                (ks[:, None] < K) & col_mask[None, :],
                other=0,
            ).to(inputs.dtype)
            accumulator += tl.dot(inputs, weights)

        if HAS_BIAS:
            biases = tl.load(bias_ptr + expert_offset * N + cols, col_mask, other=0).to(
                tl.float32
            )
            accumulator += biases[None, :]
        if ACTIVATION == 0:
            accumulator = (
                0.5
                * accumulator
                * (1.0 + tl_extra_shim.erf(accumulator * 0.7071067811865476))
            )
        elif ACTIVATION == 1:
            accumulator = tl.maximum(accumulator, 0.0)
        elif ACTIVATION == 2:
            accumulator = accumulator * tl.sigmoid(accumulator)

        output_offsets = rows[:, None] * STRIDE_OM + cols[None, :] * STRIDE_ON
        tl.store(
            output_ptr + output_offsets,
            accumulator,
            row_mask[:, None] & col_mask[None, :],
        )

def _validate_inputs(
    output,
    hidden_states,
    weights,
    biases,
    total_rows_before_expert,
    activation,
):
    supported_vendors = ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend")
    if runtime.device.vendor_name not in supported_vendors:
        raise RuntimeError("moe_gemm does not support this accelerator backend")
    if activation not in (0, 1, 2, 6):
        raise ValueError("activation must be GELU=0, RELU=1, SILU=2, or IDENTITY=6")
    if hidden_states.ndim != 2 or output.ndim != 2 or weights.ndim != 3:
        raise ValueError("input/output must be 2D and weights must be 3D")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("hidden_states must have dtype float16 or bfloat16")
    if output.dtype != hidden_states.dtype:
        raise TypeError("output and hidden_states must have the same dtype")
    if weights.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("weights must have dtype float16 or bfloat16")

    m, k = hidden_states.shape
    n = output.shape[1]
    num_experts = weights.shape[0]
    if num_experts == 0:
        raise ValueError("at least one expert is required")
    if output.shape[0] != m or weights.shape[1:] != (k, n):
        raise ValueError("input, output, and weight dimensions do not match")
    if (
        total_rows_before_expert.shape != (num_experts,)
        or total_rows_before_expert.dtype != torch.int64
    ):
        raise ValueError("row ends must have shape [experts] and dtype int64")
    if biases is not None:
        if biases.shape != (num_experts, n) or biases.dtype != hidden_states.dtype:
            raise ValueError("biases must have shape [experts, N] and the input dtype")

    tensors = [output, hidden_states, weights, total_rows_before_expert]
    if biases is not None:
        tensors.append(biases)
    if hidden_states.device.type != runtime.device.name or any(
        t.device != hidden_states.device for t in tensors
    ):
        raise ValueError("all tensors must be on the same accelerator device")
    contiguous_tensors = [weights, total_rows_before_expert]
    if biases is not None:
        contiguous_tensors.append(biases)
    if any(not t.is_contiguous() for t in contiguous_tensors):
        raise ValueError("weights, biases, and row ends must be contiguous")
    return m, n, k, num_experts

def moe_gemm(
    output,
    hidden_states,
    weights,
    biases,
    total_rows_before_expert,
    activation=6,
):
    logger.debug("TRAIN MOE_GEMM")
    m, n, k, num_experts = _validate_inputs(
        output,
        hidden_states,
        weights,
        biases,
        total_rows_before_expert,
        activation,
    )
    if m == 0 or n == 0:
        return output

    block_m = 32 if m <= 64 * num_experts else 64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, 64), num_experts)
    with torch_device_fn.device(hidden_states.device):
        moe_gemm_kernel[grid](
            output,
            hidden_states,
            weights,
            biases,
            total_rows_before_expert,
            m,
            n,
            k,
            hidden_states.stride(0),
            hidden_states.stride(1),
            output.stride(0),
            output.stride(1),
            ACTIVATION=int(activation),
            HAS_BIAS=biases is not None,
            BLOCK_M=block_m,
            BLOCK_N=64,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )
    return output
