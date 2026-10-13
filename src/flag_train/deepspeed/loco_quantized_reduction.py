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

import os
from enum import IntEnum
from typing import List

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

__all__ = ["QuantizationType", "loco_quantized_reduction", "quantized_reduction"]


class QuantizationType(IntEnum):
    Symmetric = 0
    Asymmetric = 1


_BLOCK_MAX = int(os.environ.get("FLAG_TRAIN_LQR_BLOCK_MAX", "8192"))
_BLOCK_MIN = 256
_WARPS_DIV = int(os.environ.get("FLAG_TRAIN_LQR_WARPS_DIV", "128"))
_MAX_WARPS = int(os.environ.get("FLAG_TRAIN_LQR_MAX_WARPS", "8"))


@triton.jit
def _round_half_even_f32(x):
    return (x + 12582912.0) - 12582912.0


@triton.jit
def _dr_slice(
    in_ptr,
    step_ptr,
    j,
    in_byte,
    gob,
    bmask,
    EPIT,
    GPIT,
    NUM_BITS: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    SCALAR_STEP: tl.constexpr,
    acc_hi,
    acc_lo,
):
    b8 = tl.load(in_ptr + j * EPIT + in_byte, mask=bmask, other=0)

    if NUM_BITS == 8:
        vh = b8.to(tl.float32)
        vl = vh
    else:
        vh = (b8 >> 4).to(tl.float32)
        vl = ((b8 << 4) >> 4).to(tl.float32)

    if SYMMETRIC:
        if SCALAR_STEP:
            s = tl.load(step_ptr + j * GPIT + gob)
        else:
            s = tl.load(step_ptr + j * GPIT + gob, mask=bmask, other=0.0)
        return acc_hi + vh * s, acc_lo + vl * s

    gi = 2 * gob
    if SCALAR_STEP:
        s = tl.load(step_ptr + j * 2 * GPIT + gi)
        o = tl.load(step_ptr + j * 2 * GPIT + gi + 1)
    else:
        s = tl.load(step_ptr + j * 2 * GPIT + gi, mask=bmask, other=0.0)
        o = tl.load(step_ptr + j * 2 * GPIT + gi + 1, mask=bmask, other=0.0)
    return acc_hi + vh * s + o, acc_lo + vl * s + o


@triton.jit
def _accumulate(
    in_ptr,
    in_step_ptr,
    in_byte,
    gob,
    cmask,
    EPIT,
    GPIT,
    num_tensors,
    NUM_TENSORS: tl.constexpr,
    NUM_BITS: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    CHUNK: tl.constexpr,
):
    acc_hi = tl.zeros((CHUNK,), dtype=tl.float32)
    acc_lo = tl.zeros((CHUNK,), dtype=tl.float32)

    if NUM_TENSORS > 0:
        for j in tl.static_range(NUM_TENSORS):
            acc_hi, acc_lo = _dr_slice(
                in_ptr,
                in_step_ptr,
                j,
                in_byte,
                gob,
                cmask,
                EPIT,
                GPIT,
                NUM_BITS,
                SYMMETRIC,
                True,
                acc_hi,
                acc_lo,
            )
    else:
        for j in range(num_tensors):
            acc_hi, acc_lo = _dr_slice(
                in_ptr,
                in_step_ptr,
                j,
                in_byte,
                gob,
                cmask,
                EPIT,
                GPIT,
                NUM_BITS,
                SYMMETRIC,
                True,
                acc_hi,
                acc_lo,
            )
    return acc_hi, acc_lo


@triton.jit
def _quant_params(stat, mn, NUM_BITS: tl.constexpr, SYMMETRIC: tl.constexpr):
    LEVELS = 1 << NUM_BITS

    if SYMMETRIC:
        qscale = tl.where(stat == 0.0, 1.0, LEVELS / (2.0 * stat))
        step = tl.where(stat == 0.0, 1.0, 1.0 / qscale)
        off = 0.0
    else:
        rng = stat - mn
        qscale = tl.where(rng == 0.0, 1.0, LEVELS / rng)
        step = tl.where(rng == 0.0, 1.0, 1.0 / qscale)
        off = (stat + mn) * 0.5
    return qscale, step, off


@libentry()
@triton.jit
def _dequant_reduce_kernel(
    out_ptr,
    out_step_ptr,
    in_ptr,
    in_step_ptr,
    BYTES_PER_GROUP,
    EPIT,
    EPIG,
    GPIT,
    num_tensors,
    NUM_TENSORS: tl.constexpr,
    NUM_BITS: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    BLOCK: tl.constexpr,
    EPIG_SHIFT: tl.constexpr,
    SCALAR_STEP: tl.constexpr,
):
    g = tle.program_id(0)
    kb = tl.arange(0, BLOCK)
    bmask = kb < BYTES_PER_GROUP
    in_byte = g * BYTES_PER_GROUP + kb

    if SCALAR_STEP:
        gob = (g * BYTES_PER_GROUP) >> EPIG_SHIFT
    else:
        gob = in_byte >> EPIG_SHIFT if EPIG_SHIFT >= 0 else in_byte // EPIG

    acc_hi, acc_lo = _accumulate(
        in_ptr,
        in_step_ptr,
        in_byte,
        gob,
        bmask,
        EPIT,
        GPIT,
        num_tensors,
        NUM_TENSORS,
        NUM_BITS,
        SYMMETRIC,
        BLOCK,
    )

    if SYMMETRIC:
        stat = tl.max(tl.where(bmask, tl.maximum(tl.abs(acc_hi), tl.abs(acc_lo)), 0.0))
        mn = 0.0
    else:
        stat = tl.max(tl.where(bmask, tl.maximum(acc_hi, acc_lo), float("-inf")))
        mn = tl.min(tl.where(bmask, tl.minimum(acc_hi, acc_lo), float("inf")))

    qscale, step, off = _quant_params(stat, mn, NUM_BITS, SYMMETRIC)

    one = tl.arange(0, 1)
    if SYMMETRIC:
        tl.store(out_step_ptr + g + one, tl.zeros((1,), tl.float32) + step)
    else:
        tl.store(out_step_ptr + 2 * g + one, tl.zeros((1,), tl.float32) + step)
        tl.store(out_step_ptr + 2 * g + 1 + one, tl.zeros((1,), tl.float32) + off)

    QMIN = -(1 << (NUM_BITS - 1))
    QMAX = (1 << (NUM_BITS - 1)) - 1

    qh = _round_half_even_f32((acc_hi - off) * qscale)
    qh = tl.minimum(tl.maximum(qh, QMIN), QMAX).to(tl.int32)
    if NUM_BITS == 4:
        ql = _round_half_even_f32((acc_lo - off) * qscale)
        ql = tl.minimum(tl.maximum(ql, QMIN), QMAX).to(tl.int32)
        out_v = ((qh & 0xF) << 4) | (ql & 0xF)
    else:
        out_v = qh & 0xFF

    tl.store(out_ptr + in_byte, out_v.to(tl.int8), mask=bmask)


@libentry()
@triton.jit
def _dequant_reduce_multipass_kernel(
    out_ptr,
    out_step_ptr,
    in_ptr,
    in_step_ptr,
    BYTES_PER_GROUP,
    EPIT,
    GPIT,
    num_tensors,
    num_chunks,
    NUM_TENSORS: tl.constexpr,
    NUM_BITS: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    CHUNK: tl.constexpr,
):
    g = tle.program_id(0)
    kc = tl.arange(0, CHUNK)
    one = tl.arange(0, 1)
    gob = g

    stat = 0.0 if SYMMETRIC else float("-inf")
    mn = 0.0 if SYMMETRIC else float("inf")

    for c in range(num_chunks):
        off = c * CHUNK + kc
        cmask = off < BYTES_PER_GROUP
        in_byte = g * BYTES_PER_GROUP + off
        acc_hi, acc_lo = _accumulate(
            in_ptr,
            in_step_ptr,
            in_byte,
            gob,
            cmask,
            EPIT,
            GPIT,
            num_tensors,
            NUM_TENSORS,
            NUM_BITS,
            SYMMETRIC,
            CHUNK,
        )
        if SYMMETRIC:
            stat = tl.maximum(
                stat,
                tl.max(
                    tl.where(cmask, tl.maximum(tl.abs(acc_hi), tl.abs(acc_lo)), 0.0)
                ),
            )
        else:
            stat = tl.maximum(
                stat,
                tl.max(tl.where(cmask, tl.maximum(acc_hi, acc_lo), float("-inf"))),
            )
            mn = tl.minimum(
                mn,
                tl.min(tl.where(cmask, tl.minimum(acc_hi, acc_lo), float("inf"))),
            )

    qscale, step, off_q = _quant_params(stat, mn, NUM_BITS, SYMMETRIC)

    if SYMMETRIC:
        tl.store(out_step_ptr + g + one, tl.zeros((1,), tl.float32) + step)
    else:
        tl.store(out_step_ptr + 2 * g + one, tl.zeros((1,), tl.float32) + step)
        tl.store(out_step_ptr + 2 * g + 1 + one, tl.zeros((1,), tl.float32) + off_q)

    QMIN = -(1 << (NUM_BITS - 1))
    QMAX = (1 << (NUM_BITS - 1)) - 1

    for c in range(num_chunks):
        off = c * CHUNK + kc
        cmask = off < BYTES_PER_GROUP
        in_byte = g * BYTES_PER_GROUP + off
        acc_hi, acc_lo = _accumulate(
            in_ptr,
            in_step_ptr,
            in_byte,
            gob,
            cmask,
            EPIT,
            GPIT,
            num_tensors,
            NUM_TENSORS,
            NUM_BITS,
            SYMMETRIC,
            CHUNK,
        )

        qh = _round_half_even_f32((acc_hi - off_q) * qscale)
        qh = tl.minimum(tl.maximum(qh, QMIN), QMAX).to(tl.int32)
        if NUM_BITS == 4:
            ql = _round_half_even_f32((acc_lo - off_q) * qscale)
            ql = tl.minimum(tl.maximum(ql, QMIN), QMAX).to(tl.int32)
            out_v = ((qh & 0xF) << 4) | (ql & 0xF)
        else:
            out_v = qh & 0xFF

        tl.store(out_ptr + in_byte, out_v.to(tl.int8), mask=cmask)


def _pick_block(bytes_per_group: int) -> int:
    return min(_BLOCK_MAX, max(_BLOCK_MIN, triton.next_power_of_2(bytes_per_group)))


def _pow2_shift(v: int) -> int:
    return v.bit_length() - 1 if v > 0 and (v & (v - 1)) == 0 else -1


def _pick_warps(block: int) -> int:
    if block <= 8192:
        return max(1, min(_MAX_WARPS, block // _WARPS_DIV))
    return min(32, block // 512)


def _normalize_quant_type(quant_type) -> QuantizationType:
    if isinstance(quant_type, QuantizationType):
        return quant_type

    if isinstance(quant_type, str):
        s = quant_type.strip().lower()
        if s.startswith("sym"):
            return QuantizationType.Symmetric
        if s.startswith("asym"):
            return QuantizationType.Asymmetric
        raise ValueError(f"unknown quant_type: {quant_type!r}")

    name = getattr(quant_type, "name", None)
    if isinstance(name, str):
        return _normalize_quant_type(name)

    try:
        return QuantizationType(int(quant_type))
    except (TypeError, ValueError) as e:
        raise ValueError(f"unknown quant_type: {quant_type!r}") from e


def loco_quantized_reduction(
    input_vals: torch.Tensor,
    input_scales: torch.Tensor,
    in_groups: int,
    out_groups: int,
    num_bits: int,
    quant_type=QuantizationType.Symmetric,
    devices_per_node: int = 1,
) -> List[torch.Tensor]:
    num_tensors = int(devices_per_node)
    if num_tensors < 1:
        raise ValueError(f"devices_per_node must be >= 1, got {num_tensors}")
    if num_bits not in (4, 8):
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")

    qt = _normalize_quant_type(quant_type)
    symmetric = qt == QuantizationType.Symmetric
    scales_elems = 1 if symmetric else 2

    if input_vals.dtype != torch.int8:
        raise TypeError(f"input_vals must be int8, got {input_vals.dtype}")
    if input_scales.dtype != torch.float32:
        raise TypeError(f"input_scales must be float32, got {input_scales.dtype}")
    if not input_vals.is_contiguous():
        raise ValueError("input_vals must be contiguous")
    if not input_scales.is_contiguous():
        raise ValueError("input_scales must be contiguous")

    if input_vals.shape[-1] % num_tensors != 0:
        raise ValueError(
            f"last dim {input_vals.shape[-1]} is not divisible by "
            f"devices_per_node={num_tensors}"
        )

    if out_groups < 1:
        raise ValueError(f"out_groups must be >= 1, got {out_groups}")
    if in_groups != out_groups * num_tensors:
        raise ValueError(
            f"in_groups({in_groups}) must equal "
            f"out_groups({out_groups}) * devices_per_node({num_tensors})"
        )

    total_bytes = input_vals.numel()
    groups_per_in_tensor = in_groups // num_tensors
    elems_per_in_tensor = total_bytes // num_tensors

    if elems_per_in_tensor % groups_per_in_tensor != 0:
        raise ValueError(
            f"elems_per_in_tensor({elems_per_in_tensor}) is not divisible by "
            f"groups_per_in_tensor({groups_per_in_tensor})"
        )

    elems_per_in_group = elems_per_in_tensor // groups_per_in_tensor
    elems_per_out_group = elems_per_in_tensor // out_groups

    if input_scales.numel() != in_groups * scales_elems:
        raise ValueError(
            f"input_scales.numel()={input_scales.numel()} must equal "
            f"in_groups({in_groups}) * {scales_elems} = {in_groups * scales_elems}"
        )

    out_shape = list(input_vals.shape)
    out_shape[-1] = out_shape[-1] // num_tensors
    out = torch.empty(out_shape, dtype=torch.int8, device=input_vals.device)
    out_st = torch.empty(
        (out_groups, scales_elems), dtype=torch.float32, device=input_vals.device
    )

    expected_out_bytes = out_groups * elems_per_out_group
    if out.numel() != expected_out_bytes:
        raise ValueError(
            f"output numel({out.numel()}) does not match the number of bytes "
            f"the kernel writes ({expected_out_bytes})"
        )

    block = _pick_block(elems_per_out_group)
    static_tensors = num_tensors if num_tensors in (8, 16) else 0

    with torch_device_fn.device(input_vals.device):
        if block >= elems_per_out_group:
            epig_shift = _pow2_shift(elems_per_in_group)
            scalar_step = epig_shift >= 0 and elems_per_out_group <= elems_per_in_group
            _dequant_reduce_kernel[(out_groups,)](
                out,
                out_st,
                input_vals,
                input_scales,
                elems_per_out_group,
                elems_per_in_tensor,
                elems_per_in_group,
                groups_per_in_tensor,
                num_tensors,
                NUM_TENSORS=static_tensors,
                NUM_BITS=num_bits,
                SYMMETRIC=symmetric,
                BLOCK=block,
                EPIG_SHIFT=epig_shift,
                SCALAR_STEP=scalar_step,
                num_warps=_pick_warps(block),
            )
        else:
            num_chunks = triton.cdiv(elems_per_out_group, _BLOCK_MAX)
            _dequant_reduce_multipass_kernel[(out_groups,)](
                out,
                out_st,
                input_vals,
                input_scales,
                elems_per_out_group,
                elems_per_in_tensor,
                groups_per_in_tensor,
                num_tensors,
                num_chunks,
                NUM_TENSORS=static_tensors,
                NUM_BITS=num_bits,
                SYMMETRIC=symmetric,
                CHUNK=_BLOCK_MAX,
                num_warps=_pick_warps(_BLOCK_MAX),
            )

    return [out, out_st]


quantized_reduction = loco_quantized_reduction
