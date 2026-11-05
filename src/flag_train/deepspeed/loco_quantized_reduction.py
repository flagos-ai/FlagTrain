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
from enum import IntEnum
from typing import List

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry


class QuantizationType(IntEnum):
    Symmetric = 0
    Asymmetric = 1


_VENDOR = str(getattr(runtime_device, "vendor_name", "") or "").lower()
_DEVICE = str(getattr(runtime_device, "name", "") or "").lower()
_IS_ASCEND = _VENDOR == "ascend" or _DEVICE == "npu"
_IS_CUDA_REF = _VENDOR == "nvidia"


def _device_warp_cap() -> int:
    try:
        props = torch_device_fn.get_device_properties(torch_device_fn.current_device())
        limit = int(getattr(props, "max_threads_per_block", 0))
        warp = int(getattr(props, "warp_size", 0))
        if limit > 0 and warp > 0:
            return max(1, min(32, limit // warp))
    except Exception:
        pass
    return 32 if _IS_CUDA_REF else 8


_BLOCK_MAX = 0
_BLOCK_MAX_I4 = 8192
_BLOCK_MAX_I8 = 8192
_BLOCK_MIN = 64

_UB_BUDGET = 128 * 1024
_STATIC_BLOCK = 4096
_WARPS_DIV = 256
_MAX_WARPS = _device_warp_cap()
_NUM_STAGES = 1


def _block_cap(num_bits: int) -> int:
    return _BLOCK_MAX or (_BLOCK_MAX_I8 if num_bits == 8 else _BLOCK_MAX_I4)


def _will_mask(bytes_per_group: int, block: int) -> bool:
    if block >= bytes_per_group:
        return block != bytes_per_group
    return (bytes_per_group % block) != 0


def _ub_units(num_bits: int, static_d: int, masked: bool, lean: bool) -> int:
    if num_bits == 4:
        base = 14 if lean else 22
    else:
        base = 12
    return base + (8 if masked else 0) + max(0, static_d - 1)


def _fits_ub(block: int, units: int) -> bool:
    return (not _IS_ASCEND) or block * units <= _UB_BUDGET


def _pick_tile(bytes_per_group: int, num_bits: int, symmetric: bool):
    cap = _block_cap(num_bits)
    cand = min(cap, max(_BLOCK_MIN, triton.next_power_of_2(bytes_per_group)))
    allow_lean = symmetric and num_bits == 4

    while cand > _BLOCK_MIN:
        masked = _will_mask(bytes_per_group, cand)
        if _fits_ub(cand, _ub_units(num_bits, 0, masked, False)):
            return cand, False
        if (
            cand >= bytes_per_group
            and allow_lean
            and _fits_ub(cand, _ub_units(num_bits, 0, masked, True))
        ):
            return cand, True
        cand >>= 1
    return _BLOCK_MIN, False


def _pick_static(
    num_tensors: int, bytes_per_group: int, block: int, num_bits: int, lean: bool
) -> int:
    if num_tensors not in (8, 16):
        return 0
    if _IS_ASCEND:
        if block > _STATIC_BLOCK:
            return 0
        masked = _will_mask(bytes_per_group, block)
        if not _fits_ub(block, _ub_units(num_bits, num_tensors, masked, lean)):
            return 0
    return num_tensors


def _pick_warps(block: int) -> int:
    warps = max(2, block // _WARPS_DIV)
    warps = min(warps, _MAX_WARPS, 32)
    return 1 << (warps.bit_length() - 1)


@triton.jit
def _dr_slice(
    in_ptr,
    step_ptr,
    j,
    in_byte,
    g,
    bmask,
    epit,
    gpit,
    NUM_BITS: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    MASKED: tl.constexpr,
    ALIGNED: tl.constexpr,
    acc_hi,
    acc_lo,
):
    src = j * epit + in_byte
    if MASKED:
        b8 = tl.load(in_ptr + src, mask=bmask, other=0)
    elif ALIGNED:
        b8 = tl.load(in_ptr + tl.multiple_of(src, 16))
    else:
        b8 = tl.load(in_ptr + src)

    if NUM_BITS == 8:
        vh = b8.to(tl.float32)
    else:
        vh = (b8.to(tl.int16) >> 4).to(tl.float32)

    if SYMMETRIC:
        s = tl.load(step_ptr + j * gpit + g)
    else:
        s = tl.load(step_ptr + j * 2 * gpit + 2 * g)
    hi = acc_hi + vh * s

    if NUM_BITS == 8:
        return hi, hi

    b = b8.to(tl.int16)
    vl = (((b & 0xF) ^ 8) - 8).to(tl.float32)
    lo = acc_lo + vl * s
    return hi, lo


@triton.jit
def _accumulate(
    in_ptr,
    in_step_ptr,
    in_byte,
    g,
    cmask,
    epit,
    gpit,
    num_tensors,
    NUM_TENSORS: tl.constexpr,
    NUM_BITS: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    MASKED: tl.constexpr,
    ALIGNED: tl.constexpr,
    CHUNK: tl.constexpr,
):
    acc_hi = tl.zeros((CHUNK,), dtype=tl.float32)
    if NUM_BITS == 4:
        acc_lo = tl.zeros((CHUNK,), dtype=tl.float32)
    else:
        acc_lo = acc_hi

    if NUM_TENSORS > 0:
        for j in tl.static_range(NUM_TENSORS):
            acc_hi, acc_lo = _dr_slice(
                in_ptr,
                in_step_ptr,
                j,
                in_byte,
                g,
                cmask,
                epit,
                gpit,
                NUM_BITS,
                SYMMETRIC,
                MASKED,
                ALIGNED,
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
                g,
                cmask,
                epit,
                gpit,
                NUM_BITS,
                SYMMETRIC,
                MASKED,
                ALIGNED,
                acc_hi,
                acc_lo,
            )
    return acc_hi, acc_lo


@triton.jit
def _offset_sum(
    step_ptr,
    g,
    gpit,
    num_tensors,
    NUM_TENSORS: tl.constexpr,
    NEED: tl.constexpr,
):
    total = tl.zeros((1,), dtype=tl.float32)
    if NEED:
        if NUM_TENSORS > 0:
            for j in tl.static_range(NUM_TENSORS):
                total = total + tl.load(step_ptr + j * 2 * gpit + 2 * g + 1)
        else:
            for j in range(num_tensors):
                total = total + tl.load(step_ptr + j * 2 * gpit + 2 * g + 1)
    return total


@triton.jit
def _group_stats(
    acc_hi,
    acc_lo,
    bmask,
    NUM_BITS: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    MASKED: tl.constexpr,
    LEAN_STATS: tl.constexpr,
):
    if not LEAN_STATS:
        if SYMMETRIC:
            m = tl.maximum(tl.abs(acc_hi), tl.abs(acc_lo))
            if MASKED:
                m = tl.where(bmask, m, 0.0)
            return tl.max(m), 0.0
        hi = tl.maximum(acc_hi, acc_lo)
        lo = tl.minimum(acc_hi, acc_lo)
        if MASKED:
            hi = tl.where(bmask, hi, float("-inf"))
            lo = tl.where(bmask, lo, float("inf"))
        return tl.max(hi), tl.min(lo)

    if SYMMETRIC:
        hi_fill = 0.0
        lo_fill = 0.0
    else:
        hi_fill = float("-inf")
        lo_fill = float("inf")

    if MASKED:
        hi_src = tl.where(bmask, acc_hi, hi_fill)
        hi_src_min = tl.where(bmask, acc_hi, lo_fill)
    else:
        hi_src = acc_hi
        hi_src_min = acc_hi
    hi_max = tl.max(hi_src)
    hi_min = tl.min(hi_src_min)

    if NUM_BITS == 4:
        if MASKED:
            lo_src = tl.where(bmask, acc_lo, hi_fill)
            lo_src_min = tl.where(bmask, acc_lo, lo_fill)
        else:
            lo_src = acc_lo
            lo_src_min = acc_lo
        lo_max = tl.max(lo_src)
        lo_min = tl.min(lo_src_min)
    else:
        lo_max, lo_min = hi_max, hi_min

    if SYMMETRIC:
        stat = tl.maximum(tl.maximum(hi_max, -hi_min), tl.maximum(lo_max, -lo_min))
        return stat, 0.0
    return tl.maximum(hi_max, lo_max), tl.minimum(hi_min, lo_min)


@triton.jit
def _quant_params(stat, mn, NUM_BITS: tl.constexpr, SYMMETRIC: tl.constexpr):
    LEVELS = 1 << NUM_BITS

    if SYMMETRIC:
        qscale = tl.where(stat == 0.0, 1.0, LEVELS / (2.0 * stat))
        step = tl.where(stat == 0.0, 1.0, (2.0 * stat) / LEVELS)
        off = 0.0
    else:
        rng = stat - mn
        qscale = tl.where(rng == 0.0, 1.0, LEVELS / rng)
        step = tl.where(rng == 0.0, 1.0, rng / LEVELS)
        off = (stat + mn) * 0.5
    return qscale, step, off


@triton.jit
def _store_params(out_step_ptr, g, step, off, SYMMETRIC: tl.constexpr):
    if SYMMETRIC:
        tl.store(
            out_step_ptr + g + tl.arange(0, 1),
            tl.zeros((1,), tl.float32) + step,
        )
    else:
        k2 = tl.arange(0, 2)
        vals = tl.where(
            k2 == 0,
            tl.zeros((2,), tl.float32) + step,
            tl.zeros((2,), tl.float32) + off,
        )
        tl.store(out_step_ptr + 2 * g + k2, vals)


@triton.jit
def _requant_store(
    out_ptr,
    in_byte,
    bmask,
    acc_hi,
    acc_lo,
    qscale,
    off,
    NUM_BITS: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    MASKED: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    QMAX = (1 << (NUM_BITS - 1)) - 1

    if SYMMETRIC:
        qf = (acc_hi * qscale + 12582912.0) - 12582912.0
    else:
        qf = ((acc_hi - off) * qscale + 12582912.0) - 12582912.0
    if NUM_BITS == 8:
        out_v = tl.minimum(qf, QMAX).to(tl.int8)
    else:
        qh = tl.minimum(qf, QMAX).to(tl.int16)
        if SYMMETRIC:
            qf = (acc_lo * qscale + 12582912.0) - 12582912.0
        else:
            qf = ((acc_lo - off) * qscale + 12582912.0) - 12582912.0
        ql = tl.minimum(qf, QMAX).to(tl.int16)
        out_v = ((qh << 4) | (ql & 0xF)).to(tl.int8)

    if MASKED:
        tl.store(out_ptr + in_byte, out_v, mask=bmask)
    elif ALIGNED:
        tl.store(out_ptr + tl.multiple_of(in_byte, 16), out_v)
    else:
        tl.store(out_ptr + in_byte, out_v)


@libentry()
@triton.jit
def _dequant_reduce_kernel(
    out_ptr,
    out_step_ptr,
    in_ptr,
    in_step_ptr,
    BYTES_PER_GROUP,
    EPIT,
    GPIT,
    num_tensors,
    NUM_TENSORS: tl.constexpr,
    NUM_BITS: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
    ALIGNED: tl.constexpr,
    LEAN_STATS: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    kb = tl.arange(0, BLOCK)
    bmask = kb < BYTES_PER_GROUP

    if I64:
        gg = pid.to(tl.int64)
        epit = EPIT.to(tl.int64)
        gpit = GPIT.to(tl.int64)
        in_byte = gg * BYTES_PER_GROUP.to(tl.int64) + kb
    else:
        gg = pid
        epit = EPIT
        gpit = GPIT
        in_byte = pid * BYTES_PER_GROUP + kb

    acc_hi, acc_lo = _accumulate(
        in_ptr,
        in_step_ptr,
        in_byte,
        gg,
        bmask,
        epit,
        gpit,
        num_tensors,
        NUM_TENSORS,
        NUM_BITS,
        SYMMETRIC,
        MASKED,
        ALIGNED,
        BLOCK,
    )
    off_sum = _offset_sum(
        in_step_ptr, gg, gpit, num_tensors, NUM_TENSORS, not SYMMETRIC
    )
    stat, mn = _group_stats(
        acc_hi, acc_lo, bmask, NUM_BITS, SYMMETRIC, MASKED, LEAN_STATS
    )
    qscale, step, off_q = _quant_params(stat, mn, NUM_BITS, SYMMETRIC)
    if SYMMETRIC:
        _store_params(out_step_ptr, gg, step, 0.0, SYMMETRIC)
    else:
        _store_params(out_step_ptr, gg, step, off_q + off_sum, SYMMETRIC)
    _requant_store(
        out_ptr,
        in_byte,
        bmask,
        acc_hi,
        acc_lo,
        qscale,
        off_q,
        NUM_BITS,
        SYMMETRIC,
        MASKED,
        ALIGNED,
    )


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
    MASKED: tl.constexpr,
    ALIGNED: tl.constexpr,
    LEAN_STATS: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    kc = tl.arange(0, CHUNK)

    if I64:
        gg = pid.to(tl.int64)
        epit = EPIT.to(tl.int64)
        gpit = GPIT.to(tl.int64)
    else:
        gg = pid
        epit = EPIT
        gpit = GPIT

    if SYMMETRIC:
        stat = tl.zeros((1,), dtype=tl.float32)
    else:
        stat = tl.full((1,), float("-inf"), dtype=tl.float32)
    mn = tl.full((1,), float("inf"), dtype=tl.float32)
    off_sum = _offset_sum(
        in_step_ptr, gg, gpit, num_tensors, NUM_TENSORS, not SYMMETRIC
    )

    for c in range(num_chunks):
        if I64:
            off = c.to(tl.int64) * CHUNK + kc
            in_byte = gg * BYTES_PER_GROUP.to(tl.int64) + off
        else:
            off = c * CHUNK + kc
            in_byte = gg * BYTES_PER_GROUP + off
        cmask = off < BYTES_PER_GROUP
        acc_hi, acc_lo = _accumulate(
            in_ptr,
            in_step_ptr,
            in_byte,
            gg,
            cmask,
            epit,
            gpit,
            num_tensors,
            NUM_TENSORS,
            NUM_BITS,
            SYMMETRIC,
            MASKED,
            ALIGNED,
            CHUNK,
        )
        chunk_stat, chunk_mn = _group_stats(
            acc_hi,
            acc_lo,
            cmask,
            NUM_BITS,
            SYMMETRIC,
            MASKED,
            LEAN_STATS,
        )
        stat = tl.maximum(stat, chunk_stat)
        if not SYMMETRIC:
            mn = tl.minimum(mn, chunk_mn)

    qscale, step, off_q = _quant_params(stat, mn, NUM_BITS, SYMMETRIC)
    if SYMMETRIC:
        _store_params(out_step_ptr, gg, step, 0.0, SYMMETRIC)
    else:
        _store_params(out_step_ptr, gg, step, off_q + off_sum, SYMMETRIC)

    for c in range(num_chunks):
        if I64:
            off = c.to(tl.int64) * CHUNK + kc
            in_byte = gg * BYTES_PER_GROUP.to(tl.int64) + off
        else:
            off = c * CHUNK + kc
            in_byte = gg * BYTES_PER_GROUP + off
        cmask = off < BYTES_PER_GROUP
        acc_hi, acc_lo = _accumulate(
            in_ptr,
            in_step_ptr,
            in_byte,
            gg,
            cmask,
            epit,
            gpit,
            num_tensors,
            NUM_TENSORS,
            NUM_BITS,
            SYMMETRIC,
            MASKED,
            ALIGNED,
            CHUNK,
        )
        _requant_store(
            out_ptr,
            in_byte,
            cmask,
            acc_hi,
            acc_lo,
            qscale,
            off_q,
            NUM_BITS,
            SYMMETRIC,
            MASKED,
            ALIGNED,
        )


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
    if input_vals.device != input_scales.device:
        raise ValueError(
            "input_vals and input_scales must be on the same device, got "
            f"{input_vals.device} and {input_scales.device}"
        )
    if input_vals.dim() == 0:
        raise ValueError("input_vals must have at least one dimension")

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
    if total_bytes == 0:
        raise ValueError("input_vals must contain at least one element")

    groups_per_in_tensor = in_groups // num_tensors
    elems_per_in_tensor = total_bytes // num_tensors

    if elems_per_in_tensor % groups_per_in_tensor != 0:
        raise ValueError(
            f"elems_per_in_tensor({elems_per_in_tensor}) is not divisible by "
            f"groups_per_in_tensor({groups_per_in_tensor})"
        )

    elems_per_in_group = elems_per_in_tensor // groups_per_in_tensor
    if elems_per_in_tensor % out_groups != 0:
        raise ValueError(
            f"elems_per_in_tensor({elems_per_in_tensor}) is not divisible by "
            f"out_groups({out_groups})"
        )
    elems_per_out_group = elems_per_in_tensor // out_groups
    if elems_per_in_group != elems_per_out_group:
        raise RuntimeError(
            "in_groups == out_groups * devices_per_node should make one input "
            f"group per output group, got {elems_per_in_group} vs "
            f"{elems_per_out_group}"
        )

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

    block, lean_stats = _pick_tile(elems_per_out_group, num_bits, symmetric)
    static_tensors = _pick_static(
        num_tensors, elems_per_out_group, block, num_bits, lean_stats
    )
    block_max = _block_cap(num_bits)

    use_i64 = total_bytes + block_max >= (1 << 31) or in_groups * scales_elems >= (
        1 << 31
    )

    num_warps = _pick_warps(block)

    with torch_device_fn.device(input_vals.device):
        if block >= elems_per_out_group:
            masked = block != elems_per_out_group
            aligned = (not masked) and elems_per_out_group % 16 == 0
            _dequant_reduce_kernel[(out_groups,)](
                out,
                out_st,
                input_vals,
                input_scales,
                elems_per_out_group,
                elems_per_in_tensor,
                groups_per_in_tensor,
                num_tensors,
                NUM_TENSORS=static_tensors,
                NUM_BITS=num_bits,
                SYMMETRIC=symmetric,
                BLOCK=block,
                MASKED=masked,
                ALIGNED=aligned,
                LEAN_STATS=lean_stats,
                I64=use_i64,
                num_warps=num_warps,
                num_stages=_NUM_STAGES,
            )
        else:
            num_chunks = triton.cdiv(elems_per_out_group, block)
            masked = (elems_per_out_group % block) != 0
            aligned = (not masked) and elems_per_out_group % 16 == 0
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
                CHUNK=block,
                MASKED=masked,
                ALIGNED=aligned,
                LEAN_STATS=lean_stats,
                I64=use_i64,
                num_warps=num_warps,
                num_stages=_NUM_STAGES,
            )

    return [out, out_st]
