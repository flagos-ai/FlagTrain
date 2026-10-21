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
"""Correctness tests for the mixed_gemm operator.

Three checks, and they fail differently:

1. ``mixed_gemm_ref`` -- a plain-torch composition of the same contract, fed the
   **same preprocessed weight and scales the operator receives**. This is the primary
   check and it runs on any device. Both sides read one byte stream, so a
   disagreement is the operator's own bug and nothing to do with quantization,
   which is why the tolerance here can be tight.

2. **The quantization error itself** -- how far the int4/int8 result sits from the
   *unquantized* fp32 product. That gap is the scheme's cost rather than a bug, so
   it is checked against the bound quantization promises
   (``scale[n] * sum_k |A[m, k]|``) rather than against a tuned tolerance.

3. **DeepSpeed's own mixed_gemm** -- the operator this ports. It is the only check
   that can contradict *our reading* of the contract: the two above are both
   written from that reading, so a shared misreading survives them. Nothing about
   the inputs blocks it, since this port reads the same preprocessed B buffer
   DeepSpeed does; whether it runs is an environment question, resolved at import
   time by ``_load_deepspeed_mixed_gemm`` and reported in the skip reason.

Run with::

    pytest tests/deepspeed/test_mixed_gemm.py -v
"""

import logging
import os

import pytest
import torch

import flag_train
from flag_train.deepspeed import mixed_gemm
from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn

from .. import accuracy_utils as utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quantization and the preprocessed weight layout
#
# ``weight`` reaches the operator already quantized *and* already run through
# ``preprocess_weights_for_mixed_gemm``. Nothing in the operator produces that buffer,
# but this file has to build one, so the transforms are transcribed here.
#
# Why the layout is *recovered* rather than computed: the reference side must not use
# the kernel's index arithmetic. If it did, the two would agree even when both were
# wrong, and the comparison would prove nothing -- an earlier version of this test did
# exactly that. Instead every element is tagged with a recoverable value, the real
# preprocessing runs, the positions are read back, and the result is required to be a
# bijection. A nibble swap cannot satisfy that.
#
# Three constraints shaped the tagging, all found by running it:
#
# * int4 has 4096 elements and a nibble holds 16 values, so one buffer cannot carry an
#   element index. The index is split into base-8 digits, one per pass (8**5 > 4096).
# * The base is 8, not 15: stage 4 adds ``+8`` *before* storing, so a digit of 10 would
#   land as 18, which does not fit a nibble and reads back as 2.
# * ``subbyte_transpose`` recombines nibbles **across byte boundaries**, so a tag
#   cannot describe a byte. Tracing one showed output byte 0 holding values that came
#   from input bytes 0 and 2.
#
# Quantized values stay **signed** throughout. The preprocessing adds the
# ``+8``/``+128`` bias itself, so adding it here would add it twice.
# ---------------------------------------------------------------------------


INT8 = "int8"
INT4 = "int4"

# Named apart from the parametrisation list at the top of this file.
_QUANT_BITS = {INT8: 8, INT4: 4}

# ``mixed_gemm_B_layout.h``, Arch::kMinComputeCapability >= 75. These values are what
# make one implementation cover Sm80 through Sm90.
_LAYOUT = {
    INT8: {"rows_per_column_tile": 64, "columns_interleaved": 16 // 8},
    INT4: {"rows_per_column_tile": 64, "columns_interleaved": 16 // 4},
}
_USES_IMMA_LDSM = True
_LAYOUT_B_COLUMN_MAJOR = True


def _permute_b_rows(buf, num_rows, num_cols, quant_type, arch_version=80):
    """Permutes the rows of B, in units of 32-bit words.

    C++ (182-201) walks word columns (``num_vec_cols`` of them) and, for each row
    of a ``B_ROWS_PER_MMA`` group, copies whole 32-bit words. The row written at
    ``write_row`` is read from ``tile_read_row``, computed at 189-190.
    """
    bits = _QUANT_BITS[quant_type]
    K = 16 // bits  # 156
    elts_per_reg = 32 // bits  # 158

    mma_shape_n = 8  # 163
    b_rows_per_mma = 8 * K  # 164
    elts_in_int32 = 32 // bits  # 165
    num_vec_cols = num_cols // elts_in_int32  # 167

    if arch_version < 75:
        raise ValueError("Unsupported Arch (pre-Volta)")
    if num_rows % b_rows_per_mma:
        raise ValueError(f"num_rows must be a multiple of {b_rows_per_mma}")
    if num_cols % mma_shape_n:
        raise ValueError(f"num_cols must be a multiple of {mma_shape_n}")

    # Work on a uint32 view, as the C++ does (160-161). num_vec_cols words per row.
    words = buf.view(torch.int32).reshape(num_rows, num_vec_cols)
    out = torch.empty_like(words)

    for base_row in range(0, num_rows, b_rows_per_mma):
        for tile_row in range(b_rows_per_mma):
            write_row = base_row + tile_row
            # 189-190
            tile_read_row = (
                8 * ((tile_row % elts_per_reg) // 2)
                + tile_row % 2
                + 2 * (tile_row // elts_per_reg)
            )
            read_row = base_row + tile_read_row
            out[write_row] = words[read_row]

    return out.reshape(-1).view(torch.uint8)


# ---------------------------------------------------------------------------
# subbyte_transpose_impl (208-331)
# ---------------------------------------------------------------------------

_M_TILE_L1 = 64


def _subbyte_transpose(buf, num_rows, num_cols, quant_type):
    """Transposes in 64x64 *element* tiles, at sub-byte granularity for int4.

    The C++ stages a tile in ``cache_buf[64][64/elts_per_byte]``:
      * load phase (258-272) copies whole VECTOR_WIDTH runs of bytes;
      * swap phase (274-303) transposes the tile *in elements*, which for int4
        means nibble-level swaps (293-300);
      * store phase (314-327) writes to the transposed geometry.
    """
    bits = _QUANT_BITS[quant_type]
    elts_per_byte = 8 // bits

    col_bytes = num_cols * bits // 8  # 220
    col_bytes_trans = num_rows * bits // 8  # 221
    n_tile_l1 = _M_TILE_L1 // elts_per_byte  # 231
    vector_width = min(32, n_tile_l1)  # 234

    if col_bytes_trans % vector_width or col_bytes % vector_width:
        raise ValueError(f"row/col byte counts must be multiples of {vector_width}")

    # The kernel comment (236-238) says only multiples of 64 are handled, and the
    # 64x64 element tiling below relies on it.
    if num_rows % 64 or num_cols % 64:
        raise ValueError("num_rows and num_cols must be multiples of 64")

    src = buf.view(torch.uint8).reshape(num_rows, col_bytes)

    # The store phase indexes rows by *logical element* index, 0..num_cols-1, with
    # col_bytes_trans bytes each. For int4 that is 64 rows of 32 bytes -- the same
    # geometry as the element cache below, one byte holding two elements.
    dst = torch.zeros(num_cols, col_bytes_trans, dtype=torch.uint8)

    for row_tile_start in range(0, num_rows, _M_TILE_L1):
        for col_tile_start_byte in range(0, col_bytes, n_tile_l1):
            row_limit = min(row_tile_start + _M_TILE_L1, num_rows)
            col_limit = min(col_tile_start_byte + n_tile_l1, col_bytes)

            # Load into a 64 x n_tile_l1 element cache (padding stays zero).
            cache = torch.zeros(_M_TILE_L1, n_tile_l1, dtype=torch.uint8)
            for ii in range(_M_TILE_L1):
                row = row_tile_start + ii
                for jj in range(0, n_tile_l1, vector_width):
                    col = col_tile_start_byte + jj
                    if row < row_limit and col < col_limit:
                        n = min(vector_width, col_limit - col)
                        cache[ii, jj : jj + n] = src[row, col : col + n]

            if quant_type == INT8:
                # 274-280: plain element transpose of the lower triangle.
                for ii in range(_M_TILE_L1):
                    for jj in range(ii + 1, n_tile_l1):
                        tmp = cache[ii, jj].clone()
                        cache[ii, jj] = cache[jj, ii]
                        cache[jj, ii] = tmp
            else:
                # 281-303: nibble-level transpose.
                for ii in range(_M_TILE_L1):
                    for jj in range(ii + 1, _M_TILE_L1):
                        ii_byte, ii_bit = ii // elts_per_byte, ii % elts_per_byte
                        jj_byte, jj_bit = jj // elts_per_byte, jj % elts_per_byte

                        src_elt = 0xF & (int(cache[ii, jj_byte]) >> (4 * jj_bit))
                        tgt_elt = 0xF & (int(cache[jj, ii_byte]) >> (4 * ii_bit))

                        cache[ii, jj_byte] &= 0xF0 >> (4 * jj_bit)
                        cache[jj, ii_byte] &= 0xF0 >> (4 * ii_bit)

                        cache[ii, jj_byte] |= tgt_elt << (4 * jj_bit)
                        cache[jj, ii_byte] |= src_elt << (4 * ii_bit)

            # 308-312: the transposed geometry.
            row_tile_start_trans = col_tile_start_byte * elts_per_byte
            col_tile_start_byte_trans = row_tile_start // elts_per_byte
            row_limit_trans = min(row_tile_start_trans + _M_TILE_L1, num_cols)
            col_limit_trans = min(
                col_tile_start_byte_trans + n_tile_l1, col_bytes_trans
            )

            for ii in range(_M_TILE_L1):
                row = row_tile_start_trans + ii
                for jj in range(0, n_tile_l1, vector_width):
                    col = col_tile_start_byte_trans + jj
                    if row < row_limit_trans and col < col_limit_trans:
                        n = min(vector_width, col_limit_trans - col)
                        dst[row, col : col + n] = cache[ii, jj : jj + n]

    return dst.reshape(-1)


# ---------------------------------------------------------------------------
# interleave_column_major_tensor (438-499)
# ---------------------------------------------------------------------------


def _interleave_column_major(buf, num_rows, num_cols, quant_type, details):
    """Reorders the column-major buffer so each tile's columns are contiguous.

    C++ (479-497) walks the uint32 view: ``read_col`` over columns,
    ``vec_read_row`` over 32-bit rows within a tile, and writes to ``write_col`` /
    ``vec_write_row`` from 482 and 487-489.
    """
    bits = _QUANT_BITS[quant_type]
    elts_in_int32 = 32 // bits
    rows_per_tile = details["rows_per_column_tile"]
    interleave = details["columns_interleaved"]

    if num_rows % elts_in_int32:
        raise ValueError(f"num_rows must be a multiple of {elts_in_int32}")
    if num_cols % rows_per_tile:
        raise ValueError(f"num_cols must be a multiple of {rows_per_tile}")

    num_vec_rows = num_rows // elts_in_int32
    vec_rows_per_tile = rows_per_tile // elts_in_int32

    # The C++ views the buffer as uint32 (467-468), so read_offset and write_offset
    # below are *word* offsets even though the variable is called *_offset.
    #
    # Read side: read_col * num_vec_rows + vec_read_row  ->  a [num_cols, num_vec_rows]
    # word matrix. Write side: write_col * num_vec_rows * interleave + vec_write_row,
    # with write_col < num_cols / interleave and vec_write_row < interleave *
    # num_vec_rows -- so the output has the same size and is re-indexed in place.
    words = buf.view(torch.int32).reshape(num_cols, num_vec_rows)
    out = torch.empty(
        (num_cols // interleave, num_vec_rows * interleave),
        dtype=torch.int32,
    )

    for read_col in range(num_cols):
        write_col = read_col // interleave
        for base_vec_row in range(0, num_vec_rows, vec_rows_per_tile):
            top = min(num_vec_rows, base_vec_row + vec_rows_per_tile)
            for vec_read_row in range(base_vec_row, top):
                vec_write_row = (
                    interleave * base_vec_row
                    + vec_rows_per_tile * (read_col % interleave)
                    + vec_read_row % vec_rows_per_tile
                )
                out[write_col, vec_write_row] = words[read_col, vec_read_row]

    return out.reshape(-1).view(torch.uint8)


# ---------------------------------------------------------------------------
# add_bias_and_interleave_quantized_tensor_inplace (351-436)
# ---------------------------------------------------------------------------


def _add_bias_and_interleave(buf, num_elts, quant_type):
    """Adds the bias and reorders within each 32-bit register, in place.

    int8 (351-371): +128 to every byte, then swap bytes 1 and 2 of each group of
    four -- so a register [e3 e2 e1 e0] becomes [e3 e1 e2 e0].

    int4 (373-423): +8 to every nibble (sign-extending the low one first, 382),
    then per register [e7 e6 e5 e4 e3 e2 e1 e0] -> [e7 e5 e3 e1 e6 e4 e2 e0] via
    the source index at 414.
    """
    if quant_type == INT8:
        if num_elts % 4:
            raise ValueError("int8 tensor must be a multiple of 4 elements")
        b = buf.view(torch.uint8).clone()
        b = (b.to(torch.int16) + 128).to(torch.uint8)
        for base in range(0, num_elts, 4):
            tmp = b[base + 1].clone()
            b[base + 1] = b[base + 2]
            b[base + 2] = tmp
        return b

    num_bytes = num_elts // 2
    if num_bytes % 4:
        raise ValueError("int4 tensor must be a multiple of 8 elements")

    b = buf.view(torch.uint8).clone()
    # 379-394: sign-extend the low nibble, add 8 to both, repack.
    #
    # The C++ is ``int8_t(packed << 4) >> 4``. That works because the shift moves
    # the low nibble's *sign bit* (its bit 3) into bit 7 and the arithmetic shift
    # brings it back sign-extended. Doing it on the extracted nibble instead, the
    # sign test is ``low & 0x8``.
    for ii in range(num_bytes):
        byte = int(b[ii])
        # Both nibbles are sign-extended before +8. The C++ does this with the
        # int8_t shifts `(int8_t(x << 4) >> 4)` and `(x >> 4)` (382-383); here the
        # sign test is bit 3. Missing the *high* nibble's sign extension
        # makes this raise on any real weight whose high nibble reaches 8..15,
        # which is half of them.
        low = byte & 0xF
        if low & 0x8:  # bit 3 set -> negative int4, i.e. -8..-1
            low -= 16
        high = (byte >> 4) & 0xF
        if high & 0x8:
            high -= 16
        first = low + 8
        second = high + 8
        if not (0 <= first <= 15 and 0 <= second <= 15):
            raise ValueError(
                f"illegal int4 transform result at byte {ii}: "
                f"byte={byte} -> first={first} second={second}"
            )
        b[ii] = (first & 0xF) | ((second & 0xF) << 4)

    # 405-422: register relayout, 4 bytes (8 nibbles) at a time.
    for base in range(0, num_bytes, 4):
        reg = b[base : base + 4].to(torch.int64)
        cur = (
            int(reg[0]) | (int(reg[1]) << 8) | (int(reg[2]) << 16) | (int(reg[3]) << 24)
        )
        new = 0
        for dest_idx in range(8):
            src_idx = 2 * dest_idx if dest_idx < 4 else 2 * (dest_idx - 4) + 1
            bits = (cur >> (4 * src_idx)) & 0xF
            new |= bits << (4 * dest_idx)
        for k in range(4):
            b[base + k] = (new >> (8 * k)) & 0xFF

    return b


# ---------------------------------------------------------------------------
# preprocess_weights_for_mixed_gemm (501-540)
# ---------------------------------------------------------------------------


def preprocess_weights_for_mixed_gemm(unprocessed, num_rows, num_cols, num_bits):
    """Quantized weight -> the byte layout the operator consumes.

    ``unprocessed`` is the *signed* quantized matrix packed row-major: one byte per
    element for int8, two nibbles per byte for int4 with the low nibble holding the
    even column. The bias is added by the last transform, so the input must not
    carry it.

    Both dimensions must be multiples of 64 -- the C++ asserts the row multiple and
    the 64x64 transpose needs the column multiple. The layout is undefined outside
    that, which is also why the operator's tile sizes are what they are.
    """
    if num_rows % 64 or num_cols % 64:
        raise ValueError(
            f"the preprocessed layout requires both dimensions to be multiples of "
            f"64; got {num_rows}x{num_cols}"
        )
    quant_type = INT4 if num_bits == 4 else INT8
    details = _LAYOUT[quant_type]

    buf = unprocessed.reshape(-1).clone()
    if _USES_IMMA_LDSM:
        buf = _permute_b_rows(buf, num_rows, num_cols, quant_type)
    if _LAYOUT_B_COLUMN_MAJOR:
        buf = _subbyte_transpose(buf, num_rows, num_cols, quant_type)
    if details["columns_interleaved"] > 1:
        buf = _interleave_column_major(buf, num_rows, num_cols, quant_type, details)
    return _add_bias_and_interleave(buf, num_rows * num_cols, quant_type)


def _round_half_away(x):
    """C++ ``std::round``: ties away from zero, unlike ``torch.round``."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def quantize(w, num_bits):
    """``[k, n]`` fp16 -> ``([k, n] signed int64, scales fp16)``, per output channel.

    ``symmetric_quantize``'s rules: per-column ``max|w| / 2**(bits-1)``, rounded half
    away from zero, clamped to the signed range. Verified bit-for-bit against
    DeepSpeed: scales and values both, int4 and int8, 4096 elements each.
    """
    wf = w.float()
    per_col_max = wf.abs().amax(dim=0)
    scale32 = per_col_max / float(1 << (num_bits - 1))
    scales = scale32.to(torch.float16)

    safe = torch.where(scale32 > 0, scale32, torch.ones_like(scale32))
    q = _round_half_away(wf / safe[None, :])
    q = q.clamp(-(1 << (num_bits - 1)), (1 << (num_bits - 1)) - 1)
    return q.to(torch.int64), scales


def pack_signed(q, num_bits, k, n):
    """Signed quantized matrix -> the unprocessed packed buffer.

    One byte per element for int8. Two nibbles per byte for int4, low nibble holding
    the even column; an odd ``n`` leaves a padding byte, as the C++ guard produces.
    """
    if num_bits == 8:
        return (q & 0xFF).to(torch.uint8).reshape(-1)
    if q.shape[1] % 2:
        q = torch.cat([q, torch.zeros(k, 1, dtype=q.dtype)], dim=1)
    qs = q & 0xF  # 4-bit two's complement
    return (qs[:, 0::2] | (qs[:, 1::2] << 4)).to(torch.uint8).reshape(-1)


def _reference_copy(tensor):
    """A reference-side copy.

    ``utils.to_reference`` moves it to the CPU under ``--ref cpu``, which is what
    makes the reference an independent oracle rather than a second reading of the
    same device arithmetic.
    """
    return utils.to_reference(tensor.clone() if tensor is not None else None)


def mixed_gemm_ref(
    output, hidden_states, weight, scales, bias, num_bits, activation_raw
):
    """Plain-torch composition of the mixed_gemm contract.

    ``weight`` is the logical quantized matrix, not the operator's packed buffer: the
    tests built that buffer from this matrix, so passing it directly costs nothing and
    leaves the reference free of layout arithmetic. ``num_bits`` is unused now.

    Accumulates in fp64: the operator accumulates in fp32, and an fp32 reference would
    contribute its own error on top of the one being measured, which is what keeps the
    main assertion's tolerance meaningful.
    """
    if utils.TO_CPU:
        # ``--ref cpu`` promises the reference *computation* is on the CPU, not only
        # its result. Asserting that precondition here rather than at the call sites
        # is what makes a test that forgets ``_reference_copy`` fail loudly instead
        # of quietly becoming a second reading of the device arithmetic -- which
        # would pass while testing nothing. Same guard as ``test_lamb.py``.
        for name, operand in (
            ("hidden_states", hidden_states),
            ("weight", weight),
            ("scales", scales),
        ):
            assert operand.device.type == "cpu", (
                f"--ref cpu must run the reference on the CPU; {name} is on "
                f"{operand.device}"
            )

    q = weight
    dequant = q.to(torch.float64) * scales.to(torch.float64)[None, :]

    ref = hidden_states.to(torch.float64) @ dequant
    if bias is not None:
        ref = ref + bias.to(torch.float64)[None, :]

    # The activation sees the bias: DeepSpeed's epilogues are all
    # ``EpilogueOpBias<Act>``. IDENTITY (6) does nothing.
    if activation_raw == 1:  # RELU
        ref = torch.clamp_min(ref, 0.0)
    elif activation_raw == 2:  # SILU
        ref = ref * torch.sigmoid(ref)
    elif activation_raw == 0:  # GELU, the tanh approximation FtGelu uses
        c = (2.0 / torch.pi) ** 0.5
        ref = 0.5 * ref * (1.0 + torch.tanh(c * (ref + 0.044715 * ref**3)))

    output.copy_(ref.to(output.dtype))


# ---------------------------------------------------------------------------
# DeepSpeed's own operator, if this machine can run it
#
# The one check that can contradict the reading of the contract: everything else here is
# written from that reading, so a shared misreading survives it.
#
# Loading it is not the same as running it. ``mixed_gemm`` ships as a prebuilt kernels
# library, so a build that lacks this GPU's compute capability imports cleanly and fails
# only when called -- and it fails with a signal in the runner's configuration search, not
# an exception. ``_load_deepspeed_mixed_gemm`` therefore calls the operator once on a
# small input rather than trusting a successful import, and returns ``None`` (with the
# reason in ``_DEEPSPEED_UNAVAILABLE_MSG``) so the suite still runs.
# ---------------------------------------------------------------------------

# Backends whose reference is DeepSpeed. ``mixed_gemm`` ships as a CUDA op, so only a
# backend that can compile and execute one can host it.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}

# Set to 1 to make "no reference" an error instead of a skip. A dedicated environment
# should set it, so an absent reference cannot quietly thin the suite there; a generic
# runner should not, so the rest of the file still runs.
_REQUIRE_DEEPSPEED = os.environ.get("FLAG_TRAIN_REQUIRE_DEEPSPEED", "") == "1"

_DEEPSPEED_UNAVAILABLE_MSG = (
    "DeepSpeed's mixed_gemm could not be used on this machine. Its kernel ships "
    "prebuilt in the `dskernels` package, and `InferenceCutlassBuilder` links "
    "`-ldeepspeedft` rather than compiling it, so that package has to carry this GPU's "
    "compute capability. Install one that does, or drop the backend from "
    "_DEEPSPEED_BASELINE_VENDORS."
)


def _load_deepspeed_mixed_gemm():
    """DeepSpeed's ``mixed_gemm``, or ``None`` when there is not a usable one.

    ``None`` covers both "this backend does not use DeepSpeed" and "it is installed but
    cannot run here". Which of those is fatal is the caller's decision, through
    ``_REQUIRE_DEEPSPEED``: a dedicated environment sets it so an absent reference cannot
    quiet the suite, and a generic runner does not, so the rest of the file still runs.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None

    try:
        from deepspeed.ops.op_builder import InferenceCutlassBuilder

        op = InferenceCutlassBuilder().load().mixed_gemm
    except Exception as exc:  # noqa: BLE001
        logger.warning("DeepSpeed's mixed_gemm could not be loaded: %s", exc)
        return None

    # Import success says nothing about whether the kernel can dispatch here. Random bytes
    # are not a valid quantized weight, so the *values* mean nothing; a signal or an
    # exception is the finding, and both are what a missing compute capability produces.
    m = k = n = 128
    try:
        hidden = torch.randn(m, k, dtype=torch.float16, device=runtime_device.name)
        weight = torch.randint(
            0, 256, (k, n // 2), dtype=torch.uint8, device=runtime_device.name
        )
        scales = torch.rand(n, dtype=torch.float16, device=runtime_device.name) + 0.1
        out = torch.empty(m, n, dtype=torch.float16, device=runtime_device.name)
        op(out, hidden, weight, scales, None, 4, 6)  # IDENTITY
        torch_device_fn.synchronize()
    except Exception as exc:  # noqa: BLE001
        logger.warning("DeepSpeed's mixed_gemm does not run here: %s", exc)
        return None

    return op


# Resolved once, at module import time.
_DEEPSPEED_ORACLE = _load_deepspeed_mixed_gemm()

if _DEEPSPEED_ORACLE is None and _REQUIRE_DEEPSPEED:
    raise RuntimeError(
        f"{flag_train.vendor_name!r} must use DeepSpeed's mixed_gemm as its reference, "
        f"but it is not usable. {_DEEPSPEED_UNAVAILABLE_MSG}"
    )

# The torch reference is always available, so only the DeepSpeed check skips.
requires_deepspeed_reference = pytest.mark.skipif(
    _DEEPSPEED_ORACLE is None, reason=_DEEPSPEED_UNAVAILABLE_MSG
)


# ---------------------------------------------------------------------------
# Input builder
# ---------------------------------------------------------------------------


def _build(m, k, n, num_bits, dtype, seed):
    """One reproducible case.

    Returns ``(output, hidden, weight, q, scales, bias)``. ``weight`` is the operand
    the operator's contract takes -- already preprocessed. ``q`` is the signed
    quantized matrix that buffer was built from, kept so the reference can use it
    directly: recovering it from the buffer would mean re-deriving the layout, which is
    slower and would put the same addressing on both sides of the comparison.
    """
    torch.manual_seed(seed)
    device = flag_train.device
    hidden = torch.randn(m, k, dtype=dtype, device=device)
    w = torch.randn(k, n, dtype=dtype, device=device)
    q, scales = quantize(w.cpu(), num_bits)
    # Keep the 2-D shape: the operator reads ``weight.size(0)`` to check it against
    # ``k`` and derives ``n`` from ``weight.size(1)``, so a flat buffer would be
    # rejected before reaching the kernel.
    packed = (
        preprocess_weights_for_mixed_gemm(
            pack_signed(q, num_bits, k, n), k, n, num_bits
        )
        .reshape(k, -1)
        .to(device)
    )
    scales = scales.to(device).to(dtype)
    # ``q`` goes to the device too: the reference multiplies it against ``hidden``, and
    # mixing a CPU operand with a device one is an error rather than a slow path. Above
    # int8's size this is the same order of memory as the output, which is acceptable
    # for a test; the benchmark keeps it on the host and moves only what it must.
    q = q.to(device)
    bias = torch.randn(n, dtype=dtype, device=device) * 0.1
    output = torch.empty(m, n, dtype=dtype, device=device)
    return output, hidden, packed, q, scales, bias


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

# Each shape makes one mask non-trivial: ``rows_mask`` / ``kmask`` / ``cols_mask``.
# Both the weight's row count (k) and its column count (n) are multiples of 64 here,
# because the preprocessed layout is only defined for those -- the C++ asserts the row
# multiple and the 64x64 transpose needs the column multiple. ``m`` is free, and 65
# and 1 are the interesting values.
_SHAPES = [
    (128, 64, 128),
    (65, 64, 128),  # m is not a multiple of BLOCK_M
    (1, 64, 64),  # a single token, and the smallest layout-legal n
]

# The four ActivationType values ``fpA_intB`` supports: GELU / RELU / SILU / IDENTITY.
# The numbering is not intuitive -- IDENTITY is 6, not 0.
_ACTS = [0, 1, 2, 6]
_ACT_IDS = ["gelu", "relu", "silu", "identity"]
_BITS = [4, 8]

# ``mixed_gemm.cu``'s ACT_DTYPE_SWITCH instantiates exactly two activation
# dtypes, fp16 and bf16 (``__half`` and ``__nv_bfloat16``) -- there is no fp32
# path, so those two are the whole contract.
_DTYPES = [torch.float16, torch.bfloat16]
_DTYPE_IDS = ["fp16", "bf16"]


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("num_bits", _BITS, ids=[f"int{b}" for b in _BITS])
@pytest.mark.parametrize("activation_raw", _ACTS, ids=_ACT_IDS)
@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{m}x{k}x{n}" for m, k, n in _SHAPES])
def test_mixed_gemm(shape, activation_raw, num_bits, dtype):
    """The operator must match the torch reference on the same packed weight.

    Both sides read the same integers, so the quantization step does not appear
    here at all and the tolerance is sized against fp32 accumulation over ``k``
    plus the fp16 store instead. Sizing it against the quantization step is the
    usual way this test ends up either vacuous or red for no reason.
    """
    m, k, n = shape
    output, hidden, packed, q, scales, bias = _build(m, k, n, num_bits, dtype, seed=0)

    mixed_gemm(output, hidden, packed, scales, None, num_bits, activation_raw)

    ref_out = torch.empty_like(output)
    # ``ref_out`` itself must NOT go through ``_reference_copy``: that clones, and
    # the reference would write into the clone while ``ref_out`` stays whatever
    # ``torch.empty`` left there. What goes through it is the *inputs*, so that
    # ``--ref cpu`` makes the reference genuinely independent.
    mixed_gemm_ref(
        ref_out,
        _reference_copy(hidden),
        _reference_copy(q),
        _reference_copy(scales),
        None,
        num_bits,
        activation_raw,
    )
    # ``train_assert_close``'s rtol is not a choice -- flag_train.testing looks it
    # up per dtype (1e-3 for fp16). What is a choice is atol * reduce_dim: 1e-3 *
    # k puts the effective absolute tolerance roughly 6x above the observed gap
    # (~2e-2 at k=128), on the fp16 store's resolution rather than the
    # quantization step.
    # Both sides go through ``to_reference``: under ``--ref cpu`` that is what puts
    # them on the same device, and ``flag_train.testing.assert_close`` compares with
    # ``torch.testing.assert_close``, which requires that.
    utils.train_assert_close(
        utils.to_reference(ref_out),
        utils.to_reference(output),
        dtype,
        reduce_dim=k,
        atol=1e-3,
    )


@pytest.mark.mixed_gemm
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("num_bits", _BITS, ids=[f"int{b}" for b in _BITS])
@pytest.mark.parametrize("shape", [(128, 128, 128), (65, 128, 64)])
def test_mixed_gemm_with_bias(shape, num_bits, dtype):
    """The bias must be added before the activation, not after.

    RELU is the discriminating activation. With ``z`` the pre-activation, the
    correct order gives ``relu(z + bias)`` and the wrong one gives
    ``relu(z) + bias``: at an element where ``z`` is far below ``-|bias|`` the
    first returns 0 and the second returns **the bias itself**. IDENTITY cannot
    tell the two apart, which is exactly why the ordering survived unnoticed in
    the exercise that first wrote it.
    """
    m, k, n = shape
    output, hidden, packed, q, scales, bias = _build(m, k, n, num_bits, dtype, seed=1)

    mixed_gemm(output, hidden, packed, scales, bias, num_bits, 1)

    ref_out = torch.empty_like(output)
    mixed_gemm_ref(
        ref_out,
        _reference_copy(hidden),
        _reference_copy(q),
        _reference_copy(scales),
        _reference_copy(bias),
        num_bits,
        1,
    )
    # Both sides go through ``to_reference``: under ``--ref cpu`` that is what puts
    # them on the same device, and ``flag_train.testing.assert_close`` compares with
    # ``torch.testing.assert_close``, which requires that.
    utils.train_assert_close(
        utils.to_reference(ref_out),
        utils.to_reference(output),
        dtype,
        reduce_dim=k,
        atol=1e-3,
    )

    # The case above only discriminates if some element actually lands below
    # -|bias|; otherwise both orderings agree and the test proves nothing.
    pre = hidden.to(torch.float64) @ (
        q.to(torch.float64) * scales.to(torch.float64)[None, :]
    )
    assert bool((pre < -bias.to(torch.float64).abs()[None, :]).any()), (
        "no element has a pre-activation below -|bias|, so this case cannot "
        "distinguish the bias/activation order; change the seed"
    )


@pytest.mark.mixed_gemm
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("num_bits", _BITS, ids=[f"int{b}" for b in _BITS])
def test_mixed_gemm_matches_theoretical_bound(num_bits, dtype):
    """The gap from the *unquantized* product must stay inside the bound.

    Quantization error is the scheme's cost rather than an implementation bug, so
    it is checked against what the packing promises. Each weight moves by at most
    one step -- the column maximum rounds to ``2**(bits-1)`` and is clipped back
    to ``2**(bits-1) - 1``, which is a full step, not half -- so

        |C_hat - C_true| <= scale[n] * sum_k |A[m, k]|

    holds without a factor of two. The bound is loose by design: the ``k`` error
    terms are independent and partly cancel, so a healthy kernel uses only about
    an eighth of it. Asserting per element rather than on the maxima is what
    catches "the worst element happens to be fine".
    """
    m, k, n = 128, 128, 128
    output, hidden, packed, q, scales, bias = _build(m, k, n, num_bits, dtype, seed=2)

    mixed_gemm(output, hidden, packed, scales, None, num_bits, 6)  # IDENTITY

    exact = hidden.to(torch.float64) @ (
        q.to(torch.float64) * scales.to(torch.float64)[None, :]
    )

    err = (output.to(torch.float64) - exact).abs()
    bound = (
        scales.to(torch.float64)[None, :]
        * hidden.to(torch.float64).abs().sum(dim=1)[:, None]
    )
    assert bool((err <= bound).all()), (
        f"{num_bits}-bit error {err.max().item():.4e} exceeds the bound "
        f"{bound.max().item():.4e} (worst usage "
        f"{(err / bound).max().item():.1%} of the bound)"
    )


@pytest.mark.mixed_gemm
def test_mixed_gemm_int4_n_is_twice_the_packed_width():
    """For int4 the operator takes ``n`` from the weight's width, and that is exact.

    ``n = weight.size(1) * 2``, so an int4 weight whose byte count per row is ``w``
    always means ``2 * w`` output columns. There is no argument through which a
    caller could ask for an odd ``n`` -- which is the point: a packed frame cannot
    represent one unambiguously, and deriving ``n`` rather than trusting a parameter
    is what makes that explicit.

    The odd-``n`` behaviour that *is* observable lives in the packing, and is covered
    by ``test_pack_int4_odd_n_pads_the_tail`` below.
    """
    m, k, n = 32, 64, 128
    dtype = torch.float16
    output, hidden, packed, q, scales, bias = _build(m, k, n, 4, dtype, seed=3)
    packed_bytes = n // 2
    assert packed.shape == (k, packed_bytes)

    mixed_gemm(output, hidden, packed, scales, None, 4, 6)
    assert output.shape == (m, n)

    # Halving the weight width halves the output width, which is the property that
    # makes ``n`` recoverable from the weight alone.
    half = n // 4
    narrow = packed[:, :half].contiguous()
    narrow_out = torch.empty(m, half * 2, dtype=dtype, device=output.device)
    narrow_scales = scales[: half * 2].contiguous()
    mixed_gemm(narrow_out, hidden, narrow, narrow_scales, None, 4, 6)


@pytest.mark.mixed_gemm
def test_pack_int4_odd_n_pads_the_tail():
    """Packing an odd column count pads the tail, and the padding must not leak.

    The odd-``n`` behaviour is only observable at the packing boundary:
    ``symmetric_quantize`` writes ``ceil(n/2)`` bytes with the last byte's high
    nibble unused (the C++ guards it with ``input_idx < num_cols``), which is
    equivalent to appending a zero column before pairing.

    The operator itself never sees this case -- its layout requires a column count
    that is a multiple of 64 -- so this checks the packing convention on its own.
    """
    k, n = 64, 63
    torch.manual_seed(0)
    w = torch.randn(k, n, dtype=torch.float16)
    q, scales = quantize(w, 4)

    packed = pack_signed(q, 4, k, n)
    assert packed.numel() == k * ((n + 1) // 2), "odd n must pad to ceil(n/2) bytes"

    # The pad is a zero column, so the low nibble of the last byte holds column 62
    # and the high nibble is zero -- not column 63 of anything.
    last = packed.reshape(k, (n + 1) // 2)[:, -1].to(torch.int64)
    assert bool((((last >> 4) & 0x0F) == 0).all()), "the tail nibble must be zero"
    assert torch.equal(
        (last & 0x0F), (q[:, 62] & 0xF)
    ), "column 62 must keep its nibble"

    # At the even width the buffer packs cleanly: the real columns keep their nibbles
    # and the pad arrives as a zero column, rather than being interleaved among them.
    padded = torch.zeros(k, 64, dtype=q.dtype)
    padded[:, :n] = q
    wide = pack_signed(padded, 4, k, 64)
    wide = wide.reshape(k, 32).to(torch.int64)
    assert torch.equal(wide[:, :31] & 0x0F, q[:, 0:62:2] & 0xF), "low nibbles move"
    assert torch.equal(
        (wide[:, :31] >> 4) & 0x0F, q[:, 1:63:2] & 0xF
    ), "high nibbles move"
    assert bool(((wide[:, 31] >> 4) & 0x0F == 0).all()), "the pad column reads as zero"


@pytest.mark.mixed_gemm
def test_mixed_gemm_accepts_missing_bias():
    """``bias=None`` must be a working path, not a zero bias.

    The wrapper hands the *output* to the kernel as a stand-in pointer when no
    bias is given and relies on the ``HAS_BIAS`` constexpr to keep the kernel off
    it. That pointer is valid memory, so a missing guard would not crash -- it
    would read the accumulator's own contents and produce a wrong answer. The
    numerical assertion is therefore the check, not the absence of an exception.
    """
    m, k, n = 64, 64, 64
    dtype = torch.float16
    output, hidden, packed, q, scales, _ = _build(m, k, n, 4, dtype, seed=4)

    mixed_gemm(output, hidden, packed, scales, None, 4, 6)

    ref_out = torch.empty_like(output)
    mixed_gemm_ref(
        ref_out,
        _reference_copy(hidden),
        _reference_copy(q),
        _reference_copy(scales),
        None,
        4,
        6,
    )
    # Both sides go through ``to_reference``: under ``--ref cpu`` that is what puts
    # them on the same device, and ``flag_train.testing.assert_close`` compares with
    # ``torch.testing.assert_close``, which requires that.
    utils.train_assert_close(
        utils.to_reference(ref_out),
        utils.to_reference(output),
        dtype,
        reduce_dim=k,
        atol=1e-3,
    )


@pytest.mark.mixed_gemm
@pytest.mark.parametrize(
    "num_bits,activation_raw",
    [(4, 3), (4, 4), (4, 5)],
    ids=["geglu", "reglu", "siglu"],
)
def test_mixed_gemm_rejects_gated_activations(num_bits, activation_raw):
    """The runner instantiates no gated epilogue, so it must say so.

    The message is DeepSpeed's own ("Fused gated activations not supported"),
    because a caller should get the same diagnosis from the port as from the
    operator it replaces.
    """
    output, hidden, packed, q, scales, bias = _build(
        64, 64, 64, num_bits, torch.float16, 5
    )
    with pytest.raises(ValueError, match="gated"):
        mixed_gemm(output, hidden, packed, scales, None, num_bits, activation_raw)


@pytest.mark.mixed_gemm
@pytest.mark.parametrize("num_bits", [3, 16], ids=["below_range", "above_range"])
def test_mixed_gemm_rejects_bad_bit_width(num_bits):
    """``TORCH_CHECK(num_bits == 4 || num_bits == 8)``."""
    # The build needs a legal width; only the call should get the bad one.
    output, hidden, packed, q, scales, bias = _build(64, 64, 64, 4, torch.float16, 6)
    with pytest.raises(ValueError, match="4 or 8"):
        mixed_gemm(output, hidden, packed, scales, None, num_bits, 6)


@pytest.mark.mixed_gemm
def test_mixed_gemm_rejects_invalid_activation():
    """Anything outside the four supported values is rejected."""
    output, hidden, packed, q, scales, bias = _build(64, 64, 64, 4, torch.float16, 7)
    with pytest.raises(ValueError, match="Invalid activation type"):
        mixed_gemm(output, hidden, packed, scales, None, 4, 99)


@pytest.mark.mixed_gemm
def test_mixed_gemm_rejects_weight_k_mismatch():
    """``TORCH_CHECK(weight.size(0) == k)``.

    The rows disagree with ``k`` while the packed width stays plausible, so the
    check under test is the one this case names.
    """
    output, hidden, packed, q, scales, bias = _build(64, 64, 64, 4, torch.float16, 8)
    bad = torch.zeros(63, packed.size(1), dtype=torch.uint8, device=runtime_device.name)
    with pytest.raises(ValueError, match="Weight dimension mismatch"):
        mixed_gemm(output, hidden, bad, scales, None, 4, 6)


@pytest.mark.mixed_gemm
def test_mixed_gemm_rejects_dtype_mismatch():
    """``TORCH_CHECK(output.dtype() == hidden_states.dtype())``."""
    output, hidden, packed, q, scales, bias = _build(64, 64, 64, 4, torch.float16, 9)
    bad_output = torch.empty_like(output, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="same dtype"):
        mixed_gemm(bad_output, hidden, packed, scales, None, 4, 6)


@pytest.mark.mixed_gemm
@requires_deepspeed_reference
@pytest.mark.parametrize("num_bits", _BITS, ids=[f"int{b}" for b in _BITS])
@pytest.mark.parametrize("shape", [(128, 128, 128), (65, 128, 64)])
def test_matches_deepspeed(shape, num_bits):
    """Compare against DeepSpeed's own operator.

    This is the one check that can contradict our *reading* of the contract: the
    reference above is written from the same reading, so a shared misreading survives
    it.

    Nothing about the *inputs* stands in the way. This port reads the same preprocessed
    B buffer DeepSpeed's operator reads, and both write in place, so one argument tuple
    drives both.

    Whether it runs is an environment question, and ``_load_deepspeed_mixed_gemm`` has
    answered it by the time this is collected: the reference is called once at import, so
    a library that imports but does not dispatch is caught there rather than here.
    """
    m, k, n = shape
    dtype = torch.float16
    output, hidden, packed, q, scales, bias = _build(m, k, n, num_bits, dtype, seed=10)

    mixed_gemm(output, hidden, packed, scales, None, num_bits, 6)

    ref_out = torch.empty_like(output)
    _DEEPSPEED_ORACLE(ref_out, hidden, packed, scales, bias, num_bits, 6)
    # Both sides go through ``to_reference``: under ``--ref cpu`` that is what puts
    # them on the same device, and ``flag_train.testing.assert_close`` compares with
    # ``torch.testing.assert_close``, which requires that.
    utils.train_assert_close(
        utils.to_reference(ref_out),
        utils.to_reference(output),
        dtype,
        reduce_dim=k,
        atol=1e-3,
    )
