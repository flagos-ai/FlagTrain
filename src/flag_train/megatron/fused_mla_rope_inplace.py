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


"""In-place Megatron MLA rotary embedding and its rotation gradient."""

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn


@triton.jit
def _mla_rope_inplace_kernel(
    t_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    cu_ptr,
    NOPE,
    HF: tl.constexpr,
    M: tl.constexpr,
    DIV,
    M0,
    M1,
    SH,
    COS_S0,
    SIN_S0,
    CP_RANK,
    CP_SIZE,
    NSEQ,
    SIN_SIGN,
    REMOVE: tl.constexpr,
    USE_POS: tl.constexpr,
    USE_CU: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    r = tl.program_id(0)
    h = tl.program_id(1)

    # row index -> (outer, inner) multipliers; packed layout uses DIV == 1
    s = r // DIV
    b = r - s * DIV
    base = s.to(tl.int64) * M0 + b.to(tl.int64) * M1 + h.to(tl.int64) * SH + NOPE

    # position row of the cos/sin table; position_ids is indexed by the row of
    # its own layout (dense: sequence index r // B == r // DIV, packed: r)
    if USE_POS:
        p = tl.load(pos_ptr + r).to(tl.int64)
    elif USE_CU:
        pos = r
        # ends[i] = cu_seqlens_q[i] // cp_size; locate the sequence holding pos
        seq = pos - pos
        for i in range(1, NSEQ):
            e = tl.load(cu_ptr + i).to(tl.int64) // CP_SIZE
            seq += tl.where(e <= pos, 1, 0)
        seq = tl.minimum(seq, NSEQ - 2)
        start = tl.load(cu_ptr + seq).to(tl.int64) // CP_SIZE
        end = tl.load(cu_ptr + seq + 1).to(tl.int64) // CP_SIZE
        local = pos - start
        length = end - start
        half = length // 2
        p = tl.where(
            local < half,
            local + (CP_RANK * length) // 2,
            local - half + ((2 * CP_SIZE - CP_RANK - 1) * length) // 2,
        )
        last = tl.load(cu_ptr + (NSEQ - 1)).to(tl.int64) // CP_SIZE
        p = tl.where(pos >= last, 0, p)
    else:
        p = s

    j = tl.arange(0, HF)
    kv = tl.arange(0, M)

    cl = tl.load(cos_ptr + p * COS_S0 + j).to(tl.float32)
    ch = tl.load(cos_ptr + p * COS_S0 + HF + j).to(tl.float32)
    sl = tl.load(sin_ptr + p * SIN_S0 + j).to(tl.float32) * SIN_SIGN
    sh = tl.load(sin_ptr + p * SIN_S0 + HF + j).to(tl.float32) * SIN_SIGN

    if BACKWARD and not REMOVE:
        x = tl.load(t_ptr + base + j).to(tl.float32)
        y = tl.load(t_ptr + base + HF + j).to(tl.float32)
    else:
        v = tl.load(t_ptr + base + kv)
        x, y = tl.split(tl.reshape(v, (HF, 2)))
        x, y = x.to(tl.float32), y.to(tl.float32)

    if BACKWARD:
        left = x * cl + y * sh
        right = -x * sl + y * ch
    else:
        left = x * cl - y * sl
        right = y * ch + x * sh

    if not REMOVE:
        # The old and new element orders overlap on the same storage.
        tl.debug_barrier()
    if BACKWARD or REMOVE:
        out = tl.reshape(tl.join(left, right), (M,))
        tl.store(t_ptr + base + kv, out.to(t_ptr.dtype.element_ty))
    else:
        tl.store(t_ptr + base + j, left.to(t_ptr.dtype.element_ty))
        tl.store(t_ptr + base + HF + j, right.to(t_ptr.dtype.element_ty))


def _rotate(
    t,
    cos,
    sin,
    nope_dim,
    emb_dim,
    cu_seqlens_q,
    cp_rank,
    cp_size,
    inverse,
    remove_interleaving,
    position_ids,
    backward=False,
):
    if not t.numel():
        return t
    if t.ndim == 4:
        seq, batch, heads, _ = t.shape
        rows = seq * batch
        seq_stride, batch_stride, head_stride, _ = t.stride()
    else:
        rows, heads, _ = t.shape
        batch = 1
        seq_stride, head_stride, _ = t.stride()
        batch_stride = 0

    use_pos = position_ids is not None
    use_cu = cu_seqlens_q is not None and not use_pos
    positions = position_ids.contiguous() if use_pos else t
    sequences = cu_seqlens_q.contiguous() if use_cu else t
    with torch_device_fn.device(t.device):
        _mla_rope_inplace_kernel[(rows, heads)](
            t,
            cos,
            sin,
            positions,
            sequences,
            nope_dim,
            emb_dim // 2,
            emb_dim,
            batch,
            seq_stride,
            batch_stride,
            head_stride,
            cos.stride(0),
            sin.stride(0),
            cp_rank,
            cp_size,
            sequences.numel() if use_cu else 1,
            -1.0 if inverse else 1.0,
            remove_interleaving,
            use_pos,
            use_cu,
            backward,
            num_warps=1,
        )
    return t


class _MLARope(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        inverse,
        remove_interleaving,
        position_ids,
    ):
        ctx.save_for_backward(cos, sin, position_ids)
        ctx.rotation = (
            nope_dim,
            emb_dim,
            cu_seqlens_q,
            cp_rank,
            cp_size,
            inverse,
            remove_interleaving,
        )
        return _rotate(t, cos, sin, *ctx.rotation, position_ids)

    @staticmethod
    def backward(ctx, grad):
        cos, sin, position_ids = ctx.saved_tensors
        grad = grad.clone(memory_format=torch.contiguous_format)
        return (
            _rotate(grad, cos, sin, *ctx.rotation, position_ids, backward=True),
            *([None] * 10),
        )


def fused_mla_rope_inplace(
    t,
    cos,
    sin,
    nope_dim,
    emb_dim,
    cu_seqlens_q=None,
    cp_rank=0,
    cp_size=1,
    rotary_interleaved=False,
    inverse=False,
    remove_interleaving=False,
    position_ids=None,
):
    if rotary_interleaved:
        raise ValueError("The native entry point requires rotary_interleaved=False")
    args = (
        t,
        cos,
        sin,
        nope_dim,
        emb_dim,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        inverse,
        remove_interleaving,
        position_ids,
    )
    if torch.is_grad_enabled() and t.requires_grad:
        return _MLARope.apply(*args)
    return _rotate(*args)
