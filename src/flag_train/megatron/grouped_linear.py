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


"""Grouped linear: concat_g( inp[start_g:start_{g+1}] @ W[g].T + b[g] ).

One fused Triton kernel, one launch.  The grid is 3D
(row_tiles, col_tiles, G) so every program belongs to exactly one group; the
group's row range is obtained with scalar select chains (no vector gather, no
reduction, no dynamic loop over groups).
"""

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn

_HPAD = 4  # host split scalars carried in the kernel signature


@triton.jit
def _grouped_linear_kernel(
    inp_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    splits_ptr,
    T,
    K,
    N,
    s0,
    s1,
    s2,
    s3,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DEV_SPLITS: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)

    if DEV_SPLITS:
        # exclusive start and row count of group g from the device splits
        st = tl.zeros([], tl.int32)
        cnt = tl.zeros([], tl.int32)
        for gg in tl.static_range(0, G):
            c = tl.load(splits_ptr + gg).to(tl.int32)
            st = tl.where(g > gg, st + c, st)
            cnt = tl.where(g == gg, c, cnt)
    else:
        st = tl.where(g == 0, s0, tl.zeros([], tl.int32))
        st = tl.where(g == 1, s1, st)
        st = tl.where(g == 2, s2, st)
        st = tl.where(g == 3, s3, st)

        en = T
        en = tl.where(g == 0, s1, en)
        en = tl.where(g == 1, s2, en)
        en = tl.where(g == 2, s3, en)
        cnt = en - st

    if pid_m * BLOCK_M >= cnt:
        return

    local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = st.to(tl.int64) + local
    rmask = (local < cnt) & (rows >= 0) & (rows < T)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = cols < N
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    w_base = w_ptr + (g.to(tl.int64) * N + cols[:, None]) * K
    for kk in range(0, K, BLOCK_K):
        ks = kk + offs_k
        kmask = ks < K
        a = tl.load(
            inp_ptr + rows[:, None] * K + ks[None, :],
            mask=rmask[:, None] & kmask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_base + ks[None, :],
            mask=mask_n[:, None] & kmask[None, :],
            other=0.0,
        )
        if IS_FP32:
            ai = a.to(tl.int32, bitcast=True)
            ai = (ai + 4095 + ((ai >> 13) & 1)) & -8192
            a = ai.to(tl.float32, bitcast=True)
            wi = w.to(tl.int32, bitcast=True)
            wi = (wi + 4095 + ((wi >> 13) & 1)) & -8192
            w = wi.to(tl.float32, bitcast=True)
        acc = tl.dot(a, tl.trans(w), acc)

    if HAS_BIAS:
        bias = tl.load(b_ptr + g.to(tl.int64) * N + cols, mask=mask_n, other=0).to(
            tl.float32
        )
        acc = acc + bias[None, :]

    tl.store(
        out_ptr + rows[:, None] * N + cols[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=rmask[:, None] & mask_n[None, :],
    )


def _seq_from(m_splits):
    if hasattr(m_splits, "tolist"):
        return m_splits.tolist()
    return list(m_splits)


def _blocks(n_rows, N, K):
    bn = 1 << (N - 1).bit_length()
    if bn > 64:
        bn = 64
    elif bn < 16:
        bn = 16
    bk = 1 << (K - 1).bit_length() if K > 1 else 32
    if bk > 64:
        bk = 64
    elif bk < 32:
        bk = 32
    return 32, bn, bk


def grouped_linear(inp, m_splits, weights, biases=None):
    with torch_device_fn.device(inp.device):
        return _run_contiguous(
            inp.contiguous(),
            m_splits.contiguous() if torch.is_tensor(m_splits) else m_splits,
            weights.contiguous(),
            None if biases is None else biases.contiguous(),
        )


def _run_contiguous(inp, m_splits, weights, biases):
    shp = inp.shape
    wshp = weights.shape
    T = shp[0]
    N = wshp[1]
    out = inp.new_empty(T, N)
    if T == 0 or N == 0:
        return out
    K = shp[1]

    dt = inp.dtype
    has_bias = biases is not None
    b_arg = biases if has_bias else inp
    dev_splits = torch.is_tensor(m_splits) and m_splits.device == inp.device
    splits_arg = m_splits if dev_splits else inp
    if not dev_splits:
        hsplits = _seq_from(m_splits)
        G = len(hsplits)
        if G == 0:
            return out
        starts = []
        acc = 0
        for v in hsplits:
            starts.append(acc)
            acc += int(v)
        if G > _HPAD:
            return _run_chunked(
                inp, starts, weights, biases, out, T, K, N, dt is torch.float32
            )
        n_rows = 0
        for i in range(G):
            end = starts[i + 1] if i + 1 < G else T
            r = end - starts[i]
            if r > n_rows:
                n_rows = r
        if n_rows <= 0:
            return out
        vals = tuple(starts) + (T,) * (_HPAD - G)
    else:
        G = m_splits.numel()
        if G == 0:
            return out
        vals = (0,) * _HPAD
        n_rows = T

    is_fp32 = dt is torch.float32
    bm, bn, bk = _blocks(n_rows, N, K)
    grid = (-(-n_rows // bm), -(-N // bn), G)
    consts = vals + (bm, bn, bk, G, has_bias, dev_splits, is_fp32)
    _grouped_linear_kernel[grid](
        inp,
        weights,
        b_arg,
        out,
        splits_arg,
        T,
        K,
        N,
        *consts,
        num_warps=2 if bm * bn <= 2048 else 4,
        num_stages=3,
    )
    return out


def _run_chunked(inp, starts, weights, biases, out, T, K, N, is_fp32):
    """Fallback for more groups than the kernel's host scalar slots."""
    G = len(starts)
    has_bias = biases is not None
    g_off = 0
    while g_off < G:
        g1 = g_off + _HPAD
        if g1 > G:
            g1 = G
        base = starts[g_off]
        end = starts[g1] if g1 < G else T
        rows_n = end - base
        if rows_n > 0:
            sub = tuple(starts[i] - base for i in range(g_off, g1))
            nsub = g1 - g_off
            vals = sub + (rows_n,) * (_HPAD - nsub)
            bm, bn, bk = _blocks(rows_n, N, K)
            grid = (-(-rows_n // bm), -(-N // bn), nsub)
            _grouped_linear_kernel[grid](
                inp[base:end],
                weights[g_off:g1],
                biases[g_off:g1] if has_bias else inp,
                out[base:end],
                inp,
                rows_n,
                K,
                N,
                *vals,
                bm,
                bn,
                bk,
                nsub,
                has_bias,
                False,
                is_fp32,
                num_warps=2 if bm * bn <= 2048 else 4,
                num_stages=3,
            )
        g_off = g1
    return out
