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


"""Fused DSA indexer loss (forward) implemented with Triton kernels.

Two launches:
  * row kernel: per (batch, query-row block) computes the FP32 index scores, the
    top-k key gather in the reference's order (greater-than-kth keys by index,
    then equal-to-kth keys by index), replays the reference's fixed 32-wide
    post-topk sort so the int64 index output matches exactly, then evaluates the
    teacher attention distribution and the per-row KL divergence;
  * finalize kernel: sums the per-row partials, applies the valid-row gate,
    loss_coeff and the mean / per-token denominator.
"""

import torch
import triton
import triton.language as tl

_NEG_INF = tl.constexpr(float("-inf"))
_TINY = tl.constexpr(1e-10)


@triton.jit
def _dsa_row_kernel(
    q_ptr,
    w_ptr,
    k_ptr,
    tq_ptr,
    tk_ptr,
    mask_ptr,
    lse_ptr,
    vs_ptr,
    ve_ptr,
    kp_ptr,
    topk_ptr,
    part_ptr,
    vbuf,
    ibuf,
    Sq,
    Sk,
    Di,
    Dh,
    K,
    s_q_s,
    s_q_b,
    s_q_h,
    s_q_d,
    s_w_s,
    s_w_b,
    s_w_h,
    s_k_t,
    s_k_b,
    s_k_d,
    s_tq_s,
    s_tq_b,
    s_tq_h,
    s_tq_d,
    s_tk_t,
    s_tk_b,
    s_tk_h,
    s_tk_d,
    s_m_b,
    s_m_s,
    s_m_t,
    s_l_b,
    s_l_h,
    s_l_s,
    s_vs_s,
    s_ve_s,
    s_kp_t,
    s_o_b,
    s_o_s,
    s_o_t,
    scale,
    scale_ptr,
    I: tl.constexpr,
    H: tl.constexpr,
    PAD: tl.constexpr,
    LOGPAD: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_DI: tl.constexpr,
    BLOCK_DH: tl.constexpr,
    USE_RELU: tl.constexpr,
    HAS_MASK: tl.constexpr,
    VARLEN: tl.constexpr,
    HAS_KEYPOS: tl.constexpr,
    HAS_LSE: tl.constexpr,
    TK_HEAD: tl.constexpr,
    SPARSE: tl.constexpr,
    SCALE_IS_PTR: tl.constexpr,
):
    pid_s = tl.program_id(0)
    bidx = tl.program_id(1)

    if SCALE_IS_PTR:
        scale = tl.load(scale_ptr)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    ms = offs_s < Sq
    cols = tl.arange(0, BLOCK_T)
    mt = cols < Sk

    # ---------------- index scores ----------------
    score = tl.zeros([BLOCK_S, BLOCK_T], dtype=tl.float32)
    for h in tl.static_range(I):
        acc = tl.zeros([BLOCK_S, BLOCK_T], dtype=tl.float32)
        for d0 in range(0, Di, BLOCK_DI):
            offs_d = d0 + tl.arange(0, BLOCK_DI)
            md = offs_d < Di
            qv = tl.load(
                q_ptr
                + offs_s[:, None] * s_q_s
                + bidx * s_q_b
                + h * s_q_h
                + offs_d[None, :] * s_q_d,
                mask=ms[:, None] & md[None, :],
                other=0.0,
            ).to(tl.float32)
            kv = tl.load(
                k_ptr + cols[:, None] * s_k_t + bidx * s_k_b + offs_d[None, :] * s_k_d,
                mask=mt[:, None] & md[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(qv, tl.trans(kv), input_precision="ieee")
        if USE_RELU:
            acc = tl.maximum(acc, 0.0)
        wv = tl.load(
            w_ptr + offs_s * s_w_s + bidx * s_w_b + h * s_w_h, mask=ms, other=0.0
        ).to(tl.float32)
        score = score + acc * wv[:, None]

    # ---------------- additive teacher mask tile ----------------
    if HAS_MASK:
        mv = tl.load(
            mask_ptr + bidx * s_m_b + offs_s[:, None] * s_m_s + cols[None, :] * s_m_t,
            mask=ms[:, None] & mt[None, :],
            other=0.0,
        )
    else:
        mv = tl.zeros([BLOCK_S, BLOCK_T], dtype=tl.float32)

    # ---------------- key validity / top-k scores ----------------
    alltrue = tl.full([BLOCK_S, BLOCK_T], 1, dtype=tl.int1)
    if VARLEN:
        vstart = tl.load(vs_ptr + offs_s * s_vs_s, mask=ms, other=0)
        vend = tl.load(ve_ptr + offs_s * s_ve_s, mask=ms, other=0)
        if HAS_KEYPOS:
            kp = tl.load(kp_ptr + cols * s_kp_t, mask=mt, other=0)
        else:
            kp = cols
        vmask = (
            mt[None, :]
            & (kp[None, :] >= vstart[:, None])
            & (kp[None, :] < vend[:, None])
        )
        valid = alltrue & vmask
        score_t = tl.where(vmask, score, _NEG_INF)
    else:
        if HAS_MASK:
            valid = alltrue & mt[None, :] & (tl.abs(mv) < float("inf"))
            score_t = tl.where(valid, score + mv, _NEG_INF)
        else:
            # Megatron applies the implicit causal mask to KL, after selecting
            # the indices. An explicit additive mask also affects selection.
            valid = mt[None, :] & (cols[None, :] <= offs_s[:, None])
            score_t = score
    score_t = tl.where(mt[None, :], score_t, _NEG_INF)

    # ---------------- top-k gather in the reference's order ----------------
    rbase = bidx * (Sq * PAD) + offs_s * PAD
    lanes = tl.arange(0, PAD)
    off2 = rbase[:, None] + lanes[None, :]
    ok2 = ms[:, None] & (lanes[None, :] < K)

    cand = score_t
    kth = tl.full([BLOCK_S], _NEG_INF, dtype=tl.float32)
    for j in range(0, K):
        v, ij = tl.max(
            cand, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        cand = tl.where(cols[None, :] == ij[:, None], _NEG_INF, cand)
        kth = v
    gt = mt[None, :] & (score_t > kth[:, None])
    eq = mt[None, :] & (score_t == kth[:, None])
    igt = tl.where(gt, 1, 0)
    ngt = tl.sum(igt, axis=1)
    cg = tl.cumsum(igt, axis=1)
    ce = tl.cumsum(tl.where(eq, 1, 0), axis=1)
    gpos = tl.where(gt, cg - 1, ngt[:, None] + ce - 1)
    wr = (gt | eq) & (gpos < K) & ms[:, None]
    colvals = tl.broadcast_to(cols[None, :], [BLOCK_S, BLOCK_T])
    tl.store(ibuf + rbase[:, None] + gpos, colvals, mask=wr)
    tl.store(vbuf + rbase[:, None] + gpos, score_t, mask=wr)
    # unused slots are the "invalid" entries of the reference's sort network
    padl = ms[:, None] & (lanes[None, :] >= K)
    tl.store(ibuf + off2, tl.full([BLOCK_S, PAD], -2, dtype=tl.int32), mask=padl)
    tl.store(vbuf + off2, tl.full([BLOCK_S, PAD], _NEG_INF, tl.float32), mask=padl)
    sel = (gt | eq) & (gpos < K)

    # ---------------- replay the reference post-topk ordering ----------------
    gval = tl.load(vbuf + off2, mask=ms[:, None], other=_NEG_INF)
    gid = tl.load(ibuf + off2, mask=ms[:, None], other=-2)
    gvalid = gid != -2
    for i in tl.static_range(1, LOGPAD):
        for jj in tl.static_range(0, i):
            st = 1 << (i - 1 - jj)
            sz = 1 << i
            pidx = tl.broadcast_to((lanes ^ st)[None, :], [BLOCK_S, PAD])
            pv = tl.gather(gval, pidx, 1)
            pi = tl.gather(gid, pidx, 1)
            pva = tl.gather(tl.where(gvalid, 1, 0), pidx, 1) != 0
            low = ((lanes & st) == 0)[None, :]
            a_low = tl.where(low, gval, pv)
            a_high = tl.where(low, pv, gval)
            v_low = tl.where(low, gvalid, pva)
            v_high = tl.where(low, pva, gvalid)
            sw = ((a_low > a_high) & v_low) | (~v_high)
            doswap = tl.where(((lanes & sz) != 0)[None, :], sw, ~sw)
            gval = tl.where(doswap, pv, gval)
            gid = tl.where(doswap, pi, gid)
            gvalid = tl.where(doswap, pva, gvalid)
    for jj in tl.static_range(0, LOGPAD):
        st = 1 << (LOGPAD - 1 - jj)
        pidx = tl.broadcast_to((lanes ^ st)[None, :], [BLOCK_S, PAD])
        pv = tl.gather(gval, pidx, 1)
        pi = tl.gather(gid, pidx, 1)
        pva = tl.gather(tl.where(gvalid, 1, 0), pidx, 1) != 0
        low = ((lanes & st) == 0)[None, :]
        a_low = tl.where(low, gval, pv)
        a_high = tl.where(low, pv, gval)
        v_low = tl.where(low, gvalid, pva)
        v_high = tl.where(low, pva, gvalid)
        sw = ((a_low > a_high) & v_low) | (~v_high)
        doswap = ~sw
        gval = tl.where(doswap, pv, gval)
        gid = tl.where(doswap, pi, gid)
        gvalid = tl.where(doswap, pva, gvalid)
    tl.store(
        topk_ptr + bidx * s_o_b + offs_s[:, None] * s_o_s + lanes[None, :] * s_o_t,
        tl.where((gval == _NEG_INF) | (gid == -2), -1, gid).to(tl.int64),
        mask=ok2,
    )

    # ---------------- KL support ----------------
    if HAS_MASK:
        score_kl = score_t + mv
    else:
        score_kl = score_t
    if SPARSE:
        valid = valid & sel

    live = tl.max(tl.where(valid, 1.0, 0.0), axis=1) > 0.0

    s_safe = tl.where(valid, score_kl, _NEG_INF)
    mx = tl.max(s_safe, axis=1)
    mx_safe = tl.where(mx > _NEG_INF, mx, 0.0)
    zq = tl.sum(tl.exp(s_safe - mx_safe[:, None]), axis=1)
    logzq = mx_safe + tl.log(zq)
    index_logp = tl.where(valid, score_kl - logzq[:, None], 0.0)

    # ---------------- teacher probabilities ----------------
    praw = tl.zeros([BLOCK_S, BLOCK_T], dtype=tl.float32)
    if HAS_LSE:
        max_log_mass = tl.full([BLOCK_S], _NEG_INF, tl.float32)
    for h in tl.static_range(H):
        t = tl.zeros([BLOCK_S, BLOCK_T], dtype=tl.float32)
        for d0 in range(0, Dh, BLOCK_DH):
            offs_d = d0 + tl.arange(0, BLOCK_DH)
            md = offs_d < Dh
            qv = tl.load(
                tq_ptr
                + offs_s[:, None] * s_tq_s
                + bidx * s_tq_b
                + h * s_tq_h
                + offs_d[None, :] * s_tq_d,
                mask=ms[:, None] & md[None, :],
                other=0.0,
            ).to(tl.float32)
            if TK_HEAD:
                kv = tl.load(
                    tk_ptr
                    + cols[:, None] * s_tk_t
                    + bidx * s_tk_b
                    + h * s_tk_h
                    + offs_d[None, :] * s_tk_d,
                    mask=mt[:, None] & md[None, :],
                    other=0.0,
                ).to(tl.float32)
            else:
                kv = tl.load(
                    tk_ptr
                    + cols[:, None] * s_tk_t
                    + bidx * s_tk_b
                    + offs_d[None, :] * s_tk_d,
                    mask=mt[:, None] & md[None, :],
                    other=0.0,
                ).to(tl.float32)
            t += tl.dot(qv, tl.trans(kv), input_precision="ieee")
        t = t * scale
        if HAS_MASK and not VARLEN:
            t = t + mv
        t_safe = tl.where(valid, t, _NEG_INF)
        mrow = tl.max(t_safe, axis=1)
        mrow_safe = tl.where(mrow > _NEG_INF, mrow, 0.0)
        z = tl.sum(tl.exp(t_safe - mrow_safe[:, None]), axis=1)
        lse = mrow_safe + tl.log(z)
        lse_use = tl.where(live, lse, 0.0)
        ph = tl.exp(t_safe - lse_use[:, None]) * tl.where(valid, 1.0, 0.0)
        if HAS_LSE:
            ext = tl.load(
                lse_ptr + bidx * s_l_b + h * s_l_h + offs_s * s_l_s,
                mask=ms,
                other=0.0,
            )
            diff = ext - lse
            sp = tl.maximum(diff, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(diff)))
            # A common log shift across heads cancels in the final normalization.
            log_mass = tl.where(live, -sp, _NEG_INF)
            new_max = tl.maximum(max_log_mass, log_mass)
            shift = tl.where(new_max > _NEG_INF, new_max, 0.0)
            praw = praw * tl.exp(max_log_mass - shift)[:, None]
            ph = ph * tl.exp(log_mass - shift)[:, None]
            max_log_mass = new_max
        praw = praw + ph

    mass = tl.sum(praw, axis=1)
    if HAS_LSE:
        denom_p = tl.where(mass > 0.0, mass, 1.0)
    else:
        denom_p = tl.maximum(mass, _TINY)
    p = praw / denom_p[:, None]
    lp = tl.log(tl.maximum(p, _TINY))
    rows = tl.sum(p * (lp - index_logp), axis=1)
    tl.store(part_ptr + bidx * Sq + offs_s, rows, mask=ms)


@triton.jit
def _dsa_final_kernel(
    part_ptr,
    valid_ptr,
    out_ptr,
    R,
    rows_per_b,
    s_v_b,
    s_v_s,
    coeff,
    coeff_ptr,
    HAS_VALID: tl.constexpr,
    PER_TOKEN: tl.constexpr,
    COEFF_IS_PTR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if COEFF_IS_PTR:
        coeff = tl.load(coeff_ptr)
    acc = 0.0
    den = 0.0
    for off in range(0, R, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < R
        v = tl.load(part_ptr + idx, mask=m, other=0.0)
        if HAS_VALID:
            b = idx // rows_per_b
            s = idx % rows_per_b
            f = tl.load(valid_ptr + b * s_v_b + s * s_v_s, mask=m, other=0).to(
                tl.float32
            )
            v = v * f
            den += tl.sum(tl.where(m, f, 0.0))
        else:
            den += tl.sum(tl.where(m, 1.0, 0.0))
        acc += tl.sum(v)
    if PER_TOKEN:
        res = acc * coeff
    else:
        if HAS_VALID:
            den = tl.maximum(den, 1.0)
        res = acc * coeff / den
    tl.store(out_ptr, res)


def _map_strides(t, dims, orders):
    """Map tensor strides onto the axis order orders[0]."""
    shape = tuple(t.shape)
    for order in orders:
        if len(shape) != len(order):
            continue
        fine = True
        for ax, ch in enumerate(order):
            if shape[ax] != dims[ch] and shape[ax] != 1:
                fine = False
                break
        if fine:
            st = t.stride()
            table = {}
            for ax, ch in enumerate(order):
                table[ch] = st[ax]
            return tuple(table[ch] for ch in orders[0])
    return None


def _scalar_arg(x, device):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return 0.0, x.reshape(()), True
        return 0.0, torch.empty(1, device=device, dtype=torch.float32), False
    return float(x), torch.empty(1, device=device, dtype=torch.float32), False


def fused_dsa_indexer_loss(
    q,
    weights,
    k,
    query,
    key,
    softmax_scale,
    topk,
    loss_coeff,
    mask=None,
    sparse_loss=False,
    varlen_starts=None,
    varlen_ends=None,
    key_positions=None,
    query_valid_rows=None,
    calculate_per_token_loss=False,
    use_relu=True,
    non_compressed_lse=None,
):
    Sq, B, I, Di = q.shape
    Sk = k.shape[0]
    device = q.device

    # teacher query/key: the reference inserts a head axis for the 3-d form
    if query.dim() == 3:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
    _, _, H, Dh = query.shape
    bt_q = query.shape[1]
    s_tq_s, s_tq_b, s_tq_h, s_tq_d = query.stride()
    s_tk_t, s_tk_b, s_tk_h, s_tk_d = key.stride()
    tk_head = key.shape[2] != 1 or H == 1

    K = min(int(topk), Sk)
    dummy = torch.empty(1, device=device, dtype=torch.float32)

    # ---- additive teacher mask ----
    mask_ok = mask is not None
    if mask_ok:
        dims = {"b": B, "s": Sq, "t": Sk}
        got = _map_strides(mask, dims, ["bst", "bts", "sbt", "stb", "tbs", "tsb"])
        if got is not None:
            s_m_b, s_m_s, s_m_t = got
        else:
            got = _map_strides(mask, dims, ["st"])
            if got is not None:
                s_m_b, s_m_s, s_m_t = 0, got[0], got[1]
            else:
                got = _map_strides(mask, dims, ["bt"])
                if got is None:
                    raise ValueError("unsupported mask shape %s" % (tuple(mask.shape),))
                s_m_b, s_m_s, s_m_t = got[0], 0, got[1]
    else:
        s_m_b = s_m_s = s_m_t = 0

    # ---- varlen bounds (indexed by the query-row axis, as the reference does)
    varlen = varlen_starts is not None
    if varlen:
        s_vs_s = varlen_starts.stride(0) if varlen_starts.dim() > 0 else 0
        has_ends = varlen_ends is not None
        if has_ends:
            s_ve_s = varlen_ends.stride(0) if varlen_ends.dim() > 0 else 0
        else:
            s_ve_s = 0
            varlen_ends = varlen_starts
    else:
        varlen_starts = dummy
        varlen_ends = dummy
        s_vs_s = s_ve_s = 0
        has_ends = False

    # ---- key positions ----
    has_keypos = key_positions is not None
    if has_keypos:
        kp = key_positions.reshape(-1)
        s_kp_t = kp.stride(0)
    else:
        kp = dummy
        s_kp_t = 0

    # ---- valid query rows ----
    has_valid = query_valid_rows is not None
    if has_valid:
        vr = query_valid_rows
        if vr.numel() == B * Sq:
            vr = vr.reshape(B, Sq)
            s_v_b, s_v_s = vr.stride()
        elif vr.numel() == Sq:
            vr = vr.reshape(1, Sq)
            s_v_b, s_v_s = 0, vr.stride(1)
        else:
            vr = vr.reshape(1, 1)
            s_v_b, s_v_s = 0, 0
    else:
        vr = torch.empty(1, device=device, dtype=torch.int8)
        s_v_b, s_v_s = 0, 0

    # ---- optional external teacher normalizer ----
    has_lse = non_compressed_lse is not None
    if has_lse:
        dims = {"b": bt_q, "h": H, "s": Sq}
        got = _map_strides(
            non_compressed_lse, dims, ["bhs", "bsh", "sbh", "shb", "hbs", "hsb"]
        )
        if got is not None:
            s_l_b, s_l_h, s_l_s = got
        else:
            got = _map_strides(non_compressed_lse, dims, ["hs"])
            if got is not None:
                s_l_b, s_l_h, s_l_s = 0, got[0], got[1]
            else:
                got = _map_strides(non_compressed_lse, dims, ["s"])
                if got is None:
                    raise ValueError(
                        "unsupported non_compressed_lse shape %s"
                        % (tuple(non_compressed_lse.shape),)
                    )
                s_l_b, s_l_h, s_l_s = 0, 0, got[0]
    else:
        s_l_b = s_l_h = s_l_s = 0

    scale_v, scale_ptr, scale_is_ptr = _scalar_arg(softmax_scale, device)
    coeff_v, coeff_ptr, coeff_is_ptr = _scalar_arg(loss_coeff, device)

    topk_indices = torch.empty((B, Sq, K), device=device, dtype=torch.int64)
    part = torch.empty(B * Sq, device=device, dtype=torch.float32)
    out = torch.empty((), device=device, dtype=torch.float32)

    # the reference sorts the k gathered keys in a fixed 32-wide network
    PAD = max(32, triton.next_power_of_2(K))
    LOGPAD = PAD.bit_length() - 1
    vbuf = torch.empty(B * Sq * PAD, device=device, dtype=torch.float32)
    ibuf = torch.empty(B * Sq * PAD, device=device, dtype=torch.int32)

    BLOCK_S = 16
    BLOCK_T = max(16, triton.next_power_of_2(Sk))
    BLOCK_DI = max(16, triton.next_power_of_2(Di))
    BLOCK_DH = max(16, triton.next_power_of_2(Dh))

    grid = (triton.cdiv(Sq, BLOCK_S), B)
    _dsa_row_kernel[grid](
        q,
        weights,
        k,
        query,
        key,
        mask if mask_ok else dummy,
        non_compressed_lse if has_lse else dummy,
        varlen_starts,
        varlen_ends,
        kp,
        topk_indices,
        part,
        vbuf,
        ibuf,
        Sq,
        Sk,
        Di,
        Dh,
        K,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        s_tq_s,
        s_tq_b,
        s_tq_h,
        s_tq_d,
        s_tk_t,
        s_tk_b,
        s_tk_h,
        s_tk_d,
        s_m_b,
        s_m_s,
        s_m_t,
        s_l_b,
        s_l_h,
        s_l_s,
        s_vs_s,
        s_ve_s,
        s_kp_t,
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        scale_v,
        scale_ptr,
        I=I,
        H=H,
        PAD=PAD,
        LOGPAD=LOGPAD,
        BLOCK_S=BLOCK_S,
        BLOCK_T=BLOCK_T,
        BLOCK_DI=BLOCK_DI,
        BLOCK_DH=BLOCK_DH,
        USE_RELU=bool(use_relu),
        HAS_MASK=mask_ok,
        VARLEN=varlen,
        HAS_KEYPOS=has_keypos,
        HAS_LSE=has_lse,
        TK_HEAD=tk_head,
        SPARSE=bool(sparse_loss),
        SCALE_IS_PTR=scale_is_ptr,
        num_warps=4,
        num_stages=1,
    )

    _dsa_final_kernel[(1,)](
        part,
        vr,
        out,
        B * Sq,
        Sq,
        s_v_b,
        s_v_s,
        coeff_v,
        coeff_ptr,
        HAS_VALID=has_valid,
        PER_TOKEN=bool(calculate_per_token_loss),
        COEFF_IS_PTR=coeff_is_ptr,
        BLOCK=1024,
        num_warps=4,
        num_stages=1,
    )
    return topk_indices, out
