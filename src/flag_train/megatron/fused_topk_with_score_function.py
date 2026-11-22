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


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BIG = tl.constexpr(1073741824)
_EPS = tl.constexpr(1e-20)

_PLANS = {}

_SF_IDS = {
    "softmax": 0,
    "0": 0,
    "sigmoid": 1,
    "1": 1,
    "sqrtsoftplus": 2,
    "2": 2,
}


@triton.jit
def _tie_priority(index, WIDTH: tl.constexpr, BIT_REVERSED: tl.constexpr):
    if BIT_REVERSED:
        lane = index & 31
        rev = (
            ((lane & 1) << 4)
            | ((lane & 2) << 2)
            | (lane & 4)
            | ((lane & 8) >> 2)
            | ((lane & 16) >> 4)
        )
        return rev * ((WIDTH + 31) // 32) + (index >> 5)
    return index


@triton.jit
def _router_kernel(
    logits_ptr,
    bias_ptr,
    probs_ptr,
    map_ptr,
    num_tokens,
    num_experts,
    score_fn: tl.constexpr,
    has_bias: tl.constexpr,
    is_bitmap: tl.constexpr,
    bitrev: tl.constexpr,
    pre_sm: tl.constexpr,
    num_groups: tl.constexpr,
    group_topk: tl.constexpr,
    block_g: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    e_off = tl.arange(0, BLOCK_E)
    rmask = rows < num_tokens
    emask = e_off < num_experts
    mask = rmask[:, None] & emask[None, :]
    offs = rows[:, None].to(tl.int64) * num_experts + e_off[None, :]

    neg_inf = -float("inf")
    x = tl.load(logits_ptr + offs, mask=mask, other=neg_inf).to(tl.float32)

    # ---- selection priority for equal scores ----
    # native naive path: bit-reversed warp lane order, then index within the lane
    # native radix path: ascending indices for boundary ties
    prio = _tie_priority(e_off, BLOCK_E, bitrev)

    # ---- score computation (fp32) ----
    if score_fn == 0:
        if pre_sm:
            mx = tl.max(x, 1)
            ex = tl.exp(x - mx[:, None])
            s = ex / tl.sum(ex, 1)[:, None]
        else:
            s = x
        key = s
    elif score_fn == 1:
        s = tl.sigmoid(x)
        key = s
    else:
        sp = tl.maximum(x, 0.0) + libdevice.log1p(tl.exp(-tl.abs(x)))
        s = tl.sqrt(sp)
        key = s

    if has_bias:
        b = tl.load(bias_ptr + e_off, mask=emask, other=0.0).to(tl.float32)
        key = key + b[None, :]
        if score_fn == 2:
            s = key - b[None, :]

    key = tl.where(emask[None, :], key, neg_inf)

    # ---- optional grouped routing ----
    if num_groups > 0 and group_topk > 0:
        kk: tl.constexpr = topk // group_topk
        g_off = tl.arange(0, block_g)
        g_prio = _tie_priority(g_off, block_g, bitrev)
        ge = e_off // (num_experts // num_groups)
        gs = tl.zeros((BLOCK_T, block_g), tl.float32)
        for g in range(num_groups):
            gval = tl.where(ge[None, :] == g, key, neg_inf)
            add = tl.zeros((BLOCK_T,), tl.float32)
            for _ in range(kk):
                mv = tl.max(gval, 1)
                gi = tl.min(tl.where(gval == mv[:, None], prio[None, :], _BIG), 1)
                add = add + mv
                gval = tl.where(prio[None, :] == gi[:, None], neg_inf, gval)
            gs = gs + tl.where(g_off[None, :] == g, add[:, None], 0.0)

        gsel = tl.where(g_off[None, :] < num_groups, gs, neg_inf)
        keep_e = tl.zeros((BLOCK_T, BLOCK_E), tl.int1)
        for _ in range(group_topk):
            mv = tl.max(gsel, 1)
            gi = tl.min(tl.where(gsel == mv[:, None], g_prio[None, :], _BIG), 1)
            hit = g_prio[None, :] == gi[:, None]
            selected_group = tl.min(tl.where(hit, g_off[None, :], _BIG), 1)
            keep_e = keep_e | (ge[None, :] == selected_group[:, None])
            gsel = tl.where(hit, neg_inf, gsel)
        key = tl.where(keep_e, key, neg_inf)

    # ---- top-k selection (max then native tie priority) ----
    sel = tl.zeros((BLOCK_T, BLOCK_E), tl.int1)
    for _ in range(topk):
        mv = tl.max(key, 1)
        idx = tl.min(tl.where(key == mv[:, None], prio[None, :], _BIG), 1)
        hit = (prio[None, :] == idx[:, None]) & emask[None, :]
        sel = sel | hit
        key = tl.where(hit, neg_inf, key)

    # ---- probability values ----
    if score_fn == 0:
        if pre_sm:
            val = tl.where(sel, s, 0.0) * SCALE
        else:
            mI = tl.max(tl.where(sel, x, neg_inf), 1)
            exx = tl.where(sel, tl.exp(x - mI[:, None]), 0.0)
            sI = tl.sum(exx, 1)
            val = exx / sI[:, None] * SCALE
    else:
        if topk > 1:
            n = tl.sum(tl.where(sel, s, 0.0), 1)
            val = tl.where(sel, s / (n[:, None] + _EPS), 0.0) * SCALE
        else:
            val = tl.where(sel, s, 0.0) * SCALE

    out_ty = probs_ptr.dtype.element_ty
    tl.store(probs_ptr + offs, val.to(out_ty), mask=mask)

    # ---- routing map ----
    if is_bitmap:
        nb = (num_experts + 7) // 8
        sel3 = tl.reshape(sel, (BLOCK_T, BLOCK_E // 8, 8))
        bits = (1 << tl.arange(0, 8))[None, None, :]
        packed = tl.sum(tl.where(sel3, bits, 0), 2)
        bidx = tl.arange(0, BLOCK_E // 8)
        bmask = rmask[:, None] & (bidx[None, :] < nb)
        tl.store(
            map_ptr + rows[:, None].to(tl.int64) * nb + bidx[None, :],
            packed.to(tl.uint8),
            mask=bmask,
        )
    else:
        tl.store(map_ptr + offs, sel, mask=mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _score_id(score_function):
    if score_function is None:
        return 0
    sf = _SF_IDS.get(score_function)
    if sf is not None:
        return sf
    name = (
        str(score_function).lower().replace("_", "").replace("-", "").replace(" ", "")
    )
    sf = _SF_IDS.get(name)
    if sf is None:
        raise ValueError(f"unsupported score_function: {score_function}")
    return sf


def _build_plan(num_experts, groups):
    block_e = max(8, _next_pow2(num_experts))
    if block_e <= 16:
        block_t = 8
    elif block_e <= 64:
        block_t = 4
    elif block_e <= 256:
        block_t = 2
    else:
        block_t = 1

    return block_t, block_e, _next_pow2(groups), (num_experts + 7) // 8


def fused_topk_with_score_function(
    logits,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    expert_bias,
    routing_map_format="bytemap",
):
    shape = logits.shape
    if len(shape) == 2:
        x = logits
        num_tokens = shape[0]
        num_experts = shape[1]
    else:
        num_experts = shape[-1]
        x = logits.reshape(-1, num_experts)
        num_tokens = x.shape[0]

    has_bias = expert_bias is not None
    route_format = getattr(routing_map_format, "value", routing_map_format)
    names = {None: False, 0: False, 1: True, "bytemap": False, "bitmap_u8": True}
    if route_format not in names:
        raise ValueError(f"Unsupported routing map format: {routing_map_format}")
    is_bitmap = names[route_format]
    sf = _score_id(score_function)
    if has_bias and (
        sf == 0
        or expert_bias.shape != (num_experts,)
        or expert_bias.device != logits.device
        or expert_bias.dtype != torch.float32
        or not expert_bias.is_contiguous()
    ):
        raise ValueError(
            "Expert bias requires sigmoid or sqrtsoftplus and a contiguous FP32 vector on the input device"
        )
    groups = int(num_groups or 0)
    selected_groups = int(group_topk or 0)
    k = int(topk)
    if not 0 < k <= num_experts:
        raise ValueError("topk must be positive and cannot exceed num_experts")
    if groups or selected_groups:
        if not (
            0 < selected_groups <= groups <= num_experts
            and num_experts % groups == 0
            and k % selected_groups == 0
            and k <= (num_experts // groups) * selected_groups
        ):
            raise ValueError("Invalid expert grouping and group topk")
    ckey = (num_experts, groups)
    plan = _PLANS.get(ckey)
    if plan is None:
        plan = _build_plan(num_experts, groups)
        _PLANS[ckey] = plan
    block_t, block_e, block_g, nb = plan

    scale = 1.0 if scaling_factor is None else float(scaling_factor)
    device = x.device

    probs = torch.empty_like(x)
    if is_bitmap:
        rmap = torch.empty((num_tokens, nb), dtype=torch.uint8, device=device)
    else:
        rmap = torch.empty((num_tokens, num_experts), dtype=torch.bool, device=device)

    grid = ((num_tokens + block_t - 1) // block_t,)
    if num_tokens:
        with torch.cuda.device(device):
            _router_kernel[grid](
                x,
                expert_bias if has_bias else x,
                probs,
                rmap,
                num_tokens,
                num_experts,
                sf,
                has_bias,
                is_bitmap,
                k < 16 or num_experts > 8160,
                bool(use_pre_softmax),
                groups,
                selected_groups,
                block_g,
                k,
                block_t,
                block_e,
                scale,
            )

    if len(shape) != 2:
        probs = probs.view(shape)
        rmap = rmap.view(shape[:-1] + (nb,)) if is_bitmap else rmap.view(shape)
    return probs, rmap
