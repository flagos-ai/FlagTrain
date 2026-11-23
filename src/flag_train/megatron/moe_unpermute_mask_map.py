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


"""TransformerEngine mask-map unpermute for ordinary floating tensors."""

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn


@triton.jit
def _unpermute_kernel(
    inp_ptr,
    rid_ptr,
    prob_ptr,
    pad_ptr,
    out_ptr,
    E: tl.constexpr,
    H: tl.constexpr,
    rid_s0,
    rid_s1,
    inp_s0,
    inp_s1,
    prob_s0,
    prob_s1,
    out_s0,
    out_s1,
    num_hb,
    HAS_PROBS: tl.constexpr,
    HAS_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_h = pid % num_hb
    t = pid // num_hb
    t64 = t.to(tl.int64)

    row = rid_ptr + t64 * rid_s0
    c = tl.load(row + (2 * E) * rid_s1)
    c = tl.minimum(tl.maximum(c, 0), E)

    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for j in range(0, c):
        a = tl.load(row + j * rid_s1)
        e = tl.load(row + (E + j) * rid_s1)
        if HAS_PAD:
            a = a + tl.load(pad_ptr + e)
        v = tl.load(
            inp_ptr + a.to(tl.int64) * inp_s0 + h * inp_s1, mask=hmask, other=0.0
        )
        if HAS_PROBS:
            w = tl.load(prob_ptr + t64 * prob_s0 + e.to(tl.int64) * prob_s1)
            acc += v.to(tl.float32) * w.to(tl.float32)
        else:
            acc += v.to(tl.float32)

    tl.store(
        out_ptr + t64 * out_s0 + h * out_s1,
        acc.to(out_ptr.dtype.element_ty),
        mask=hmask,
    )


def moe_unpermute_mask_map(
    inp, row_id_map, merging_probs, num_tokens, num_experts, hidden_size, pad_offsets
):
    if not inp.numel():
        return inp.clone()
    out = inp.new_empty((num_tokens, hidden_size))
    if not out.numel():
        return out

    has_probs = merging_probs is not None
    has_pad = pad_offsets is not None
    prob = merging_probs if has_probs else out
    pad = pad_offsets.contiguous() if has_pad else out
    block_h = max(64, min(1024, triton.next_power_of_2(hidden_size)))
    num_hb = triton.cdiv(hidden_size, block_h)
    with torch_device_fn.device(inp.device):
        _unpermute_kernel[(num_tokens * num_hb,)](
            inp,
            row_id_map,
            prob,
            pad,
            out,
            num_experts,
            hidden_size,
            row_id_map.stride(0),
            row_id_map.stride(1),
            inp.stride(0),
            inp.stride(1),
            prob.stride(0) if has_probs else 0,
            prob.stride(1) if has_probs else 0,
            out.stride(0),
            out.stride(1),
            num_hb,
            HAS_PROBS=has_probs,
            HAS_PAD=has_pad,
            BLOCK_H=block_h,
            num_warps=4,
        )
    return out
