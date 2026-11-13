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
"""MUSA implementation of the FP6-weight FP16-activation linear operator."""

import logging

import torch
import triton

import flag_train
from flag_train.deepspeed.cuda_wf6af16_linear import (
    _cuda_wf6af16_linear_gemm,
    _gemm_configs,
    _pick_split_k,
    cuda_wf6af16_split_k_reduce_kernel,
)
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

# Multi-stage uint32 loads are lowered to async global-to-shared copies on MUSA
# and can fault with a misaligned address during the split-K decode case.
_configs = _gemm_configs()
for config in _configs:
    config.num_stages = 1

cuda_wf6af16_linear_kernel = libentry()(
    triton.autotune(
        configs=_configs,
        key=["M", "N", "K", "split_k"],
    )(_cuda_wf6af16_linear_gemm)
)


def cuda_wf6af16_linear(
    output,
    hidden_states,
    weights_2bit,
    weights_4bit,
    scale,
    out_channels=None,
    tokens=None,
    in_channels=None,
    split_k=None,
    wide_load=None,
):
    """Run FP6 linear with the MUSA autotune configurations."""
    logger.debug("TRAIN_MTHREADS CUDA_WF6AF16_LINEAR")

    assert hidden_states.dtype == torch.float16, "activations must be fp16"
    assert output.dtype == torch.float16, "output must be fp16"
    assert scale.dtype == torch.float16, "scales must be fp16"
    assert weights_2bit.dtype == torch.uint8 and weights_4bit.dtype == torch.uint8
    assert (
        hidden_states.device.type == flag_train.device
    ), f"cuda_wf6af16_linear only supports {flag_train.device} tensors"

    tokens = tokens if tokens is not None else hidden_states.shape[0]
    in_channels = in_channels if in_channels is not None else hidden_states.shape[1]
    out_channels = out_channels if out_channels is not None else scale.shape[0]

    assert hidden_states.shape == (tokens, in_channels)
    assert output.shape == (tokens, out_channels)
    assert in_channels % 4 == 0, "in_channels must be a multiple of 4"
    assert weights_2bit.shape == (out_channels, in_channels // 4)
    assert weights_4bit.shape == (out_channels, in_channels // 2)
    # The kernel indexes raw pointers with row-major strides.
    for t in (output, hidden_states, weights_2bit, weights_4bit, scale):
        assert t.is_contiguous(), "cuda_wf6af16_linear requires contiguous tensors"

    if split_k is None:
        split_k = _pick_split_k(tokens, out_channels, in_channels)
    elif in_channels % (split_k * 4) != 0:
        # Each slice must start on a byte boundary (4 codes per byte).
        logger.debug(
            "cuda_wf6af16_linear: split_k=%d misaligns K=%d slices, using split_k=1",
            split_k,
            in_channels,
        )
        split_k = 1

    # The word-wide weight loads need every row and every split-K slice to
    # start on a 4-byte boundary of both packed slices: K % 16 == 0 keeps the
    # rows aligned, and k_per_split % 16 == 0 keeps the slices aligned. They
    # only pay on the decode path's narrow tiles; on the wide prefill tiles
    # they measure neutral to clearly worse, so the gate also requires the
    # decode token range.
    k_per_split = in_channels // split_k
    if wide_load is None:
        wide_load = (
            tokens <= 64
            and in_channels % 16 == 0
            and k_per_split % 16 == 0
            and weights_2bit.data_ptr() % 4 == 0
            and weights_4bit.data_ptr() % 4 == 0
        )

    def grid(meta):
        return (
            triton.cdiv(out_channels, meta["BLOCK_M"])
            * triton.cdiv(tokens, meta["BLOCK_N"])
            * split_k,
        )

    with torch_device_fn.device(hidden_states.device):
        if split_k == 1:
            cuda_wf6af16_linear_kernel[grid](
                output,
                hidden_states,
                weights_2bit,
                weights_4bit,
                scale,
                out_channels,
                tokens,
                in_channels,
                split_k,
                SPLIT_K_MODE=False,
                WIDE_LOAD=wide_load,
            )
        else:
            workspace = torch.empty(
                (split_k, tokens, out_channels),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            cuda_wf6af16_linear_kernel[grid](
                workspace,
                hidden_states,
                weights_2bit,
                weights_4bit,
                scale,
                out_channels,
                tokens,
                in_channels,
                split_k,
                SPLIT_K_MODE=True,
                WIDE_LOAD=wide_load,
            )
            total = tokens * out_channels
            cuda_wf6af16_split_k_reduce_kernel[(triton.cdiv(total, 1024),)](
                workspace,
                output,
                total,
                split_k,
                BLOCK_SIZE=1024,
            )

    return output
