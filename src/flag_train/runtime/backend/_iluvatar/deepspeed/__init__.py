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
"""Iluvatar ``deepspeed`` operators.

``blocked_flash`` is here because the generic launcher rejects this device
outright, sizes its tile for more shared memory than the BI-V150 has, and
compiles its two-loop KV walk into wrong numbers here; the module's docstring
has all three.

``mixed_gemm`` is specialised here: the generic fused kernel cannot be pipelined
on this backend's Triton and, unpipelined, stays 3-4x behind the vendor BLAS past
L2.
"""
