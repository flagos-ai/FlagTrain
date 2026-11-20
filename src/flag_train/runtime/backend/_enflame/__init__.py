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

import importlib.util
import os
import re

from backend_utils import VendorDescriptor

# NOTE: transfer_to_gcu is not used anywhere
# try:
#     from torch_gcu import transfer_to_gcu  # noqa: F401
# except Exception:
#    logger.warning("torch_gcu not installed")

# TODO: Revise the following imports to be exception free
if importlib.util.find_spec("triton.backends.enflame") is None:
    from triton_gcu.triton.driver import _GCUDriver
else:
    from triton.backends.enflame.driver import _GCUDriver

driver = _GCUDriver()
arch = driver.get_arch()
arch_version = int(re.search(r"gcu(\d+)", arch).group(1))

# Ask the backend which fp8 dtypes the target supports rather than matching
# arch numbers here: GCUOptions is the compiler's own capability table, so a
# new target is picked up without editing this file. gcu300 aborts the process
# on unsupported fp8 casts instead of raising a catchable exception.
_gcu_compiler = importlib.import_module(
    _GCUDriver.__module__.rsplit(".", 1)[0] + ".compiler"
)
fp8_enabled = bool(_gcu_compiler.GCUOptions().supported_fp8_dtypes)

# int64 is supported everywhere except gcu300, where the topsaten path aborts
# the process instead of raising. Kept hardcoded until the backend exposes a
# capability query for it: GCUOptions.enable_i64 reflects the compile-time
# ENABLE_I64_CHECK setting, not what the target can do.
int64_enabled = arch_version != 300

# Complex tensors abort the process on gcu300: even creating one via
# torch.randn(dtype=complex64) reaches topsatenNormal with ComplexFloat and
# fails TOPSATEN_STATUS_NOT_SUPPORT, which is a C++ terminate rather than a
# catchable exception.
complex_enabled = arch_version != 300

vendor_info = VendorDescriptor(
    vendor_name="enflame",
    device_name="gcu",
    device_query_cmd="",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    int64_enabled=int64_enabled,
    fp8_enabled=fp8_enabled,
    complex_enabled=complex_enabled,
    tle_enabled=True,
)

os.environ["ARCH"] = str(arch_version)
ARCH_MAP = {"3": "gcu300", "4": "gcu400"}
# i64 to/copy is not supported in gcu300
if arch_version == 300:
    CUSTOMIZED_UNUSED_OPS = ()
elif arch_version == 400 or arch_version == 410:
    CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
