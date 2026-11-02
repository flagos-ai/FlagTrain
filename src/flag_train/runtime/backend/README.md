## Multiple backend adaptations
### Introduction
The `flag_train` operator library provides the ability to access multiple backends.  If you are a chip vendor and wish to integrate  `your flag_train code` into our official main branch, you simply need to follow these steps to complete the process.

#### step 1:
Create a folder named after your vendor in the  `FlagTrain/src/flag_train/runtime/backend directory`, following the pattern `_vendorname`. For example, you can refer to the structure of  `FlagTrain/src/flag_train/runtime/backend/_nvidia`.

#### step 2:
Create the necessary files, including but not limited to `__init__.py`, `heuristics_config_utils.py`, `tune_configs.yaml`, as well as the operator packages described in step 2.4. This is an example under _nvidia file:
```
├── __init__.py
├── deepspeed
│   ├── __init__.py
│   └── blocked_flash.py
├── heuristics_config_utils.py
├── megatron
│   └── __init__.py
├── transformer_engine
│   └── __init__.py
└── tune_configs.yaml
```

##### step 2.1  `__init__.py`

You can copy `FlagTrain/src/flag_train/runtime/backend/_nvidia/__init__.py` and the  ***only change***  you need to make is to configure the `VendorDescriptor` class
```
VendorDescriptor(
    vendor_name="xxx", device_name="xxx", device_query_cmd="xxx"
)
```
###### Necessary fields:
- `vendor_name` is your vendorname, like `nvidia`

- `device_name` is your devicename, like `cuda`

- `device_query_cmd` is a command that can only be successfully executed on your vendor's device, like `nvidia-smi`

###### Optional fields:
- `dispatch_key` is The operator registration field of torch.library.Library in pytorch, like `PrivateUse1`

##### step 2.2  `heuristics_config_utils.py`

You should configure  `triton.heuristics` params in `FlagTrain/src/flag_train/runtime/backend/_vendorname/heuristics_config_utils.py`and you can  refer to `FlagTrain/src/flag_train/runtime/backend/_nvidia/heuristics_config_utils.py`

##### step 2.3  `tune_configs.yaml`
You should configure  `triton.autotune` params in `FlagTrain/src/flag_train/runtime/backend/_vendorname/tune_configs.yaml` and you can refer to `FlagTrain/src/flag_train/runtime/backend/_nvidia/tune_configs.yaml`

##### step 2.4  operator packages (`deepspeed`, `megatron`, `transformer_engine`)
`vendor-customized operator` implementations live in the package that mirrors the
generic `flag_train` package they replace: an override for
`flag_train.deepspeed.blocked_flash` is `deepspeed/blocked_flash.py` here, one for
`flag_train.megatron.fused_gated_norm` is `megatron/fused_gated_norm.py`, and so
on. Re-export each operator from its package's `__init__.py` and list it in that
package's `__all__`; that is what `SpecOpRegistrar` registers, and a name that is
not listed keeps the generic implementation.

For instance, to customise `flag_train.deepspeed.lamb`, place the implementation in
`deepspeed/lamb.py` and configure `deepspeed/__init__.py` accordingly:
```python
from .blocked_flash import blocked_flash
from .lamb import lamb

__all__ = ["blocked_flash", "lamb"]
```
