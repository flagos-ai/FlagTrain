# DeepSpeed FP8 GEMM — BF16 validation

Implementation: `src/flag_train/deepspeed/fp8_gemm_triton.py`, exported as
`flag_train.deepspeed.matmul_fp8_triton(x, weight, scale, quantization_group_size)`.
The kernel fuses DeepSpeed custom uint8 E4M3 decoding, scaling and GEMM with FP32
accumulation. Output has the activation dtype. It is forward-only. Weights are
not `torch.float8_e4m3fn`: byte zero represents `2**-7` before scaling; groups
index flattened `[K, N]` elements. Production code imports no DeepSpeed helpers.

## Current acceptance scope

This suite validates **BF16 only on NVIDIA**, following the LAMB component test
and `GenericBenchmark` structure. Existing FP16 entry points are not covered by
this acceptance result. Other backends, gradients and end-to-end training are
outside this report.

The baseline is the **unmodified** public API
`deepspeed.ops.fp_quantizer.matmul_fp8`. No replacement kernels, launch overrides,
scale padding or fallback baseline are installed. Missing dependencies are errors.

All core shapes use `K=4096`, `N=8192`, `quantization_group_size=128`:

| Origin | M | Treatment on the pinned environment |
| --- | --- | --- |
| Upstream, supported | 1, 2, 4, 8, 32, 64, 128 | Original baseline comparison and timing |
| Upstream, resource-limited | 256, 512, 1024, 2048 | Baseline skip; candidate still tested independently |
| Supplemental | 16, 17, 48, 96 | Original baseline comparison and timing |

Supplemental shapes were chosen before timing: M=16/17 exercises a tile-size
boundary, while M=48/96 exercises partial tiles. Shapes are not removed for
candidate correctness failures or speedups below the threshold.

## Narrow skip policy

Four original BF16 launches require 263168 bytes of shared memory per block,
exceeding the measured H20 limit of 232448 bytes. The skip requires all of:

- Exactly those four M values and the stated K/N/group size.
- GEMM source SHA256 `5c376c95cca24995283a13ca2a6ec7773cbad20aaa90e74e0ea7a18fe8826096`
  (observed at DeepSpeed commit `79caae1c04fca210345bdeb03bcacf87b1ac2f23`).
- Triton 3.6.x, H20 device name, and per-block shared-memory limit below 263168.

Other versions, shapes and errors do not inherit this skip. Numerical failures
are not skipped. These are 11 usable comparisons and four visible upstream skips,
not 15 baseline passes.

## Correctness

Every candidate core shape is checked against upstream's reference:
`torch.matmul(x, fpq.dequantize(weight, scale=fpq.scale))`. An independent
arithmetic E4M3 decoder must also match DeepSpeed's CUDA decoder. For usable
baseline shapes, original DeepSpeed is checked against this reference and the
candidate is compared directly to original DeepSpeed.

Repository `train_assert_close` ultimately uses `torch.testing.assert_close`:
BF16 rtol=0.016 and fixed atol=0.025. Core checks retain upstream's native-dtype
mean-relative-error formula and threshold `<0.004`, plus an FP32 relative mean
as an additional stable assertion. Neither check replaces a failed assertion.
There are also 25 candidate-only edge/oracle/invalid-input checks; these do not
claim that the original wrapper supports the extended input contracts.

The functional suite collects 55 nodes: 15 candidate core checks, 15 original
baseline checks and 25 edge/oracle/invalid-input checks. Expected outcomes on
the pinned environment are 51 passed and four skipped.

## Performance

Each core shape is a separate pytest node. NVIDIA kernel mode uses
`triton.testing.do_bench()` via `GenericBenchmark` with its configured warmup,
repetitions and median statistic. Both timed calls allocate their output and
consume identical inputs. Quantization, reference construction and correctness
checks occur outside timing. Every measured case must satisfy
`DeepSpeed_latency / FlagTrain_latency > 0.8`; a lower result fails, not skips.

The JSON retains all 15 pytest outcomes and the 11 measured rows; skipped cases
have no speedup. `--list-cases` lists all shapes without loading DeepSpeed.
`--case-id` supports an explicit known skip and still rejects unknown IDs.

## Commands

Install DeepSpeed with `DS_BUILD_FP_QUANTIZER=1 pip install -e .`, then FlagTrain
with `pip install -e .`, in the same environment. On the development server:

```bash
cd /workspace/fp8_gemm_dev/FlagTrain-nv-pr
source /workspace/fp8_gemm_dev/venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/src:$PWD"
python -m pytest tests/deepspeed/test_fp8_gemm.py -q -ra \
  --record=json --output=/tmp/fp8_bf16_accuracy.json
python -m pytest benchmark/deepspeed/test_fp8_gemm.py -q -s -ra --mode=kernel \
  --record=json --output=/tmp/fp8_bf16_benchmark.json
```

Use `python -m pytest` to avoid a system executable selecting another Python.
No baseline-selection environment variable is needed.

The development environment uses PyTorch 2.11.0a0+eb65b36914.nv26.02, CUDA 13.1,
Triton 3.6.0 and DeepSpeed 0.17.7+79caae1 on H20-3e. Two pre-existing external
DeepSpeed build adaptations permit Triton 3.6 and rename a CUDA translation unit
to avoid duplicate object names. They do not modify the GEMM kernel or launch
and are not part of this FlagTrain contribution.

## Measured acceptance (2026-10-14)

The contribution was integrated onto FlagTrain main commit
`e4abdf01b4490f5dc2522577922bc8ec50e0be2f` and revalidated through the component
public API. The top-level exports, shared test framework, CI and other operators
are unchanged. The development commands above explicitly select the PR worktree
through `PYTHONPATH`, without changing the original checkout's editable install.

On the environment above, the BF16 functional suite completed with **51 passed,
4 skipped**. The four skips affect only original-baseline comparisons: all 15
candidate core shapes passed against the torch reference. Performance completed
with **11 passed, 4 skipped**; every measured ratio exceeded 0.8x.

| M | Origin | DeepSpeed (ms) | FlagTrain (ms) | Ratio |
| --- | --- | ---: | ---: | ---: |
| 1 | Upstream | 0.051744 | 0.056608 | 0.914x |
| 2 | Upstream | 0.059712 | 0.056352 | 1.060x |
| 4 | Upstream | 0.059584 | 0.056608 | 1.053x |
| 8 | Upstream | 0.059520 | 0.056384 | 1.056x |
| 16 | Supplemental | 0.059808 | 0.056864 | 1.052x |
| 17 | Supplemental | 0.075968 | 0.060384 | 1.258x |
| 32 | Upstream | 0.075872 | 0.056672 | 1.339x |
| 48 | Supplemental | 0.072064 | 0.074688 | 0.965x |
| 64 | Upstream | 0.070592 | 0.074720 | 0.945x |
| 96 | Supplemental | 0.146784 | 0.089056 | 1.648x |
| 128 | Upstream | 0.147328 | 0.096704 | 1.523x |

Ratios below 1 mean slower than DeepSpeed, despite satisfying the requested
0.8x floor. These are measurements on one pinned environment, not a guarantee
for other devices or software versions.

A second complete run also produced 11 passed and four skipped, with ratios
from 0.917x to 1.641x (`benchmark_repeat.json`). Single-case selection was checked:
a supported ID records exactly one timing row; a known unsupported ID records
no timing rows and an explicit skip; an unknown ID fails with exit code 1.
Unselected pytest parameter nodes can report passed but do not count as timed
cases; use recorded timing rows, not that CLI run's total passed count.

The 25 edge/oracle/invalid-input checks also passed under Compute Sanitizer
memcheck with **0 errors**. LAMB and packbits functional regression:
**106 passed**. Black, isort, repository-configured flake8, component export
checks and `git diff --check` passed.

Raw logs and JSON are retained outside the repository at
`/workspace/fp8_gemm_dev/results/nv_submission_20261014/`, including
`accuracy.json`, `benchmark.json`, `memcheck.log` and `other_ops.log`.
Previous mixed-baseline results are not evidence for this BF16-only acceptance.
