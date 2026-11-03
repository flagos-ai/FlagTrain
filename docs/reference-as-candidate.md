# Reference as candidate (`--ref-as-candidate`)

`--ref-as-candidate` is an opt-in validation mode. It replaces the candidate
implementation with the explicit reference that the test or benchmark has
already declared, so correctness and timing exercises the reference contract
itself instead of silently measuring the candidate.

The flag is off by default and changes nothing unless it is passed.

## Correctness tests

Call the shared helper with an exact public operator path and the reference
callable the test already trusts:

```python
from flag_train.reference_candidate import get_candidate

op = get_candidate("flag_train.megatron.fast_gelu", reference=fast_gelu_ref)
y = op(x)
```

Accepted paths are `flag_train.<component>.<name>` and
`<component>.<name>`, where `<component>` is `megatron`, `deepspeed`, or
`transformer_engine`. Bare operator names are rejected; there is no second
name-discovery policy.

* Without the flag, `get_candidate` returns the live public attribute, so CLI
  overrides installed on `flag_train` are observed.
* With the flag, the reference is selected before any operator lookup. The
  candidate export does not need to exist.
* `reference` must be callable and must match the candidate's signature,
  inputs, outputs, and mutation contract. Device or module-specific adapters
  belong in the calling test, not in the helper.
* A missing or non-callable reference fails the selected test with
  `ReferenceContractError: ... No predefined reference ...`.
* A test that does not call `get_candidate` fails in this mode instead of
  running the ordinary candidate. Reference calls made only by fixtures or by
  another test case do not count.

Run only the tests that have opted in:

```sh
pytest tests -m "<operator>" --ref-as-candidate
```

## Benchmarks

The benchmark timing path uses the benchmark's explicit `torch_op` as the
candidate-side reference. It does not fall back to `train_op` or `use_train`,
and it does not resolve an eager candidate. The native-baseline timing is
unchanged and is not counted as candidate execution.

```sh
pytest benchmark -m "<operator>" --level core --ref-as-candidate --record json
```

Each measured workload must actually invoke the selected reference; otherwise
the test fails. Cases with no selected workload are reported as not selected,
not as successful reference timings. JSON/log records produced by this mode
include `candidate_source: "reference"` so the timings are not mistaken for
optimized candidate performance.

## Rejected combinations

`--ref-as-candidate` is a focused validation mode, so it rejects options that
change what is being validated: operator overrides, dummy/report-only modes,
case listing, query, collect-only/collect-marks, and parallel/xdist execution.
A clear `pytest.UsageError` is emitted. Ordinary test/case selection and
`-k`-style node filtering remain available.
