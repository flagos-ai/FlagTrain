# 移植 FlagGems-vllm 的 `--base-data` 与加速比计算（与上游方案保持一致）

状态：计划（未实施）
基线输入：`/root/wuhan/flagos-ops-baseline/op_flagtrain_lamb_h800_baseline.json`
目标用例：`benchmark/deepspeed/test_lamb.py`（`lamb`，7 个 shape，`torch.float32`）

> 说明：FlagTrain 之前的 `speedup/` 目录是临时实现，已移除。本计划不再依赖
> 该目录；代码落点、字段名和计算语义全部对齐 FlagGems-vllm 的实现。

---

## 1. 目标

让 FlagTrain 的 benchmark 支持与 FlagGems-vllm 完全相同的用法：

```bash
pytest benchmark/deepspeed/test_lamb.py -m lamb --level core \
  --record json --output benchmark_lamb.json \
  --continue-on-collection-errors \
  --base-data=/root/wuhan/flagos-ops-baseline/op_flagtrain_lamb_h800_baseline.json
```

产出：

1. 每个 shape 的 `speedup`（本机 torch baseline / FlagTrain 实现，已有）；
2. 每个 shape 的 `compared_speedup`（vs NVIDIA 基线，按芯片峰值归一化）；
3. 控制台的 “vs Base” 列与 `[vs Base]` 日志；
4. JSON 里的 `compared_speedup` / `base_miss_reason`；
5. 汇总工具输出的 dtype/op 级加速比（统计口径与 FlagGems-vllm 一致）。

---

## 2. 对齐原则

| 维度 | FlagGems-vllm | FlagTrain 本次落地 |
|---|---|---|
| 选项名 | `--base-data=<path>` | 同名、同 `=` 形式 |
| 加载失败 | 打印提示，返回 `None`，benchmark 照常跑 | 完全相同 |
| 字段 | `BenchmarkMetrics.compared_speedup`、`base_miss_reason` | 同名；`base_miss_reason` 新增 |
| 查找 | `lookup_base_record_ex` / `lookup_base_record` | 同名、同签名、同回退顺序 |
| 归一化 | `lookup_scaling_factor` | 同名、同规则（measured > nominal、vendor+chip） |
| 计算位置 | `benchmark/base.py` 的测量循环内 | FlagTrain 的 `_run_legacy` / `_run_cases` 内 |
| 公式 | `compared_speedup = (base_ms/latency) / factor` | 完全相同 |
| 展示 | `BenchmarkResult` 的 “vs Base” 列 | 完全相同 |
| 统计 | `run_tests.py::parse_perf_data` + `psum_text::parse_lines` | 移植到 `tools/perf_summary.py`，公式一致 |
| 阈值 | 文档写 0.9，代码不 gate | 默认不 gate |

**唯一有意适配**：FlagGems-vllm 用 `torch.cuda.get_device_name()` 取芯片名；
FlagTrain 支持多厂商，改用
`flag_train.runtime.torch_device_fn.get_device_properties(0)` 的 `name`
（兼容 `name` / 字典等属性名）。语义不变，只是取名字段换成跨厂商可用的。

---

## 3. 代码落点与改动

### 3.1 `benchmark/conftest.py`（对齐 FlagGems-vllm `conftest.py:98-120,203-214,293`）

1. `BenchConfig.__init__`（当前 `conftest.py:83`）新增：
   ```python
   self.base_data = None
   ```
2. `pytest_addoption`（当前 `conftest.py:175`）新增：
   ```python
   parser.addoption(
       "--base-data",
       action="store",
       default=None,
       required=False,
       help="Path to an NVIDIA baseline JSON (e.g. op_flagtrain_perf_baseline.json). "
            "When provided, a 'vs Base' column and compared_speedup are produced. "
            "If omitted or unreadable, behavior is unchanged.",
   )
   ```
3. 新增 `_load_base_data(path)`（逐字移植上游语义）：
   - `path` 为空 → 打印 `[base-data] 未指定 --base-data，'vs Base' 列关闭。`，返回 `None`；
   - 读取/JSON 解析失败 → 打印失败原因，返回 `None`，**不抛异常**；
   - 成功 → 打印算子数与 `_scaling_factors` 是否存在，返回 dict。
4. `pytest_configure`（当前 `conftest.py:332` 之后）：
   ```python
   Config.base_data = _load_base_data(config.getoption("--base-data"))
   ```
5. 传参统一用 `--base-data=<path>`（等号形式），避免 pytest 在注册选项前把绝对路径
   当成测试目标（上游 `tools/run_tests.py:933-942` 的同一处理）。

### 3.2 `benchmark/consts.py`（对齐 FlagGems-vllm `consts.py:228-561,584-640`）

1. `BenchmarkMetrics`（当前 `consts.py:108`）新增：
   ```python
   base_miss_reason: Optional[str] = None
   ```
   并把它加入 `ALL_AVAILABLE_METRICS` 的排除集合（当前 `consts.py:135-139`），
   与上游一致，避免 `--metrics` 误选。
2. 逐字移植以下函数（放在 `BenchmarkResult` 之前，与上游位置一致）：
   - `_DTYPE_TO_SF_KEY`、`_BOTTLENECK_TO_SF`；
   - `_first_tensor_shape`、`_all_tensor_shapes_from_record`、
     `_all_scalars_from_record`；
   - `_base_shapes_from_config`、`_base_scalars_from_config`；
   - `_resolve_op_entry`、`_build_signature_index`、`_match_by_signature`；
   - `lookup_base_record_ex`、`lookup_base_record`；
   - `_detect_self_chip`、`lookup_scaling_factor`。
3. `_detect_self_chip` 的适配（见 §2）：用 `flag_train.runtime.torch_device_fn`
   的设备属性取 `name`，兼容 dict / `name` / 其他属性名；缓存一次。
4. `BenchmarkResult.__str__`（当前 `consts.py:239`）：
   - `show_vs_base = any(m.compared_speedup is not None for m in self.result)`；
   - 在 `Train Speedup` 后追加 `f"{'vs Base':>20}"`；
   - `_format_metrics(ele, show_vs_base)` 增加参数，命中时输出
     `compared_speedup`（3 位小数），否则 `N/A`；
   - 未命中基线时输出与改动前**完全一致**（不新增列）。

### 3.3 `benchmark/base.py`（对齐 FlagGems-vllm `base.py:466-500`）

把上游 `run()` 里 `if Config.base_data is not None and metric.latency:` 的整块逻辑
移植过来。FlagTrain 有两条测量路径，用一个私有方法避免复制两份：

```python
def _record_base_comparison(self, metric, dtype):
    if Config.base_data is None or not metric.latency:
        return
    base_rec, miss_reason = lookup_base_record_ex(
        Config.base_data,
        self.op_name,
        dtype,
        metric.shape_detail,
        getattr(self, "base_variant", None),
    )
    base_ms = base_rec.get("latency_ms") if base_rec else None
    if base_ms is not None:
        speedup = base_ms / metric.latency          # 原始跨硬件时间比
        factor = lookup_scaling_factor(
            Config.base_data, vendor_name,
            base_rec.get("bottle_neck_unit"), dtype,
        )
        if factor:
            speedup /= factor                       # 峰值差距归一化
        metric.compared_speedup = speedup
        print(
            f"[vs Base] {self.op_name} shape={metric.shape_detail} "
            f"base={base_ms:.6f}ms train={metric.latency:.6f}ms "
            f"factor={factor} vs_base={speedup:.3f}"
        )
    else:
        metric.base_miss_reason = miss_reason
        print(f"[vs Base][未匹配] {miss_reason}")
```

调用点（测量成功之后、`_emit_result` 之前）：

- `_run_legacy`（当前 `base.py:548`）：
  ```python
  metric = self._measure_input(input)
  self._record_base_comparison(metric, dtype)
  metrics.append(metric)
  ```
- `_run_cases`（当前 `base.py:577`）：
  ```python
  metric = self._measure_input(input, case_id=case.case_id)
  self._record_base_comparison(metric, dtype)
  metrics.append(metric)
  ```

保持 `_measure_input` 现有签名不变；`speedup = latency_base/latency`
（`base.py:607-616`）仍然是“本机 torch 比值”，`compared_speedup` 是新增的
第二口径。

### 3.4 基线文件入库

与上游把基线放在 `tools/op_vllm_perf_baseline.json` 一致，FlagTrain 使用：

```
tools/op_flagtrain_perf_baseline.json
```

- 本地验证阶段可直接使用用户指定的绝对路径：
  `/root/wuhan/flagos-ops-baseline/op_flagtrain_lamb_h800_baseline.json`；
- 正式入库时把重新采集（NCU 有效）的文件复制到 `tools/`，CI 使用
  `--base-data=tools/op_flagtrain_perf_baseline.json`，与上游 CI 的写法一致
  （上游 `.github/workflows/ops-test.yaml:335`）。

### 3.5 新增 `tools/perf_summary.py`（对齐 `run_tests.py` + `psum_text`）

FlagTrain 没有多 GPU 调度器，只移植**统计与报告**部分，公式与上游一致：

- 逐 shape（对应 `run_tests.py:765-861` 的 `parse_perf_data`）：
  ```python
  speedup = res.get("speedup", 0.0)
  compared = res.get("compared_speedup", None)
  if compared is not None:
      speedup = compared          # vs Base 命中时覆盖
  # 只对数值型计入；None 排除；0.0 计入
  ```
  逐 dtype 求算术平均，保留 `details[shape]` 与 `base_miss_reasons`。
- 逐 op（对应 `psum_text:373-446` 的 `parse_lines`）：
  对有值的 dtype 再求算术平均，得到 `PerfAvg`。
- 输出：`summary.json`、`summary.csv`、`summary.md`。
- 参数：`--input`（可重复）/目录、`--output-dir`、可选 `--min-speedup`
  （默认不 gate，与上游一致；显式设置时低于阈值返回非 0）。
- 明确沿用上游的统计口径：
  - 命中与未命中的 shape 可能在同一 dtype 内混算；
  - `None` 不计 0；`0.0` 计入（以 `parse_perf_data` / `psum_text` 为准，
    不采用 `psum_html` 的 `if speedup:` 分支）；
  - 未命中原因去重后单独输出。

### 3.6 CI（第二阶段）

`.github/workflows/backend-test.yaml` 的性能步骤：

- 在 pytest 命令后追加
  `--base-data=tools/op_flagtrain_perf_baseline.json`（nvidia 与 vendor 作业一致）；
- 跑完后执行 `python tools/perf_summary.py --input <result.json> --output-dir ...`；
- 上传 summary 与 benchmark JSON；
- 默认不加 gate；如需验收阈值，用 `--min-speedup 0.8` 显式开启。

---

## 4. lamb 的计算与匹配

### 4.1 公式（与上游一致）

```
speedup          = latency_base / latency
raw_vs_base      = base_ms_NVIDIA / latency
factor           = P'_bottleneck,dtype / P_bottleneck,dtype
compared_speedup = raw_vs_base / factor
```

`factor` 缺失时保留 `raw_vs_base`。

### 4.2 匹配

- `op_name = "lamb"`；
- `dtype = torch.float32`；
- `record_shapes` 的第一个张量是 `p`，shape `[N]`；baseline 的 key 是
  `str([N])`（如 `"[1024]"`）；
- 直查失败时用签名回退：`[N], [0], [N], [N], [N]` 的排序多重集合与
  baseline `config.inputs` 一致；
- 超参数标量仅用于二次判别。

### 4.3 factor

- baseline 的 `bottle_neck_unit` → `mem`/`tensor`/`cuda`；
- dtype → `fp32`/`fp16`/`bf16`；
- 芯片按 vendor + 名称子串匹配；`measured` 优先、`nominal` 兜底；
- 芯片缺失或 `bottle_neck_unit=None` → factor 为 `None`。

---

## 5. 基线文件的现状与前置条件

当前输入文件（`op_flagtrain_lamb_h800_baseline.json`）已检查：

| 项目 | 现状 |
|---|---|
| `lamb` 记录 | 7 个 shape，`torch.float32`，`latency_ms` 均有值 |
| `bottle_neck_unit` | **全部为 `null`** |
| `num_kernels` / `U_*` / `F_*` | **0 / 空**（NCU `ERR_NVGPUCTRPERM` 的占位值） |
| `_scaling_factors` | 存在，15 颗芯片，`reference_chip=H800` |
| 采集设备 | `NVIDIA A100-SXM4-40GB`（与 H800 参考不一致） |

因此：

- **阶段 A（现在可做）**：用这份文件验证 `--base-data` 链路。预期
  `factor=None`，`compared_speedup` 为未归一化时间比；本地 `speedup`
  （本机 DeepSpeed fused_lamb / FlagTrain lamb）应与它接近（两侧同为
  fused_lamb，差异来自 do_bench 口径）。
- **阶段 B（归一化验收）**：在允许 NCU 的 **H800** 上重新采集，确认每条记录
  `num_kernels == 3`、`bottle_neck_unit` 非空，再跑同一命令；此时
  `factor=1.0`（H800 参考芯片），`compared_speedup` 才是上游口径的
  利用率归一化加速比。
- 若必须在 A100 上归一化：先把 A100 的 bandwidth/tensor/vector 规格加入
  `hardware_specs.json`，用 `--reference-chip A100` 重新采集。

---

## 6. 测试计划

### 6.1 单元测试（不需要 GPU）

新增 `tests/test_benchmark_base_data.py`：

- 三级匹配命中（op / shape / dtype）；
- 签名回退（key_shape 与第一个张量 shape 不一致）；
- 未命中原因：op 缺失 / shape 缺失 / dtype 缺失；
- `_scaling_factors`：measured 优先、nominal 兜底、芯片名匹配、缺失 → `None`；
- `compared_speedup = base_ms/latency/factor` 数值断言；
- `factor=None` 时保留原始比值。

新增 `tests/test_perf_summary.py`：

- `compared_speedup` 覆盖 `speedup`；
- 混合命中/未命中/`None`；
- dtype 算术平均、op 算术平均；
- `0.0` 计入、`None` 排除；
- 未命中原因汇总；
- `--min-speedup` 退出码。

### 6.2 lamb 集成测试

```bash
cd /root/wuhan/FlagTrain
PYTHONPATH=src python -m pytest benchmark/deepspeed/test_lamb.py -m lamb \
  --level core --record json --output /tmp/benchmark_lamb.json \
  --continue-on-collection-errors \
  --base-data=/root/wuhan/flagos-ops-baseline/op_flagtrain_lamb_h800_baseline.json
```

断言：

- 7 个 shape 都有 `compared_speedup`（阶段 A：`factor=None` 的原始比值）；
- 控制台出现 `[vs Base] ... factor=None ...`；
- JSON 的 `details[].result[]` 含 `compared_speedup`、`base_miss_reason=None`；
- 不带 `--base-data` 时，输出与改动前完全一致。

### 6.3 回归

- `PYTHONPATH=src python -m pytest tests/deepspeed -q` 全绿；
- `python -m pytest benchmark/deepspeed/test_lamb.py -m lamb --level core` 正常。

---

## 7. 验收标准

- [ ] `--base-data` 未指定/文件不可读：benchmark 行为与改动前完全一致。
- [ ] 带 `--base-data`：lamb 的 7 条记录都有 `compared_speedup`。
- [ ] 未命中：只写 `base_miss_reason`，benchmark 不失败。
- [ ] “vs Base” 列仅在存在 `compared_speedup` 时出现。
- [ ] 阶段 A：`factor=None`，`compared_speedup == base_ms/latency`。
- [ ] 阶段 B（有效 H800 基线）：`factor=1.0`，`compared_speedup == base_ms/latency`；
      H100/其他芯片按 measured 因子归一化。
- [ ] `tools/perf_summary.py` 的 dtype/op 均值与手算一致。
- [ ] 单元测试与 `tests/deepspeed` 回归通过。

---

## 8. 风险与待确认

1. **当前基线不可用于归一化**：`bottle_neck_unit=None`、U/F 为 0；阶段 A 只能
   验证机制。需要 NCU 权限恢复后在 H800 重采。
2. **参考芯片不一致**：数据采自 A100，`reference_chip=H800`；需重采或补 A100 规格。
3. **测量协议差异**：采集器 `do_bench(mean, 25/100 ms)`，benchmark
   `do_bench(median, 1000/100 ms)`；raw ratio 有几个百分点噪声。
4. **厂商芯片名**：`_detect_self_chip` 必须用 `torch_device_fn` 的属性，不能
   用 `torch.cuda.get_device_name()`。
5. **统计口径**：同一 dtype 内可能混算“归一化值”和“未命中的本机值”，与上游一致；
   summary 里单独输出 miss 数量。
6. **`base_miss_reason` 进入 JSON schema**：需加入 `ALL_AVAILABLE_METRICS`
   排除集；下游读取 JSON 的代码需容忍新字段。
7. **不移植的部分**：FlagTrain 没有 `run_tests.py` 的多 GPU 调度器和
   `psum_html/psum_web`；只移植统计公式到 `tools/perf_summary.py`。如果要求
   连调度器一起移植，需要单独评估。

---

## 9. 实施顺序

1. `benchmark/consts.py`：字段、查找、缩放、vs Base 列；
2. `benchmark/conftest.py`：`--base-data` + 加载；
3. `benchmark/base.py`：`_record_base_comparison` + 两个调用点；
4. 单元测试；
5. lamb 阶段 A 集成测试（当前文件，raw ratio）；
6. `tools/perf_summary.py` + 测试；
7. H800 重采后在阶段 B 验证归一化；
8. CI 接入与文档（可独立 PR）。

预计：新增 `consts.py` 约 250 行、`perf_summary.py` 约 150 行、测试约 200 行；
修改 `conftest.py` / `base.py` 约 60 行。
