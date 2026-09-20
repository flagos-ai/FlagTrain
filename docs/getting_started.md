# 入门

FlagTrain 当前提供按组件划分的 Python 包和目录骨架，尚无算子实现、上游框架接入或
可运行训练入口。没有经过验证的框架版本、硬件兼容性或性能结论。

## 安装骨架

在独立 Python 环境中，从仓库根目录执行：

```sh
python -m pip install -e .
```

此命令仅安装当前包骨架，不安装 Megatron、DeepSpeed 或 Transformer Engine，
也不提供训练能力。可用以下命令检查包路径：

```sh
python -c "import flag_train; import flag_train.megatron; import flag_train.deepspeed; import flag_train.transformer_engine"
```

导入成功只说明骨架可被 Python 找到，不代表任何算子、上游框架或硬件通过验证。

## 后续开发

三个组件的实现分别位于 `src/flag_train/megatron/`、`src/flag_train/deepspeed/` 和
`src/flag_train/transformer_engine/`。对应的 `tests/`、`benchmark/`、`examples/`
目录目前只有贡献说明；具体执行命令应与首个真实实现一起提交。

参与前请阅读 [开发说明](development.md)、[兼容性记录](compatibility.md) 和
[贡献指南](../CONTRIBUTING.md)。
