# Megatron 示例

本目录当前没有可运行示例或训练入口。这里放置使用 `flag_train.megatron` 后续实现的
最小示例；仅有目录与包名不表示 Megatron 已完成接入。

贡献示例时，应写明依赖及版本、硬件、输入来源、执行步骤和预期输出，并在声明的环境
实际运行。示例引用的功能应已有前向、梯度和精度验证；若需要多进程，说明启动方式及
资源要求。正确性测试归入 [`tests/megatron/`](../../tests/megatron/)，性能测量归入
[`benchmark/megatron/`](../../benchmark/megatron/)。
