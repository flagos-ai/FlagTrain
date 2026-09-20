# Transformer Engine 示例

本目录当前没有可运行示例或训练入口。这里放置使用 `flag_train.transformer_engine`
后续实现的最小示例；仅有目录与包名不表示 Transformer Engine 已完成接入。

贡献示例时，应写明依赖及版本、硬件、输入来源、执行步骤和预期输出，并在声明的环境
实际运行。示例引用的功能应已有前向、梯度和精度验证；涉及低精度时，说明精度和缩放
配置及误差预期。正确性测试归入
[`tests/transformer_engine/`](../../tests/transformer_engine/)，性能测量归入
[`benchmark/transformer_engine/`](../../benchmark/transformer_engine/)。
