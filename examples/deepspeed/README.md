# DeepSpeed 示例

本目录当前没有可运行示例或训练入口。这里放置使用 `flag_train.deepspeed` 后续实现的
最小示例；仅有目录与包名不表示 DeepSpeed 已完成接入。

贡献示例时，应写明依赖及版本、硬件、输入来源、执行步骤和预期输出，并在声明的环境
实际运行。示例引用的功能应已有前向、梯度和精度验证；涉及分片、卸载或梯度累积时，
提供对应配置和资源要求。正确性测试归入 [`tests/deepspeed/`](../../tests/deepspeed/)，
性能测量归入 [`benchmark/deepspeed/`](../../benchmark/deepspeed/)。
