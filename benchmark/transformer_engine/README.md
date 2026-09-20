# Transformer Engine 性能基准

本目录当前没有基准脚本或性能结果。这里放置 `src/flag_train/transformer_engine/`
后续实现对应的性能测量；先通过
[`tests/transformer_engine/`](../../tests/transformer_engine/) 中相关的正确性验证。

贡献基准时，应提供可复现入口和参考实现版本，在相同硬件、软件栈、输入形状、批量、
数据类型及精度策略下比较。记录预热与重复次数、设备同步方式、计时范围、统计方法及
峰值显存；说明量化、缩放等开销是否计入，不将不同精度的结果直接作为实现加速比。
