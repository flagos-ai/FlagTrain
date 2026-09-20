# Transformer Engine 正确性测试

本目录当前没有可执行测试。这里放置 `src/flag_train/transformer_engine/` 后续实现对应的测试，
不代表已验证 Transformer Engine 的功能或兼容性。

贡献实现时，应同时提供参考实现及其版本，比较前向输出和输入、参数梯度，说明
数据类型、误差容限及边界输入。涉及低精度计算时，还需记录缩放策略及数值范围，
验证对应精度下的误差和稳定性，不以导入成功作为正确性证据。

测试入口和运行命令随首个可执行测试一起补充；性能测量放在
[`benchmark/transformer_engine/`](../../benchmark/transformer_engine/)。
