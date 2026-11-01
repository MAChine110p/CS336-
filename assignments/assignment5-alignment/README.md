# CS336 2026 年春季作业 5：对齐

作业的完整说明请参见作业讲义：
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

我们还提供了一份关于安全对齐、指令微调和 RLHF 的补充作业（完全可选）：
[cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

如果发现作业讲义或代码存在问题，欢迎提交 GitHub issue 或创建包含修复内容的
pull request。

## 设置环境

与之前的作业一样，我们使用 `uv` 管理依赖。

1. 先安装除 `flash-attn` 外的所有包，再安装全部包（`flash-attn` 比较特殊）：
```
uv sync --no-install-package flash-attn
uv sync
```

2. 运行必需的单元测试：

``` sh
uv run pytest tests/test_grpo.py
```

最初，所有测试都应因 `NotImplementedError` 而失败。
要将你的实现接入测试，请完成
[./tests/adapters.py](./tests/adapters.py) 中的函数。
