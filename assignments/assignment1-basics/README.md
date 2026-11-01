# CS336 2025 年春季作业 1：基础

作业的完整说明请参见作业讲义：
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

如果发现作业讲义或代码存在问题，欢迎提交 GitHub issue 或创建包含修复内容的
pull request。

## 设置环境

### 环境

我们使用 `uv` 管理环境，以确保可复现性、可移植性和易用性。
请从[这里](https://github.com/astral-sh/uv#installation)安装 `uv`（推荐），
或运行 `pip install uv`/`brew install uv`。
建议阅读[这份指南](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)，
了解如何使用 `uv` 管理项目（不会后悔的！）。

现在可以使用以下命令运行仓库中的任意代码：

```sh
uv run <python_file_path>
```

需要时，环境会自动解析并激活。

### 运行单元测试

```sh
uv run pytest
```

最初，所有测试都应因 `NotImplementedError` 而失败。
要将你的实现接入测试，请完成
[./tests/adapters.py](./tests/adapters.py) 中的函数。

### 下载数据

下载 TinyStories 数据和 OpenWebText 的一个子样本：

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
