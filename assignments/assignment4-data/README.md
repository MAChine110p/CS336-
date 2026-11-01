# CS336 2026 年春季作业 4：数据

作业的完整说明请参见作业讲义：
[cs336_assignment4_data.pdf](./cs336_assignment4_data.pdf)

如果发现作业讲义或代码存在问题，欢迎提交 GitHub issue 或创建包含修复内容的
pull request。

## 设置环境

本目录的结构如下：

- [`./cs336_basics`](./cs336_basics)：该模块包含作业 1 中由课程团队提供的语言模型实现。
  你将使用这份训练代码在过滤后的数据上训练语言模型。不要修改训练逻辑，因为提交到
  排行榜的代码必须完全使用该逻辑。
- [`./cs336_data`](./cs336_data)：该文件夹基本为空！你将在此模块中实现数据过滤和处理代码。

其结构大致如下：

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   └── ... an optimized training implementation ...
├── cs336_data  # TODO(you): code that you'll write for assignment 4
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 4 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 4 ...
```

与之前的作业一样，我们使用 `uv` 管理依赖。

## 下载数据

### 面向学生

Data is available at `/shared-data`, such as `uv run modal shell ./scripts/download_data.py::main --cmd "ls -l /shared-data"`

若只下载离线运行所需的文件，请运行 `uv run scripts/download_data.py --offline-only`。
若下载学生所需的全部数据，请运行 `uv run modal run scripts/download_data.py`。

### 面向非学生

若只下载离线运行所需的文件，请运行 `uv run scripts/download_data.py --offline-only`。

下载完整的非离线数据前，请先实现
[`./cs336_data/wet_files.py`](./cs336_data/wet_files.py) 中的 `is_english` 方法。

#### Modal 环境

Remove `environment_name` from `shared_data_volume` in [`./cs336_data/modal_utils.py`](./cs336_data/modal_utils.py)

```python
uv run modal run scripts/download_data.py
```

#### 非 Modal 环境

Change the path in [`./cs336_data/common.py`](./cs336_data/common.py) and run `uv run scripts/download_data.py`.

如果想下载少于 2.5k 个 WET 文件，可以考虑修改
[`./cs336_data/wet_files.py`](./cs336_data/wet_files.py) 中 `EnglishWetFiles` 的 `n_files`。

## 在 Modal 上训练

[`./scripts/train.py`](./scripts/train.py) 中的 Modal 入口包含完整训练配置；
使用 `--train-bin` 传入 GPT-2 分词训练数据的路径。

最终训练运行使用 8 个 B200 GPU：

```sh
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin
```

## 提交

提交时请运行 `./test_and_make_submission.sh`。该脚本会安装代码依赖、运行测试，
并创建包含输出的 zip 文件。解压提交的压缩包后，应能够运行
`./test_and_make_submission.sh` 来验证测试结果。
