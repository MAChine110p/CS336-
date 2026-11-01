# CS336 2026 年春季作业 2：系统

作业的完整说明请参见作业讲义：
[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf)

如果发现作业讲义或代码存在问题，欢迎提交 GitHub issue 或创建包含修复内容的
pull request。

## 设置环境

本目录的结构如下：

- [`./cs336-basics`](./cs336-basics)：包含一个
  `cs336_basics` 模块及其关联的 `pyproject.toml`。该模块包含作业 1 中由课程团队
  提供的语言模型实现。如果想使用自己的实现，可以将此目录替换为自己的实现。
- [`./cs336_systems`](./cs336_systems)：该文件夹基本为空！你将在此模块中实现
  优化后的 Transformer 语言模型。可以从作业 1 的 `cs336-basics` 中复制所需代码，
  作为起点。此外，还将在此模块中实现分布式训练和优化。

其结构大致如下：

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

如果想使用自己的作业 1 实现，请用自己的实现替换 `cs336-basics` 目录，或编辑外层
`pyproject.toml` 文件，使其指向自己的实现。

0. 我们使用 `uv` 管理依赖。运行以下命令可以验证 `cs336-basics` 包中的代码是否可访问：

```sh
$ uv run python
Using CPython 3.13.13
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 78 packages in 168ms
Python 3.13.13 (main, Apr  7 2026, 20:49:46) [Clang 22.1.1 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cs336_basics
...
```

`uv run` 会根据 `pyproject.toml` 文件中的声明自动安装依赖。

## 提交

提交时请运行 `./test_and_make_submission.sh`。该脚本会安装代码依赖、运行测试，
并创建包含输出的 gzip 压缩 tar 包。解压提交的 tar 包后，应能够运行
`./test_and_make_submission.sh` 来验证测试结果。
