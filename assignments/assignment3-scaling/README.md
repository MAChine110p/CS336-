# CS336 2026 年春季作业 3：扩展

作业的完整说明请参见作业讲义：
[cs336_assignment3_scaling.pdf](./cs336_assignment3_scaling.pdf).

如果发现作业讲义或代码存在问题，欢迎提交 GitHub issue 或创建包含修复内容的
pull request。

## 面向学生

安装 uv：

```sh
uv sync
```

将 `A3_API_KEY` 设置为你的 8 位学号：

```sh
export A3_API_KEY=06123456
```

托管的训练 API 地址为：

```text
http://hyperturing.stanford.edu:8000
```

点击查看[文档](http://hyperturing.stanford.edu:8000/docs)和[控制面板](http://hyperturing.stanford.edu:8000/dashboard)。

提交和检查训练运行的示例请参见
[`./examples/client_example.ipynb`](./examples/client_example.ipynb)。

## 面向非学生

安装依赖：

```sh
uv sync --extra server
```

下载已分词数据：

```sh
uv run modal run scripts/1_download_tokenized_data.py
```

直接运行训练：

```sh
uv run cs336_scaling/training/run.py
```

要运行 API 和调度器，请设置：

```sh
DATABASE_URL_PROD="postgresql://..."
DATABASE_URL_DEV="postgresql://..."
INTERNAL_API_KEY="SOMEKEY"
```

Then run:

```sh
DB_ENV=prod uv run fastapi run &
DB_ENV=prod uv run dispatcher &
```
