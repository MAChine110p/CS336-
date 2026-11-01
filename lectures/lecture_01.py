import os

import regex
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
from edtrace import link, text, image
from lecture_util import article_link, post_link, video_link, get_local_url
from references import shannon_1950, lstm_1997, brants_2007, bengio_2003, glorot_2010, seq2seq_2014
from references import bahdanau_2015_attention, transformer_2017, gpt2_2019, t5_2019, kaplan_scaling_laws_2020, mup_2022
from references import dpo_2023, adamw_2017, adam_2014, grpo, ppo_2017, muon_2024
from references import large_batch_training_2018, wsd_2024, cosine_learning_rate_2017, moe_2017, switch_transformers_2021, auxfree_2024, mtp_2024
from references import megatron_lm_2019, shazeer_2020, elmo_2018, bert_2018
from references import rms_norm_2019, layernorm_2016, pre_post_norm_2020, qk_norm_2023
from references import rope_2021, soap_2024, sparse_transformer_2019, gqa_2023, mla_2024
from references import linear_attention_2020, mamba_2_2024, gdn_2024, mamba_3_2026
from references import megabyte_2023, byt5_2021, blt_2024, tfree_2024, hnet_2025, sennrich_2016, zero_2019, gpipe_2018
from references import regmix_2025, olmix_2026, wrap_2024

from references import gpt_3_2020, gpt_4_2023, instruct_gpt_2022
from references import the_pile_2020, gpt_j_2021, opt_175b_2022, bloom_2022, palm_2022, chinchilla_2022
from references import llama_2023, llama_2_2023, llama_3_2024
from references import mistral_7b_2023, mixtral_2024
from references import deepseek_67b_2024, deepseek_v2_2024, deepseek_v3_2024, deepseek_r1_2025, deepseek_v3_2_2025
from references import qwen_2_5_2024, qwen_3_2025
from references import kimi_1_5_2025, kimi_k2_5_2026
from references import glm_4_5_2025, glm_5_2026
from references import minimax_m2_5_2026
from references import xiaomi_mimo_v2_2026

from references import marin_8b_2025, marin_32b_2025
from references import olmo_7b_2024, olmo_2_2025, olmo_3_2025
from references import nemotron_15b_2024, nemotron_3_2025

import tiktoken

def main():
    welcome()
    why_this_course_exists()
    current_lm_landscape()

    what_is_this_program()

    course_logistics()
    course_syllabus()

    tokenization()  # 第一个单元

    text("下节课：资源核算")


def welcome():
    text("## CS336：从零开始构建语言模型（2026 年春季）"),

    image("images/course-staff.png", width=600)
    text("……为大家带来 CS336 的第三次开课。")

    text("第二次开课（2025 年春季）的讲课视频在 [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) 上。")
    text("有什么新变化？")
    text("- 沿用同样的'从零开始'理念")
    text("- 优先讲解性价比最高的概念，不要只见树木不见森林")
    text("- 更多地覆盖现代语言模型的组成部分（专家混合、长上下文、智能体）")


def why_this_course_exists():
    text("## 我们为什么要开设这门课程？")

    text("问题：研究者们正在与底层技术**脱节**。")
    text("- 2016 年：研究者自己实现并训练模型。")
    text("- 2018 年：研究者下载模型（例如 BERT）并进行微调。")
    text("- 今天：研究者对 API 模型（例如 GPT/Claude/Gemini）进行提示。")

    text("向更高层次的抽象迈进提升了生产力，但是")
    text("- 这些抽象是存在泄漏的（与编程语言或操作系统不同）。")
    text("- 仍然有一些基础研究需要拆开整个技术栈才能完成。")

    text("要进行**基础研究**，就必须**完整理解**这项技术。")

    text("本课程的理念：**通过构建来理解**。")
    text("但有一个小小的问题……")

    text("## 语言模型的工业化")
    image("https://upload.wikimedia.org/wikipedia/commons/c/cc/Industrialisation.jpg", width=400)

    text("前沿模型非常昂贵：")
    text("- 2023 年：据说 GPT-4 的训练成本高达 1 亿美元。 "), article_link("https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/")
    text("- 2025 年：xAI 建造了拥有 23 万块 GPU 的集群来训练 Grok。 "), article_link("https://x.com/elonmusk/status/1947701807389515912")

    text("前沿模型是如何构建的，没有任何公开细节。")
    text("来自 GPT-4 技术报告 "), link(gpt_4_2023), text("：")
    image("images/gpt4-no-details.png", width=600)

    text("前沿模型对我们来说是遥不可及的。")
    text("我们可以构建小型语言模型（<10 亿参数），但这可能无法代表大型语言模型。")

    text("例 1：花在注意力与 MLP 上的 FLOPs 比例会随规模变化。 "), post_link("https://x.com/stephenroller/status/1579993017234382849")
    image("images/roller-flops.png", width=400)
    text("例 2：能力随规模涌现 "), link("https://arxiv.org/pdf/2206.07682")
    image("images/wei-emergence-plot.png", width=600)

    text("## 我们在这门课上学到的东西，哪些能迁移到前沿模型上？")
    text("知识分为三种类型：")
    text("- **机制**：事物如何工作（Transformer 是什么、模型并行如何运作）")
    text("- **思维方式**：榨干硬件的每一分性能，认真对待规模扩展")
    text("- **直觉**：哪些数据和建模决策能带来良好的准确率")

    text("我们可以教授机制和思维方式（这些是可以迁移的）。")
    text("我们只能部分教授直觉（不一定能跨规模迁移）。")

    text("## 直觉？🤷")
    text("有些设计决策根本无法（尚无法）从道理上论证，纯粹来自实验。")
    text("例：Noam Shazeer 那篇引入 SwiGLU 的论文 "), link(shazeer_2020)
    image("images/divine-benevolence.png", width=600)

    text("## 苦涩的教训（The ?? ??）")
    text("错误解读：规模就是一切，算法不重要。")
    text("正确解读：能够随规模扩展的算法才是重要的。")
    text("### 准确率 = 效率 x 资源")
    text("事实上，规模越大，效率越重要（浪费不起）。")
    link("https://arxiv.org/abs/2005.04305"), text(" 表明，2012 年到 2019 年间 ImageNet 上的算法效率提升了 44 倍。")

    text("一个思考框架：在给定的算力和数据预算下，能构建出的最好模型是什么？")
    text("换句话说，**最大化效率**！")


def current_lm_landscape():
    text("## 前神经网络时代（2010 年代之前）")
    text("- 用语言模型度量英语的熵 "), link(shannon_1950)
    text("- N-gram 语言模型（用于机器翻译和语音识别系统） "), link(brants_2007)

    text("## 神经网络成分（2010 年代）")
    text("- 长短期记忆网络（LSTM） "), link(lstm_1997)
    text("- 首个神经语言模型 "), link(bengio_2003)
    text("- 序列到序列建模（用于机器翻译） "), link(seq2seq_2014)
    text("- Adam 优化器 "), link(adam_2014)
    text("- 注意力机制（用于机器翻译） "), link(bahdanau_2015_attention)
    text("- Transformer 架构（用于机器翻译） "), link(transformer_2017)
    text("- 专家混合（Mixture ? experts） "), link(moe_2017)
    text("- 模型并行 "), link(gpipe_2018), link(zero_2019), link(megatron_lm_2019)

    text("## 早期基础模型（2010 年代末期）")
    text("- ELMo：用 LSTM 预训练，微调可提升下游任务 "), link(elmo_2018)
    text("- BERT：用 Transformer 预训练，微调可提升下游任务 "), link(bert_2018)
    text("- Google 的 T5（11B）：把一切任务都转化为文本到文本 "), link(t5_2019)

    text("## 拥抱规模扩展")
    text("- OpenAI 的 GPT-2（1.5B）：流畅的文本，零样本能力的最初迹象 "), link(gpt2_2019)
    text("- 缩放定律：为规模扩展提供希望/可预测性 "), link(kaplan_scaling_laws_2020)
    text("- OpenAI 的 GPT-3（175B）：上下文学习 "), link(gpt_3_2020)
    text("- Google 的 PaLM（540B）：超大规模，但训练不足 "), link(palm_2022)
    text("- DeepMind 的 Chinchilla（70B）：算力最优的缩放定律 "), link(chinchilla_2022)

    text("## 开放模型")
    text("早期尝试（尝试复现 GPT-3）：")
    text("- EleutherAI 的开放数据集（The Pile）和模型（GPT-J） "), link(the_pile_2020), link(gpt_j_2021)
    text("- Meta 的 OPT（175B）：GPT-3 复现，遇到很多硬件问题 "), link(opt_175b_2022)
    text("- Hugging Face / BigScience 的 BLOOM（176B）：专注于数据获取 "), link(bloom_2022)

    text("可信的开放权重模型（权重 + 论文）：")
    text("- Meta 的 Llama 系列模型 "), link(llama_2023), link(llama_2_2023), link(llama_3_2024)
    text('- Mistral 的模型 '), link(mistral_7b_2023), link(mixtral_2024)
    text("- DeepSeek 的模型 "), link(deepseek_67b_2024), link(deepseek_v2_2024), link(deepseek_v3_2024), link(deepseek_r1_2025), link(deepseek_v3_2_2025)
    text("- 阿里巴巴的 Qwen 模型 "), link(qwen_2_5_2024), link(qwen_3_2025)
    text("- 月之暗面（Moonshot）的 Kimi 模型 "), link(kimi_1_5_2025), link(kimi_k2_5_2026)
    text("- 智谱（Z.ai）的 GLM 模型 "), link(glm_4_5_2025), link(glm_5_2026)
    text("- Minimax 的模型 "), link(minimax_m2_5_2026)
    text("- 小米的 MIMO 模型 "), link(xiaomi_mimo_v2_2026)
    text("这些模型正在逼近闭源模型（GPT、Claude、Gemini 等）。")

    text("开源模型（权重 + 论文 + 代码 + 数据）：")
    text("- AI2 的 Olmo 系列模型 "), link(olmo_7b_2024), link(olmo_2_2025), link(olmo_3_2025)
    text("- NVIDIA 的 Nemotron 系列模型 "), link(nemotron_15b_2024), link(nemotron_3_2025)
    text("- Marin 的模型（开放开发过程） "), link(marin_8b_2025), link(marin_32b_2025)

    text("开放性对信任和创新非常重要 "), link("https://arxiv.org/abs/2403.07918")
    text("正是开放模型带来的思想让我们得以教授 CS336。")

    text("什么是语言模型？")
    text("- 2018 年（BERT）：一个用来微调的东西")
    text("- 2020 年（GPT-3）：一个用来提示的东西")
    text("- 2022 年（ChatGPT）：一个可以对话的东西 "), link(title="示例对话", url="https://huggingface.co/datasets/HuggingFaceTB/smoltalk/viewer/all/train?row=72&conversation-viewer=72")
    text("- 2026 年（智能体）：一个能自主行动的东西 "), link(title="示例轨迹", url="https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/viewer/default/train?conversation-viewer=1")

    text("基本原理是不变的（注意力、内核、优化）。")
    text("只是规格不同（上下文更长，推理效率更加重要）。")


def what_is_this_program():
    text("这是一个*可执行讲义*（executable lecture），一个通过运行来呈现讲义内容的程序。")
    text("可执行讲义让我们可以：")
    text("- 查看并运行代码（因为一切都是代码！）")
    total = 0  # @inspect total
    for x in [1, 2, 3]:  # @inspect x
        total += x  # @inspect total
    text("- 查看讲义的层级结构")


def course_logistics():
    text("所有信息都在网上："), link(title="课程网站", url="https://stanford-cs336.github.io/spring2026/")

    text("这是一门 5 学分的课程。")
    text("2024 年春季课程评价中的一条评论：")
    text("> *整个作业的工作量大致相当于 CS 224n 全部 5 次作业加上期末项目的总和。而这仅仅是第一次作业。*")

    text("## 你为什么应该选这门课")
    text("- 你有一种执念，想要彻底理解事物是如何运作的。")
    text("- 你想锻炼你的研究工程能力。")

    text("## 你为什么不应该选这门课")
    text("- 你这个季度其实想做出研究成果。（去和你的导师聊聊。）")
    text("- 你想学习 AI 中最热门的新技术（例如多模态、RAG 等）。（你应该去上研讨课。）")
    text("- 你想在自己的应用领域取得好效果。（你应该直接提示或微调现成的模型。）")

    text("## 如何在家自学跟上进度")
    text("- 所有讲义材料和作业都会发布在网上，欢迎自学。")
    text("- 课程通过 [CGOE](https://cgoe.stanford.edu/) 录制。")

    text("## 作业")
    text("- 5 次作业（基础知识、系统、缩放定律、数据、对齐）。")
    text("- 没有脚手架代码，但我们提供单元测试和适配器接口，帮助你检查正确性。")
    text("- 先在本地实现以测试正确性，然后在集群上运行进行基准测试（准确率和速度）。")
    text("- 部分作业设有排行榜（在给定训练预算下最小化困惑度）。")

    text("## AI 使用政策")
    text("- 编程智能体可以解出所有作业，但你什么也学不到。")
    text("- AI 在回答问题和辅导方面可以非常有用。")
    text("- 你必须使用我们提供的 AGENTS.md 文件，它要求 AI 以教学为导向。")
    text("- 请阅读我们的 [AI 使用政策指南](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0)。")

    text("## 算力")
    text("- 感谢 [Modal](https://modal.com/) 提供算力。🙏")
    text("- 请阅读关于如何访问和使用算力的[指南](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef)。")


def course_syllabus():
    basics()         # 作业 1：分词、模型架构、训练
    systems()        # 作业 2：内核、并行、推理
    scaling_laws()   # 作业 3：缩放定律
    data()           # 作业 4：评估、收集、转换、过滤、去重、混合
    alignment()      # 作业 5：RLHF、RL 算法、RL 系统

    text("记住，一切的核心都是**效率**：")
    text("- 资源：数据 + 硬件（算力、内存、通信带宽）")
    text("- 在固定的资源下，如何训练出最好的模型？")

    text("如今，我们受限于算力，因此各种设计决策都体现着如何榨干给定硬件的性能。")
    text("- 系统：显然与效率有关")
    text("- 分词：直接处理原始字节很优雅，但在当今的模型架构下计算效率低")
    text("- 模型架构：许多改动都是为了减少内存或 FLOPs（例如共享 KV 缓存、滑动窗口注意力）")
    text("- 数据过滤：避免把宝贵的算力浪费在糟糕/无关的数据上")
    text("- 缩放定律：用更少的算力在小型模型上做超参数调优")

    text("明天，我们将变为受数据约束……")


class Tokenizer(ABC):
    """分词器的抽象接口。"""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


def basics():
    text("目标：能够训练一个基础语言模型")
    text("组成部分：分词、模型架构、训练")

    text("## 分词")
    text("模型操作的基本单元是什么？")
    text("形式化地说：分词器在原始输入（字节）和整数序列（token）之间进行转换")
    image("images/tokenized-example.png", width=600)
    text("流行的分词器：**字节对编码**（BPE） "), link(sennrich_2016)
    text("直觉：把输入切分成频繁出现的块")
    text("效率视角")
    text("- 缩短上下文长度（1000 字节 → 约 250 个 token）")
    text("- 自适应计算（在输入中有趣的部分投入更多建模容量）")

    text("梦想：无分词器的模型架构，直接在字节上操作 "), link(byt5_2021), link(megabyte_2023), link(blt_2024), link(tfree_2024), link(hnet_2025)
    text("这些方法很有前景，但尚未扩展到前沿规模。")

    text("## 模型架构")
    text("起点：原始 Transformer "), link(transformer_2017)
    image("images/transformer-architecture.png", width=500)

    text("各项改进：")
    text("- 激活函数：ReLU、SwiGLU "), link(shazeer_2020)
    text("- 位置编码：正弦编码、RoPE "), link(rope_2021)
    text("- 归一化：LayerNorm、RMSNorm、QK ??、pre-?? 与 post-?? "), link(layernorm_2016), link(rms_norm_2019), link(qk_norm_2023), link(pre_post_norm_2020)
    text("- 注意力：全注意力、稀疏/局部注意力、分组查询注意力（GQA）、多头潜在注意力（MLA） "), link(sparse_transformer_2019), link(gqa_2023), link(mla_2024)
    text("- 循环/状态空间模型/线性注意力：Mamba、Gated DeltaNet "), link(linear_attention_2020), link(mamba_2_2024), link(gdn_2024), link(mamba_3_2026)
    text("- MLP：稠密、专家混合 "), link(moe_2017), link(switch_transformers_2021)
    text("- 形状（隐藏维度、深度、头数、专家数）")
    #

    text("## 训练")
    text("如何设置模型的参数？")
    text("- 损失函数（例如多 token 预测） "), link(mtp_2024), link(deepseek_v3_2024)
    text("- 优化器（例如 AdamW、SOAP、Muon） "), link(adam_2014), link(adamw_2017), link(soap_2024), link(muon_2024)
    text("- 初始化规模（例如 Xavier 初始化、muP） "), link(glorot_2010), link(mup_2022)
    text("- 学习率调度（例如余弦、WSD） "), link(cosine_learning_rate_2017), link(wsd_2024)
    text("- 正则化（例如 dropout、权重衰减）")
    text("- 批大小（例如临界批大小） "), link(large_batch_training_2018)
    text("- MoE 特有：负载均衡（例如 aux-??） "), link(auxfree_2024), link(deepseek_v3_2024)

    text("## 作业 1（基础知识）")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment1-basics"), link(title="PDF", url="https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2026_assignment1_basics.pdf")
    text("- 实现 BPE 分词器")
    text("- 实现 Transformer、交叉熵损失、AdamW 优化器、训练循环")
    text("- 进行资源核算")
    text("- 在 TinyStories 和 OpenWebText 上训练")
    text("- 排行榜：在 B200 上用 45 分钟最小化 OpenWebText 困惑度 "), link(title="去年的排行榜", url="https://github.com/stanford-cs336/spring2025-assignment1-basics-leaderboard")

    text("高层原则：一切都是为了平衡以下三者：")
    text("- 表达能力（能表示数据中的复杂依赖关系）")
    text("- 稳定性（让参数和梯度范数保持在恰到好处的区间）")
    text("- 效率（在硬件上跑得快，训练和推理都是）")


def systems():
    text("目标：榨干硬件（GPU 或 TPU）的性能")
    text("组成部分：内核、并行、推理")

    text("## 基础")
    text("- 资源核算：模型的内存和计算特性")
    total_flops = 6 * 70e9 * 1e12  # 在 1T token 上训练 70B 参数 = 4.2e23 FLOPs @inspect total_flops
    image("images/compute-memory.png", width=300)
    text("- 模型参数必须从内存（HBM）搬到计算单元（SM）")
    text("- 例：B200 在 bf16 下可进行 2.25 PFLOP/秒的运算，内存带宽为 8TB/秒")
    text("- Roofline 分析：弄清楚我们究竟受算力约束还是受内存约束")
    text("- 基准测试与性能分析（nsight）：看看实践中到底发生了什么")

    text("[DGX B200](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html)：")
    image("https://docs.nvidia.com/dgx/dgxb200-user-guide/_images/dgx-b200-system-topology.png", width=500)

    text("## 内核")
    text("- 内核是在 GPU 上运行的函数")
    text("- 使用 PyTorch 时，每个基础算子都会启动一个标准内核")
    text("- 可以编写自定义内核让 GPU 飞起来")
    text("- 原则：组织计算以最小化数据搬运")
    text("- 朴素做法：读 HBM；计算 A；写 HBM；读 HBM；计算 B；写 HBM")
    text("- 融合做法：读 HBM；计算 A 和 B；写 HBM")
    text("- 策略：算子融合（矩阵乘法 + 激活函数）、分块（FlashAttention）")
    text("- Warp 分支发散、内存合并访问、bank 冲突、占用率、批量异步内存传输")
    text("- 用 CUDA/**Triton**/CUTLASS/ThunderKittens 编写内核")

    text("## 并行")
    text("- 如果我们有 1024 块 GPU 呢？")
    text("- GPU 之间的数据搬运更慢，但'最小化数据搬运'的原则依然成立")
    text("- 使用经典的集合通信操作（例如 gather、??、??-??）")
    text("- 将内存（参数、激活值、梯度、优化器状态）分片到多块 GPU 上")
    text("- 如何切分计算：{数据、张量、流水线、序列、专家}并行")

    text("## 推理")
    text("目标：根据提示生成 token（真正使用模型时必需！）")
    text("强化学习、测试时计算、评估也需要推理")
    text("两个阶段：prefill（预填充）和 decode（解码）")
    image("images/prefill-decode.png", width=500)
    text("- Prefill（与训练类似）：token 是给定的，可以一次性全部处理（受算力约束）")
    text("- Decode：需要一次生成一个 token（受内存约束）")
    text("加速解码的方法：")
    text("- 使用更便宜的模型（通过模型剪枝、量化、蒸馏）")
    text("- 投机解码：用更便宜的\"草稿\"模型生成多个 token，再用完整模型并行打分（精确解码！）")
    text("- 系统优化：融合内核、连续批处理")

    text("## 作业 2（系统）")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment2-systems"), link(title="2025 年春季 PDF", url="https://github.com/stanford-cs336/assignment2-systems/blob/spring2025/cs336_spring2025_assignment2_systems.pdf")
    text("- 用 Triton 实现融合的 RMSNorm 内核")
    text("- 实现分布式数据并行训练")
    text("- 实现优化器状态分片")
    text("- 对实现进行基准测试和性能分析")

    text("推荐书籍：[How to Scale Your Model](https://jax-ml.github.io/scaling-book/)")
    text("- 在概念层面清晰地讲解了如何构建 LLM 系统")
    text("- 出自 Google，以 TPU 为主，但高层概念是相通的")


def scaling_laws():
    text("背景：如果你有 1e25 FLOPs 的算力，你会用哪些超参数来训练一个好模型？")
    text("在全规模下做超参数调优太贵了！")

    text("关键的概念转变：不要只考虑单一规模，而是想一个**缩放配方**（?? recipe，FLOPs → 超参数）")
    text("对于一个缩放配方：")
    #类似于现在测试样本上训练，再扩大
    text("- 做实验，在多个较小规模下计算损失（例如最高到 1e24 FLOPs）")
    text("- 拟合一条缩放定律，预测该缩放配方在目标规模下的损失（例如 1e25 FLOPs）")

    text("现在你可以：")
    text("1. 利用小规模实验来优化面向更大规模的缩放配方")
    text("2. 在真正跑实验之前，就预测出目标规模下的损失！")
    text("缩放定律不会自动出现，需要精心构造缩放配方。")
    text("以某种方式对模型进行参数化，从而获得**超参数迁移** "), link(mup_2022)
    text("可预测性至少与最优性同等重要！")

    text("问题：给定 FLOPs 预算（C = 6 N D），应该用更大的模型（N）还是在更多 token 上训练（D）？")
    text("经典的算力最优缩放定律： "), link(kaplan_scaling_laws_2020), link(chinchilla_2022)
    text("- ISOFLOP 曲线：对多个较小的 FLOPs 预算，找出最优的 N")
    text("- 然后拟合缩放定律，外推到大的 FLOPs 预算")
    image("images/chinchilla-isoflop.png", width=800)
    text("一句话结论：D = 20 N 大致最优（例如 70B 参数模型应在约 1.4T token 上训练）")
    text("注意：这没有考虑推理成本（此时更想要较小的模型）")

    text("来自 Marin 的实时例子 "), post_link("https://x.com/percyliang/status/2034367256277533100")
    image("https://pbs.twimg.com/media/HDuErvvbsAAQ5Yt?format=jpg&name=4096x4096", width=600)
    text("本周应该就能训练完成，到时候看看我们能多好地匹配预先注册的损失！")

    text("## 作业 3（缩放定律）")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment3-scaling"), link(title="2025 年春季 PDF", url="https://github.com/stanford-cs336/assignment3-scaling/blob/master/cs336_spring2025_assignment3_scaling.pdf")
    text("- 我们基于以往的运行定义一个训练 API（超参数 → 损失）")
    text("- 提交\"训练任务\"（在 FLOPs 预算内）并收集数据点")
    text("- 对数据点拟合缩放定律")
    text("- 提交外推的超参数和损失预测")
    text("- 排行榜：在给定 FLOPs 预算下最小化损失")


def data():
    text("问题：我们希望模型具备哪些能力？")
    text("多语言？擅长对话？智能体编程能力？")

    text("## 评估")
    text("评估的目的是什么？")
    text("1. 内部：指导模型开发（跨规模的平滑性、相对性能重要）")
    text("2. 外部：衡量真实用例的绝对质量（生态效度重要）")
    text("评估示例：")
    text("1. 困惑度：最好在私有文档上运行，而不是互联网上的（避免数据污染）")
    text("2. 高级用例：GPQA、HLE、SWE-Bench、Terminal-Bench")
    text("语言模型是通用的，因此需要多样化的评估集！")

    text("## 数据收集")
    text("- 数据不会从天上掉下来。")
    text("- 来源：从互联网爬取的网页、书籍、arXiv 论文、GitHub 代码等。")
    image("https://ar5iv.labs.arxiv.org/html/2101.00027/assets/pile_chart2.png", width=600)
    text("- 依据合理使用原则在版权数据上训练？ "), link("https://arxiv.org/pdf/2303.15715.pdf")
    text("- 可能不得不购买数据授权（例如 Google 购买 Reddit 数据） "), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    text("- 原始数据是 HTML、PDF、目录（而非文本），需要处理")

    text("## 数据处理")
    text("- 转换：把 HTML/PDF 转成文本（提取正文）")
    text("- 过滤：保留高质量数据，去除有害内容（借助分类器）")
    text("- 去重：节省算力，避免记忆化；使用布隆过滤器或 MinHash")
    text("- 数据混合：每个来源应该上调/下调多少权重？ "), link(regmix_2025), link(olmix_2026)
    text("- 改写/合成数据：用 LM 增强真实数据，使其更接近下游任务 "), link(wrap_2024)

    text("数据的类型：")
    text("- 预训练数据：大规模且多样")
    text("- 中期训练数据：高质量，包括长上下文")
    text("- 后训练数据：监督微调（对话、带工具调用的智能体轨迹）")

    text("## 作业 4（数据）")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment4-data"), link(title="2025 年春季 PDF", url="https://github.com/stanford-cs336/assignment4-data/blob/spring2025/cs336_spring2025_assignment4_data.pdf")
    text("- 把 ?? ?? 的 HTML 转成文本")
    text("- 训练分类器，过滤质量和有害内容")
    text("- 用 MinHash 去重")
    text("- 排行榜：在给定 token 预算下最小化困惑度")


def alignment():
    text("到目前为止，我们在完全监督下训练模型（预测下一个 token）。")
    text("现在模型已经比较像样了，我们可以从**弱监督**进一步改进它。")
    text("为什么用弱监督？因为评价一个东西往往比生成它更容易。")

    text("基本模板：")
    text("1. 让模型生成回复。")
    text("2. 用{人类、验证器、LM 裁判}给回复打分。")
    text("3. 更新模型，使其偏好更好的回复。")

    text("算法：")
    text("- 来自强化学习的近端策略优化（PPO） "), link(ppo_2017), link(instruct_gpt_2022)
    text("- 直接偏好优化（DPO）：用于偏好数据，更简单 "), link(dpo_2023)
    text("- 组相对偏好优化（GRPO）：去掉价值函数 "), link(grpo)

    text("挑战：")
    text("- RL 算法不稳定且难以调参")
    text("- 在大规模下，这需要大量新基础设施（异步 rollout 的推理）")
    text("- 需要不断在系统效率与在策略（?-policy）程度之间做权衡")

    text("## 作业 5（对齐）")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment5-alignment"), link(title="2025 年春季 PDF", url="https://github.com/stanford-cs336/assignment5-alignment/blob/spring2025/cs336_spring2025_assignment5_alignment.pdf")
    text("- 实现直接偏好优化（DPO）")
    text("- 实现组相对偏好优化（GRPO）")


############################################################
# 分词

def tokenization():
    text("本单元的灵感来自 Andrej Karpathy 关于分词的视频，值得一看！ "), video_link("https://www.youtube.com/watch?v=zduSFxRajkE")

    intro_to_tokenization()
    tokenization_examples()
    character_tokenizer()
    byte_tokenizer()
    word_tokenizer()
    bpe_tokenizer()

    text("总结：")
    text("- 分词器：字符串 ↔ token（整数索引）")
    text("- 基于字符、字节、单词的分词都是很不理想的")
    text("- BPE 是一种有效的、数据驱动的启发式方法")
    text("- 分词目前是独立的步骤，也许有一天能从字节开始端到端完成……")

    text("但无论什么方案都需要满足：")
    text("1. 模型（例如 Transformer）应该在序列（文本、视频、DNA 等）的块（抽象单元）上操作")
    text("2. 块应该是可变长度的（给有趣的块分配更多模型容量）")


class CharacterTokenizer(Tokenizer):
    """把字符串表示为 Unicode 码点序列。"""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """把字符串表示为字节序列。"""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  # @inspect string_bytes
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """返回 `indices`，但把所有出现的 `pair` 替换为 `new_index`。"""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


@dataclass(frozen=True)
class BPETokenizerParams:
    """指定一个 BPETokenizer 所需的全部信息。"""
    vocab: dict[int, bytes]     # 索引 -> 字节
    merges: dict[tuple[int, int], int]  # 索引1,索引2 -> 新索引



class BPETokenizer(Tokenizer):
    """给定一组合并规则和词表的 BPE 分词器。"""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # 注意：这是一个非常慢的实现
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)  # @stepover
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


def get_compression_ratio(string: str, indices: list[int]) -> float:  # @inspect string indices
    """给定已被分词为 `indices` 的 `string`，返回每个 token 对应的 UTF-8 字节数。"""
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)                       # @inspect num_tokens
    return num_bytes / num_tokens


def get_gpt5_tokenizer():
    # 代码：https://github.com/openai/tiktoken
    return tiktoken.get_encoding("o200k_base")


def intro_to_tokenization():
    text("原始文本通常表示为 Unicode 字符串。")
    string = "Hello, 🌍! 你好!"

    text("语言模型在 token 序列上定义概率分布（通常用整数索引表示）。")
    indices = [15496, 11, 995, 0]

    text("所以我们需要一个把字符串*编码*为 token 的过程。")
    text("我们还需要一个把 token *解码*回字符串的过程。")
    text("一个 "), link(Tokenizer), text(" 就是实现了 encode 和 decode 方法的类。")


def tokenization_examples():
    text("想感受一下分词器的工作方式，可以玩玩这个 "), link(title="交互式网站", url="https://tiktokenizer.vercel.app/?encoder=gpt2")

    text("## 观察")
    text("- 一个单词及其前面的空格属于同一个 token（例如 \" world\"）。")
    text("- 同一个词在开头和中间的表示不同（例如 \"hello hello\"）。")
    text("- 数字每几位被切成一个 token。")

    text("下面是 OpenAI 的 GPT-5 分词器（tiktoken）的实际效果。")
    tokenizer = get_gpt5_tokenizer()  # @stepover
    string = "Hello, 🌍! 你好!"  # @inspect string

    text("验证 encode() 和 decode() 可以往返还原：")
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string

    text("压缩比：每个 token 对应的字节数")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    text("压缩比越大，序列越短（这是好事，因为注意力对序列长度是平方级的）。")
    text("可以通过增大**词表规模**来提高压缩比（可能的 token 取值增多），但这会导致稀疏性。")
    vocabulary_size = tokenizer.n_vocab  # @inspect vocabulary_size

    text("我们来看看真实的词表： "), link(title="词表", url=get_local_url("var/gpt5_tokenizer_vocab.txt"))
    output_tokenizer(tokenizer, "var/gpt5_tokenizer_vocab.txt")  # @stepover


def output_tokenizer(tokenizer, path: str):
    """把 `tokenizer` 的词表逐行写入 `path`。"""
    if not os.path.exists(path):
        vocab = [b.decode("utf-8", errors="replace") for b in tokenizer.token_byte_values()]
        with open(path, "w") as f:
            for token in vocab:
                f.write(token + "\n")


def character_tokenizer():
    text("Unicode 字符串是 Unicode 字符的序列。")
    text("每个字符可以通过 `ord` 转换为码点（整数）。")
    assert ord("a") == 97
    assert ord("🌍") == 127757
    text("也可以通过 `chr` 转换回来。")
    assert chr(97) == "a"
    assert chr(127757) == "🌍"

    text("现在我们来构建一个 `Tokenizer`，并确保它能往返还原：")
    tokenizer = CharacterTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # 调用 ord @inspect indices @stepover
    reconstructed_string = tokenizer.decode(indices)  # 调用 chr @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    text("Unicode 字符大约有 15 万个。 "), link(title="维基百科", url="https://en.wikipedia.org/wiki/List_of_Unicode_characters")
    vocabulary_size = max(indices) + 1  # 这是下界 @inspect vocabulary_size
    text("问题 1：词表非常庞大。")
    text("问题 2：很多字符相当罕见（例如 🌍），对词表的利用效率很低。")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio @stepover
    text("这个分词器集两家之短（词表大，压缩比又低）。")


def byte_tokenizer():
    text("Unicode 字符串可以表示为字节序列，每个字节可由 0 到 255 之间的整数表示。")
    text("最常见的 Unicode 编码是 "), link(title="UTF-8", url="https://en.wikipedia.org/wiki/UTF-8")

    text("有些 Unicode 字符只用一个字节表示：")
    assert bytes("a", encoding="utf-8") == b"a"
    text("另一些则需要多个字节：")
    assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"

    text("现在我们来构建一个 `Tokenizer`，并确保它能往返还原：")
    tokenizer = ByteTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices @stepover
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    text("词表非常小巧：一个字节可以表示 256 个值。")
    vocabulary_size = 256  # @inspect vocabulary_size
    text("那压缩比呢？")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio @stepover
    assert compression_ratio == 1
    text("压缩比很糟糕，这意味着序列会太长。")
    text("考虑到 Transformer 的上下文长度有限（注意力是平方级的），这看起来不太妙……")


def word_tokenizer():
    text("另一种方法（更接近传统 NLP 的做法）是把字符串切分成单词。")
    string = "I'll say supercalifragilisticexpialidocious!"

    chunks = regex.findall(r"\w+|.", string)  # @inspect chunks
    text("这个正则表达式把所有字母数字字符保持在了一起（单词）。")

    text("要把它变成一个 `Tokenizer`，我们需要把这些块映射成整数。")
    text("然后，我们可以建立从每个块到整数的映射。")

    text("好处：每个 token 都是有意义的（毕竟单词是人类发明的）。")

    vocabulary_size = "训练数据中不同块的数量"
    compression_ratio = get_compression_ratio(string, chunks)  # @inspect compression_ratio @stepover
    text("压缩比不错，但词表规模可能非常巨大。")

    text("此外：")
    text("- 很多词很罕见，模型从中学不到多少东西。")
    text("- 这种方法显然无法提供固定的词表规模。")
    text("- 训练时没见过的新词会被赋予特殊的 UNK token，这很丑陋，还会扰乱困惑度的计算。")


def bpe_tokenizer():
    text("## 字节对编码（BPE）")
    text("BPE 算法由 Philip Gage 于 1994 年提出，用于数据压缩。 "), article_link("http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM")
    text("后来被引入 NLP，用于神经机器翻译。 "), link(sennrich_2016)
    text("(在此之前的论文都使用基于单词的分词。)")
    text("随后 GPT-2 采用了 BPE。 "), link(gpt2_2019)

    text("基本思想：在原始文本上*训练*分词器，构建一个为数据量身定制的词表。")
    text("直觉：常见的字节序列用单个 token 表示，罕见的序列用多个 token 表示。")

    text("概要：从每个字节作为一个 token 开始，反复合并最常出现的相邻 token 对。")

    text("## 训练分词器")
    string = "the cat in the hat"  # @inspect string
    params = train_bpe(string, num_merges=3)

    text("## 使用分词器")
    text("现在，给定一段新文本，我们就可以对其进行编码。")
    tokenizer = BPETokenizer(params)  # @stepover
    string = "the quick brown fox"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    text("在作业 1 中，你将在以下几个方面超越这个实现：")
    text("- encode() 目前会遍历所有合并规则。只需遍历真正相关的合并规则。")
    text("- 检测并保留特殊 token（例如 <|endoftext|>）。")
    text("- 使用预分词（例如 GPT-2 分词器的正则表达式）。")
    text("- 尽量让实现跑得尽可能快。")


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    text("从 `string` 的字节列表开始。")
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # 索引1, 索引2 => 合并后的索引
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # 索引 -> 字节

    for i in range(num_merges):
        # 统计每对相邻 token 出现的次数
        counts = count_adjacent_pairs(indices)  # @inspect counts @stepover

        # 找出最常见的一对
        pair = max(counts, key=counts.get)  # @inspect pair

        # 合并这一对
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices @stepover

    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio

    return BPETokenizerParams(vocab=vocab, merges=merges)


def count_adjacent_pairs(indices: list[int]) -> dict[tuple[int, int], int]:
    """返回一个字典，把 `indices` 中每对相邻 token 映射到其出现的次数。"""
    counts = defaultdict(int)
    for index1, index2 in zip(indices, indices[1:]):
        counts[(index1, index2)] += 1
    return counts


if __name__ == "__main__":
    main()
