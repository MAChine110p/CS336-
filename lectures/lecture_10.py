from dataclasses import dataclass

from sympy import symbols, oo
from edtrace import text, link, image
from lecture_util import article_link
from references import Reference, gqa_2023, mla_2024, longformer_2020, sparse_transformer_2019, mistral_7b_2023, deepseek_v4_2026

# Define symbols corresponding ? the ?? ? the Transformer ??
B, S, T, D, F, N, K, H, L, V = symbols("B S T D F N K H L V", positive=True)
c = symbols("c", positive=True)  # Just a constant that helps with taking limits
memory_bandwidth = symbols("memory_bandwidth", positive=True)

scaling_book_transformers = Reference(title="Scaling book chapter on Transformers", url="https://jax-ml.github.io/scaling-book/transformers/")
scaling_book_inference = Reference(title="Scaling book chapter on inference", url="https://jax-ml.github.io/scaling-book/inference/")

def main():
    text("## Lecture 10: ??")
    image("images/inference-schema.png", width=600)

    text("### ?? the ?? ????")
    landscape()
    review_transformer()
    review_of_arithmetic_intensity()
    arithmetic_intensity_of_inference()
    throughput_and_latency()

    text("### Taking shortcuts (lossy)")
    reduce_kv_cache_size()
    quantization()
    model_pruning()

    text("??: ?? ?? complexity ??? hurting accuracy")

    text("???? recipe:")
    text("1. Define ?? ?? ??")
    text("2. ?? ?? ??")

    text("Distillation recipe:")
    text("1. Define ?? ?? ??")
    text("2. ??? weights ?? original ?? (?? ? a ?? ??)")
    text("3. Repair ?? ?? (distillation)")

    text("### ?? shortcuts ?? double ?? (lossless)")
    speculative_sampling()

    text("### Handling dynamic ????")
    text("Batching ?? ?? ? ?? traffic ? tricky ??:")
    text("1. ?? arrive at ?? times (waiting ?? ?? ? ? ?? early ??)")
    text("2. ?? ? ?? prefixes (e.g., system ???, generating multiple samples)")
    text("3. ?? ? ?? ?? (padding ? inefficient)")

    continuous_batching()
    paged_attention()

    text("### ??")
    text("- ?? ? ?? (actual ??, ??, ?? ??)")
    text("- ?? characteristics compared ? ?? (??-bound, dynamic)")
    text("- Techniques: ?? ??, quantization, pruning/distillation, speculative sampling")
    text("- ?? ? systems (speculative execution, paging)")
    text("- ?? ?? ? huge potential ?? improvement")


def landscape():
    text("?? ?? ? ? ?? places:")
    text("- Actual ?? (chatbots, ?? completion, agents, ?? ?? processing)")
    text("- ?? ?? (e.g., ? ?? following)")
    text("- ?? ?? (?? ?? generations, ?? ?? ??)")

    text("??? **efficiency** matters: ?? ? ??-?? cost, ?? ? repeated ?? times")
    text("- OpenAI ?? ~8.6T token ? day "), article_link("https://www.pymnts.com/artificial-intelligence-2/2025/openai-bests-google-in-race-for-consumer-ai-token-consumption/")
    text("- ?? ??, DeepSeek v4 ? ?? ? 32T token "), link(deepseek_v4_2026)

    text("Moreover:")
    text("- Chatbots: most token ? meant ?? human consumption (humans ? bottleneck)")
    text("- Agents: ?? → internal trace → ?? ?? human (?? ? token generated ?? grow unbounded)")
    text("- token generated = ?? spent")

    text("Companies doing ?? (a ?? deal ?? anyone ? ? a product ? platform):")
    text("- Providers serving closed ?? (OpenAI, Anthropic, Google, etc.)")
    text("- Providers serving open-weight ?? (??, Fireworks, Baseten, DeepInfra, Groq, Cerebras, etc.)")

    text("Open-source packages:")
    text("- vLLM: ? Berkeley, pioneered PagedAttention, popular ?? ? default "), link(title="GitHub", url="https://github.com/vllm-project/vllm")
    text("- SGLang: ? Berkeley, pioneered RadixAttention, ? ?? agentic ???? "), link(title="project", url="https://sgl-project.github.io/")
    text("- TensorRT-LLM: ? NVIDIA, highly optimized ?? GPUs "), article_link("https://nvidia.github.io/TensorRT-LLM/overview.html")
    text("- llama.cpp: C++ ?, supports CPU ??, ?? locally "), link(title="GitHub", url="https://github.com/ggml-org/llama.cpp")

    text("?? ? huge. ?? ? ? it ??.")

    text("?? ? \"??\" ?? (metrics)?")
    text("- ??-?-??-token (TTFT): ?? long ?? ?? ?? ?? ?? ?? (?? ??? ??)")
    text("- ?? (?/token): ?? ?? token ?? ?? *??* ?? (?? ??? ??)")
    text("- Throughput (token/?): ?? ?? token ?? ?? *??* ?? (?? ?? processing)")

    text("?? governs efficiency?")
    text("- ?? (??): you ?? ?? token, ?? parallelize ?? ?? (matmul ? Transformer)")
    text("- ??: you ? ? ?? sequentially, ??'t parallelize ?? ??, ?? harder ? fully utilize ??")


def review_transformer():
    link(scaling_book_transformers)
    text("Notation (similar ? einops):")
    text("- Symbols denote ?? (?? their ??): B (??), T (??), D (?? dim), H (? dim)")
    text("- ??: BT<font color=\"red\">D</font> x <font color=\"red\">D</font>H → BTH")
    text("- <font color=\"red\">Contracting (red)</font> ?? ?? ? ?? operands ?? disappear ? ??")
    text("- Regular (black) ?? ?? ? ?? operand ?? stay ? ??")
    text("- ??: <font color=\"blue\">B</font><font color=\"red\">D</font> x <font color=\"blue\">B</font><font color=\"red\">D</font> → B")
    text("- <font color=\"blue\">Batching (blue)</font> ?? ?? ? ?? operands ?? stay ? ??")

    image("https://jax-ml.github.io/scaling-book/assets/img/transformer-diagram.png", width=800)
    text("Conventions:")
    text("- F = 4 D (MLP ?-projects ?? 4x the ?? ??)")
    text("- D = N H (?? ?? ?? ?? N ?)")
    text("- N = K G (?? GQA, ?? ? ? ?? ?? K groups)")
    text("- S = T (?? ??, condition ? S ?? token ? ?? T ?? token)")


def review_of_arithmetic_intensity():
    text("??: multiply X <font color=\"gray\">(B x D)</font> ?? W <font color=\"gray\">(D x F)</font> ??")
    text("Intuition: B ? ?? ??, D ? ?? ??, F ? ?-projection ?? ? MLP")

    text("?'s ? FLOPs ?? ?? ??/?? accounting ?? the ???? (X * W).")
    flops = 0
    bytes_transferred = 0

    # ?? the ??
    text("1. ?? X <font color=\"gray\">(B x D)</font> ? HBM")
    bytes_transferred += 2*B*D   # 2 bytes for bf16
    text("2. ?? W <font color=\"gray\">(D x F)</font> ? HBM")
    bytes_transferred += 2*D*F
    text("3. ?? Y = X <font color=\"gray\">(B x D)</font> @ W <font color=\"gray\">(D x F)</font>")
    flops += 2*B*D*F
    text("4. ?? Y <font color=\"gray\">(B x F)</font> ? HBM")
    bytes_transferred += 2*B*F

    assert flops == 2*B*D*F
    assert bytes_transferred == 2*B*D + 2*D*F + 2*B*F

    text("Recall ? **arithmetic intensity** ? ?? ?? ?? we ? ? ?? transferred (?? ? ? ?).")
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity

    text("Assuming B ? ?? ?? ? D ?? F, ?? ???? simplify:")
    intensity = intensity.subs(D, c*B).subs(F, c*B).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B

    text("Accelerator intensity ? H100:")
    flops_per_second = 989e12
    memory_bandwidth = 3.35e12
    accelerator_intensity = flops_per_second / memory_bandwidth  # @inspect accelerator_intensity
    assert round(accelerator_intensity) == 295

    text("?? ?? intensity > accelerator intensity, **??-bound** (?)")
    text("?? ?? intensity < accelerator intensity, **??-bound** (?)")
    text("??: ??-bound iff B > 295")

    text("Extreme case (B = 1, corresponding ? ??-?? product):")
    text("- Arithmetic intensity: 1")
    text("- ??-bound (?? D x F ??, ?? ? 2 D F FLOPs)")
    text("- ?? basically ?? ?? ? ??...")


def arithmetic_intensity_of_inference():
    link(scaling_book_inference)

    image("https://jax-ml.github.io/scaling-book/assets/img/naive-inference-1400.webp", width=800)
    text("Naive ??: ? ?? ?? token, feed history ?? Transformer")
    text("Complexity: generating T token ?? O(T^3) FLOPs (?? feedforward ?? ? O(T^2))")

    text("Observation: a lot ? the ?? ?? ? ?? ?? prefixes")
    text("Solution: ?? **KV cache** ? HBM")
    image("https://jax-ml.github.io/scaling-book/assets/img/cached-inference-1400.webp", width=800)
    text("KV cache: ?? ?? ?? (B), token (S), ? (L), ? (K), ?? an H-dimensional ??")

    text("?? stages ? ??:")
    text("1. **Prefill**: ?? a ???, encode ?? vectors (parallelizable ?? ? ??)")
    text("2. **??**: ?? ?? ?? token (??)")

    text("?'s ?? the FLOPs ?? ?? IO ?? ?? the MLP ?? ??? ?.")
    text("S ? the ?? ? token we're conditioning ?, T ? the ?? ? token we're generating.")
    text("Later, we'll specialize ? prefill (T = S) ?? ?? (T = 1).")

    text("### MLP ? (? looking at the ?? multiplications)")
    flops = 0
    bytes_transferred = 0

    # ?? the ??
    text("1. ?? X <font color=\"gray\">(B x T x D)</font> ? HBM")
    bytes_transferred += 2*B*T*D
    text("2. ?? Wup <font color=\"gray\">(D x F)</font>, Wgate <font color=\"gray\">(D x F)</font>, Wdown <font color=\"gray\">(F x D)</font> ? HBM")
    bytes_transferred += 3 * 2*D*F
    text("3. ?? U = X <font color=\"gray\">(B x T x D)</font> @ Wup <font color=\"gray\">(D x F)</font>")
    flops += 2*B*T*D*F
    text("4. ?? U <font color=\"gray\">(B x T x F)</font> ? HBM")
    bytes_transferred += 2*B*T*F
    text("5. ?? G = X <font color=\"gray\">(B x T x D)</font> @ Wgate <font color=\"gray\">(D x F)</font>")
    flops += 2*B*T*D*F
    text("6. ?? G <font color=\"gray\">(B x T x F)</font> ? HBM")
    bytes_transferred += 2*B*T*F
    text("7. ?? Y = GeLU(G)*U <font color=\"gray\">(B x T x F)</font> @ Wdown <font color=\"gray\">(F x D)</font>")
    flops += 2*B*T*D*F
    text("8. ?? Y <font color=\"gray\">(B x T x D)</font> ? HBM")
    bytes_transferred += 2*B*T*D

    assert flops == 6*B*T*D*F
    assert bytes_transferred == 4*B*T*D + 4*B*T*F + 6*D*F

    # ?? the arithmetic intensity
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    text("?? ? B*T ? ?? smaller ? D ?? F.")
    intensity = intensity.subs(D, c*B*T).subs(F, c*B*T).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B*T

    text("?? the ?? stages:")
    text("1. Prefill: ?? ? ? ??-bound (?) ?? making `B*T` ?? enough (?? ??, long ??)")
    text("2. ??: ?? problems")
    text("- Generating ?? token at a ?? (T = 1)")
    text("- B ? ?? ? concurrent ??, unpredictable ?? ??? ??")

    text("### ??? ? (focusing ? the ?? multiplications ? FlashAttention)")
    text("- S ? ?? ? ?? token (?? generated)")
    text("- T ? ?? ? ??? token (? ?? logits ??)")
    flops = 0
    bytes_transferred = 0

    # ?? the ??
    text("1. ?? Q <font color=\"gray\">(B x T x D)</font>, K <font color=\"gray\">(B x S x D)</font>, V <font color=\"gray\">(B x S x D)</font> ? HBM")
    bytes_transferred += 2*B*T*D + 2*B*S*D + 2*B*S*D
    text("2. ?? A = Q <font color=\"gray\">(B x T x D)</font> @ K <font color=\"gray\">(B x S x D)</font>")
    flops += 2*B*S*T*D
    text("3. ?? Y = softmax(A) <font color=\"gray\">(B x S x T x K x G)</font> @ V <font color=\"gray\">(B x S x K x H)</font>")
    flops += 2*B*S*T*D
    text("4. ?? Y <font color=\"gray\">(B x T x D)</font> ? HBM")
    bytes_transferred += 2*B*T*D

    assert flops == 4*B*S*T*D
    assert bytes_transferred == 4*B*S*D + 4*B*T*D

    # ?? the arithmetic intensity
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    assert intensity == S*T / (S + T)

    text("?? the ?? stages:")
    text("1. Prefill: T = S")
    prefill_intensity = intensity.subs(T, S).simplify()  # @inspect prefill_intensity
    assert prefill_intensity == S/2  # Good!
    text("2. ??: T = 1")
    generate_intensity = intensity.subs(T, 1).simplify()  # @inspect generate_intensity
    assert generate_intensity < 1  # Bad!

    text("Unlike MLPs, no dependence ? B, ?? batching doesn't help!")
    text("????")
    text("- ? MLP ?, ?? ?? hits the ?? MLP weights (Wup, Wgate, Wdown don't depend ? B)")
    text("- ? ??? ?, ?? ?? ? its own KV cache vectors (Q, K, V ?? depend ? B)")

    text("??:")
    text("- Prefill ? ??-bound, ?? ? ??-bound")
    text("- Prefill MLP intensity: `B*S`")
    text("- Prefill ??? intensity: `S/2`")
    text("- ?? MLP intensity: `B` (?? concurrent ??)")
    text("- ?? ??? intensity: `<1` (impossible ? ??)")


@dataclass(frozen=True)
class TransformerPerformanceStats:
    """
    Performance stats of a Transformer:
    - num_params: number of parameters (in bytes)
    - memory: total memory usage (parameters + KV cache) in bytes
    - latency: time to generate one token (seconds/token)
    - throughput: tokens generated per second
    """
    num_params: int
    memory: int
    latency: float
    throughput: float

    def substitute(self, key, value):
        """Substitute `key` with `value` in all stats."""
        return TransformerPerformanceStats(
            self.num_params.subs(key, value).simplify(),
            self.memory.subs(key, value).simplify(),
            self.latency.subs(key, value).simplify(),
            self.throughput.subs(key, value).simplify(),
        )


def compute_transformer_performance_stats(config) -> TransformerPerformanceStats:  # @inspect config
    """Compute various performance stats for the Transformer given `config`."""

    # ?? ? ?? ? the Transformer
    num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L

    # How much ?? the ?? take
    parameter_size = 2*num_params  # 2 for bf16 (training requires a larger multiple)

    # How much the KV cache takes per ?? (S token, K heads, H head dim, L ?)
    kv_cache_size_per_seq = S * (K*H) * L * 2 * 2  # 2 for key + value, 2 for bf16

    # ?? ?? usage
    memory = B * kv_cache_size_per_seq + parameter_size

    # *Latency* ? determined ?? ?? IO (?? ?? ?? ?? KV cache ?? ?? ??)
    latency = memory / memory_bandwidth

    # *Throughput* ? the inverse ? latency, ?? we're generating B token ? ??
    throughput = B / latency

    # Substitute config
    num_params = num_params.subs(config).simplify()  # @inspect num_params
    memory = memory.subs(config).simplify()  # @inspect memory
    latency = latency.subs(config).simplify()  # @inspect latency
    throughput = throughput.subs(config).simplify()  # @inspect throughput

    return TransformerPerformanceStats(num_params, memory, latency, throughput)


def llama2_13b_config(args={}):
    return {
        S: 1024,   # Sequence length
        D: 5120,   # Model dim
        F: 13824,  # Feed-forward dim
        N: 40,     # Number of query heads
        K: 40,     # Number of key/value heads
        H: 128,    # Head dimension
        L: 40,     # Number of layers
        V: 32000,  # Vocabulary size
        memory_bandwidth: 3.35e12,  # Memory bandwidth (bytes/second)
        **args
    }


def throughput_and_latency():
    text("?? we ? shown ? ?? ? ??-bound.")
    text("? us ?? ?? the theoretical maximum ?? ?? throughput ? a single ??.")
    text("Assumption: ?? overlap ?? ?? ?? perfectly ?? ignore overhead.")

    text("Instantiate ?? ?? throughput ?? Llama 2 13B ? an H100:")
    config = llama2_13b_config()
    stats = compute_transformer_performance_stats(config)

    # ?? ?? 1
    b1 = stats.substitute(B, 1)  # @inspect b1 @stepover

    # ?? ?? 64
    b64 = stats.substitute(B, 64)  # @inspect b64 @stepover
    text("??: ?? ??, ?? throughput")

    # ?? ?? 256
    b256 = stats.substitute(B, 256)  # @inspect b256 @stepover
    text("??: even ?? ??, even ?? throughput")
    h100_memory = 80e9  # H100 memory in bytes
    assert b256.memory > h100_memory  # Doesn't fit in memory!
    text("??: doesn't fit ?? ?? ?? throughput gains ? diminishing ?...")

    text("?? increasing ?? ?? ?:")
    text("- Worsens ?? ?? larger KV cache (O(B) ??) ? ??/??")
    text("- Improves throughput ?? amortizes the cost ? reading ??")

    text("**Tradeoff** ?? ?? ?? throughput:")
    text("1. Smaller ?? sizes yield ?? ?? ?? ?? throughput")
    text("2. Larger ?? sizes yield ?? throughput ?? ?? ??")

    text("?? ??: ?? you launch M copies ? the ??, ?? ? the ??, throughput increases ?? M!")
    text("Harder ??: shard the ?? ?? the KV cache "), link(scaling_book_inference)

    text("??: ??-?-??-token (TTFT) ? essentially a ?? ? prefill ??")
    text("?? smaller ?? sizes ?? prefill ?? ?? TTFT")
    text("?? larger ?? sizes ?? ?? ? ?? throughput")


def reduce_kv_cache_size():
    text("Recall ? ?? ? the bottleneck ?? ??.")
    text("?? ?'s ?? ? ?? the ?? ? the KV cache")
    text("...?? ? sure we don't lose ? ?? accuracy.")

    text("### Grouped-?? ??? (GQA) "), link(gqa_2023)
    image("https://jax-ml.github.io/scaling-book/assets/img/gmqa.png", width=800)
    text("??: N ?? ?, ?? ? K key ?? ? ?, ?? interacting ? N/K ?? ?")
    text("Multi-headed ??? (MHA): K=N")
    text("Multi-?? ??? (MQA): K=1")
    text("Group-?? ??? (GQA): K ? somewhere ? ??")

    text("??/throughput improves: "), link(gqa_2023)
    image("images/gqa-speed.png", width=500)

    text("??? ? GQA ?? ?? ?? throughput?")
    text("GQA reduces the KV cache ?? a factor ? N/K.")
    text("Reminder: reducing ?? usage leads ? speedup (since we're ??-bound).")

    # Original Llama 2 13B (MHA)
    config = llama2_13b_config({K: 40, B: 64})  # @stepover
    k40_b64 = compute_transformer_performance_stats(config)  # @inspect k40_b64 @stepover

    # GQA ? 1:5 ratio (K:N)
    config = llama2_13b_config({K: 8, B: 64})  # Use GQA with 1:5 ratio @stepover
    k8_b64 = compute_transformer_performance_stats(config)  # @inspect k8_b64 @stepover
    text("??: ?? ??, ?? ?? throughput (?? it fits ? ?? ??!)")

    # ?? ???? ?? the ?? ??
    config = llama2_13b_config({K: 8, B: 256})  # Increase batch size @stepover
    k8_b256 = compute_transformer_performance_stats(config)  # @inspect k8_b256 @stepover
    text("??: ?? ??, ?? ?? throughput (?? ?? fits ? ??!)")

    text("?? ? accuracy doesn't drop: "), link(gqa_2023)
    image("images/gqa-accuracy.png", width=800)

    text("### Multi-? latent ??? (MLA) "), link(mla_2024)
    image("images/mla-schema.png", width=800)
    text("Normal ???: KV cache consists ? K = W_K h, V = W_V h (N*H ??)")
    text("MLA: ?? compressed ?? c = W_c h (C ??), project ? ? K = W_K c, V = W_V c ?? needed")
    text("DeepSeek v2: ?? N*H = 16384 ? C = 512")
    text("Wrinkle: MLA ? ? compatible ? RoPE, ?? ?? ? add additional 64 ?? ?? RoPE, ?? 512 + 64 = 576 ?? ??")
    text("??/throughput improvements follow similarly ? the KV cache reduction ?? argued earlier")

    text("?'s ?? ?? the accuracy.")
    text("??, MHA ? ?? ? GQA (?? ?? ??) [? 8] "), link(mla_2024)
    image("images/mla-accuracy.png", width=800)
    text("?, MLA ? even a bit ?? ? MHA (?? ?? cheaper) [? 9] "), link(mla_2024)
    image("images/mla-accuracy2.png", width=800)

    text("### Cross-? ??? (CLA) "), link("https://arxiv.org/abs/2405.12981")
    image("images/cla-diagram.png", width=500)
    text("??: share KVs ?? **?** (?? ?? GQA shares KVs ?? ?)")
    text("Empirically improves the pareto frontier ? accuracy ?? KV cache ?? (?? ?? throughput)")
    image("images/cla-results.png", width=700)

    text("### ?? (sliding window) ??? "), link(longformer_2020), link(sparse_transformer_2019), link(mistral_7b_2023)
    image("images/longformer-attention.png", width=800)
    text("??: ?? ?? at the ?? ???, ?? ? most relevant ?? modeling")
    text("Effective ??? scales linearly ? the ?? ? ?")
    text("KV cache ? independent ? ?? ??!")
    text("Problem: this ?? ?? hurt accuracy")
    text("Solution: interleave ?? ??? ? ?? ??? (hybrid ?)")

    text("### DeepSeek v4 ???")
    text("- Supports 1M ??? ?? "),  link(deepseek_v4_2026)
    image("images/deepseek-v4-attention.png", width=800)
    text("- Compressed Sparse ??? (CSA): compresses ?? m token ?? 1")
    text("- DeepSeek Sparse ??? (DSA): selects the top k")
    text("- Heavily Compressed ??? (HCA): compresses even ??")

    text("??:")
    text("- Goal: ?? the KV cache ?? (since ?? ? ??-bound) ??? hurting accuracy")
    text("- ??-dimensional KV cache (GQA, MLA, CLA)")
    text("- ?? ??? (truncates the KV cache) ? ?? ? the ?")
    text("- ?? ??: linear ??? / state-space-?? (Mamba 2, GatedDeltaNet), diffusion ??")


def quantization():
    text("????: ?? the precision ? numbers")
    text("?? ?? ??? ?? ??/throughput (since ?? ? ??-bound).")
    text("? course we ? ? worry ?? accuracy...")

    # Mechanics
    x = 5.2342  # @inspect x
    scale = 0.1
    zero_point = 4
    x_quant = round(x / scale) + zero_point  # Quantize  @inspect x_quant
    x_approx = (x_quant - zero_point) * scale  # Dequantize @inspect x_approx

    image("https://www.datocms-assets.com/104802/1709770809-twitter-post-20.png", width=400), article_link("https://www.baseten.co/blog/fp8-efficient-model-inference-with-8-bit-floating-point-numbers/")
    text("- fp32 (4 ??): needed ?? ?? ?? ??? states ?? ??")
    text("- bf16 (2 ??): default ?? ??")
    text("- fp8 (1 ??) [-240, 240] ?? e4m3 ? H100s: ?? ?? ?? you dare "), link("https://arxiv.org/pdf/2310.18313")
    text("- int8 (1 ??) [-128, 127]: ?? accurate ?? cheaper ? fp8, ?? ?? ?? ? "), link("https://arxiv.org/pdf/2303.17951")
    text("- int4 (0.5 ??) [-8, 7]: cheaper, even ?? accurate "), link("https://arxiv.org/pdf/2303.17951")

    link(title="Overview of approaches", url="https://apxml.com/posts/llm-quantization-techniques-explained")

    text("Quantization-aware ?? (QAT)")
    text("- ?? ??, quantize-??-dequantize ?? the ???? ? simulate quantization ??")
    text("- Pro: weights ? ?? ? ?? ? quantization")
    text("- Con: ?? ?? ??-?? ??")

    text("Post-?? quantization (PTQ):")
    text("- Done ?? ??, ?? ?? cheaper")
    text("- ?? ? ?? ?? ? determine ?? ?? zero point ?? ?? ? ? tensor")
    text("- GPTQ: ?? Hessian information ? update non-quantized weights ? account ?? quantization ?? "), link("https://arxiv.org/abs/2210.17323")

    text("### ???-aware quantization (AWQ)")
    link("https://arxiv.org/abs/2306.00978")
    text("- Observation: ?? ??? channels ? ??")
    text("- Weights ? hit those matter ??")
    text("- Allocate ?? precision ? those weights")
    text("- ??: ?? ?? weights (0.1-1%) ? ?? ? ? precision based ? ???")
    text("- fp16 → int3 ?? 4x ?? ??, 3.2x speedup")
    image("images/awq-schema.png", width=800)


def model_pruning():
    text("????: ?? rip ? parts ? an ?? ?? ? ? it cheaper")
    text("...?? ?? fix it ?.")

    text("?? ? NVIDIA "), link("https://arxiv.org/abs/2407.14679")
    image("images/pruning-kd-loop.png", width=600)
    text("Algorithm:")
    text("1. Identify ?? {?, ?, ?? ??} ? a ?? calibration ??? (1024 samples)")
    text("2. Remove unimportant ? ? ?? a smaller ??")
    text("3. Distill the original ?? ?? pruned ??")

    text("??:")
    image("images/pruning-kd.png", width=500)

    # TODO


def speculative_sampling():
    text("Recall the ?? stages ? ??:")
    text("- Prefill: ?? a ??, encode token ? ?? (??-bound) [??: ? gives you ??]")
    text("- ??: ?? ?? token at a ?? (??-bound)")
    text("????, ?? ? ?? ? ??.")

    text("Speculative sampling "), link("https://arxiv.org/abs/2211.17192"), link("https://arxiv.org/abs/2302.01318")
    text("- ?? a cheaper **draft ??** p ? guess a ?? token (e.g., 4)")
    text("- ?? ? ?? ?? q (?? token ? ??), ?? accept ?? it looks ?")
    link(title="Speculative sampling video", url="https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-1-Illustration.mp4")
    article_link("https://research.google/blog/looking-back-at-speculative-decoding/")

    image("images/speculative-sampling-algorithm.png", width=600)
    text("?? modified rejection sampling ? proposal p ?? ?? q")
    text("Modification: ?? ?? at least ?? candidate (rejection sampling ? ?? looping)")
    text("Key property: guaranteed ? ? an **?? ??** ? the ?? ??!")

    text("Proof ?? ??: ?? ?? vocabulary ?? {A, B}")
    text("- ?? ?? ??: [q(A), q(B)]")
    text("- Draft ?? ??: [p(A), p(B)]")
    text("- ?? p(A) > q(A) [draft ?? oversamples A].")
    text("- ?? p(B) < q(B) [draft ?? undersamples B].")
    text("- Residual ?? max(q-p, 0): [0, 1]")
    text("?? the ?? ? speculatively sampling a token:")
    text("- P[sampling A] = p(A) * (q(A) / p(A)) + p(B) * 1 * 0 = q(A)")
    text("- P[sampling B] = p(B) * 1 + p(A) * (1 - q(A) / p(A)) * 1 = q(B)")

    image("images/speculative-sampling-results.png", width=600)
    image("images/speculative-sampling-stats.png", width=600)

    text("???:")
    text("- ?? ?? ? 70B ??, draft ?? ? 8B ??")
    text("- ?? ?? ? 8B ??, draft ?? ? 1B ??")
    text("- ?? ? ? draft ?? ?? close ? ?? (distillation)")

    text("Extensions ? ?? the draft ??:")
    text("- Medusa: draft ?? generates multiple token ? ?? "), link("https://arxiv.org/abs/2401.10774")
    text("- EAGLE: draft ?? takes ?-level features ? ?? ?? "), link("https://arxiv.org/pdf/2401.15077")
    image("images/medusa-eagle.png", width=600)

    text("??:")
    text("- ?? sampling ? ?? ?? (thanks ? math)!")
    text("- Exploits asymmetry ?? ?? ?? ??")
    text("- Lots ? room ?? innovation ? the draft ?? (involves ??)")


def continuous_batching():
    link(title="Orca: A Distributed Serving System for Transformer-Based Generative Models", url="https://www.usenix.org/system/files/osdi22-yu.pdf"), link(title="talk", url="https://www.youtube.com/watch?v=Ob9PPLxETYU")

    text("Problem:")
    text("- ??: ?? a dense ? ? token (?? ?? x ?? ??)")
    text("- ??: ?? arrive ?? finish at ?? times, ?? you ? a ragged array")
    image("https://images.ctfassets.net/xjan103pcp94/1LJioEsEdQQpDCxYNWirU6/82b9fbfc5b78b10c1d4508b60e72fdcf/cb_02_diagram-static-batching.png", width=600)

    text("Solution: ??-level ??")
    text("- Decode ?? ?? ??")
    text("- Add ?? ?? ? the ?? ?? they arrive (?? don't ? ? ?? ?? ?? completes)")

    text("Problem:")
    text("- Batching ? works ?? ?? ?? ? the ?? dimensionality (right?)")
    text("- ?? ?? ?? ?? ? a ?? ??")

    text("Solution: selective batching")
    text("- ??: ?? ?? ?? ? the ?? ??, operate ? a B x S x H tensor")
    text("- ?? we ?? ? ?? ??: [3, H], [9, H], [5, H], etc.")
    text("- ??? ??: ?? ?? ?? ???")
    text("- Non-??? ??: concatenate ?? the ?? ?? ? [3 + 9 + 5, H]")


def paged_attention():
    text("?? ? introduced vLLM ? addition ? PagedAttention "), link("https://arxiv.org/pdf/2309.06180.pdf")

    text("?? status quo:")
    text("- ?? comes ?")
    text("- Allocate section ? KV cache ?? ??? ?? ?? (? ? a max ??)")
    image("images/paged-attention-fragmentation.png", width=800)
    text("Problem: fragmentation (?? ?? ? your ?? drive)")
    text("- ?? ?? wasteful since we ?? ?? ?? fewer token (internal fragmentation)!")
    text("- ?? ? extra unused space ?? sections (external fragmentation)!")

    text("Solution: PagedAttention (remember operating systems)")
    text("- Divide the KV cache ? a ?? ?? non-contiguous **?**")
    image("images/paged-attention-blocks.png", width=400)

    text("?? ?? share the KV caches:")
    image("images/paged-attention-logical.png", width=800)

    text("? general, multiple types ? sharing KV caches ?? ??:")
    image("images/paged-attention-sharing.png", width=600)
    text("- Sharing the system ???")
    text("- Sampling multiple ?? ? ??? (e.g., ?? program synthesis)")

    text("Solution: share prefixes, copy-?-?? at the ? level")
    image("images/paged-attention-parallel.png", width=600)

    text("?? vLLM optimizations:")
    text("- ?? ? fuse ? ?? ?? ??? (?? ?? launch overhead)")
    text("- ?? latest ?? (FlashAttention, FlashDecoding)")
    text("- ?? CUDA graphs ? avoid ?? launch overhead")

    text("??: ?? ?? ? operating systems (paging) ? ? ?? ? ?? ?? dynamic ????")


if __name__ == "__main__":
    main()
