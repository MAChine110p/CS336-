import os
import time
from typing import Callable
import torch
from torch.profiler import ProfilerActivity
import triton
import triton.language as tl
from edtrace import text, link, image
from lecture_util import get_local_url
from gpu_util import cuda_if_available


def main():
    text("???? GPU ?????")
    text("??? ????/???? + ????")

    review_of_gpus()
    benchmarking_and_profiling()           # ??????
    naive_vs_builtin_vs_compiled_gelu()    # ????? GeLU ??

    # ?? Triton ??
    triton_introduction()
    triton_gelu_example()      # ?????
    triton_softmax_example()   # ?? (row fits in a block)
    triton_row_sum_example()   # ?? (row doesn't fit in block)
    triton_matmul_relu_example()    # ?????????

    text("???")
    text("- ?????? (PyTorch, Triton, PTX) ? ?? you ???")
    text("- ???? (SMs, warps, occupancy, bank conflicts, etc.) ? optimize ??")
    text("- ???????????")
    text("- ????????????????????")
    text("- Triton?????????????????????????? HBM?")
    text("- ??: GeLU (elementwise), softmax (?-wise), ? sum (baby tiling), matmul (tiling)")

    text("??????? GPU?")


def review_of_gpus():
    text("## ??")
    image("images/gpu-hardware.png", width=800)
    text("| Accelerator                        | A100      | H100      | B200      |", verbatim=True)
    text("+------------------------------------+-----------+-----------+-----------+", verbatim=True)
    text("| # SMs                              |       108 |       132 |       148 |", verbatim=True)
    text("+------------------------------------+-----------+-----------+-----------+", verbatim=True)
    text("| Register size (per SM)             |    256 KB |    256 KB |    256 KB |", verbatim=True)
    text("| L1 cache + shared memory (per SM)  |    192 KB |    256 KB |    256 KB |", verbatim=True)
    text("| L2 cache size                      |     40 MB |     50 MB | 96-126 MB |", verbatim=True)
    text("| HBM size                           |     80 GB |     80 GB |    192 GB |", verbatim=True)
    text("+------------------------------------+-----------+-----------+-----------+", verbatim=True)
    text("| Register bandwidth                 | ~116 TB/s | ~401 TB/s | ~447 TB/s |", verbatim=True)
    text("| L1 cache + shared memory bandwidth |  ~19 TB/s |  ~33 TB/s |  ~19 TB/s |", verbatim=True)
    text("| L2 cache bandwidth                 | ~5-8 TB/s |  ~12 TB/s |   ~9 TB/s |", verbatim=True)
    text("| HBM bandwidth                      |    2 TB/s | 3.35 TB/s |    8 TB/s |", verbatim=True)

    text("?B200 ?? Tensor Core ????????TMEM??????????????????????????")

    text("## ????")
    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=600)
    text("- *??*??????????")
    text("- *???*????????CTA??????")
    text("- *??*???????")

    text("?H100 ? B200 ??????????????????????")

    text("?????????")
    text("??????? (e.g., GeLU), ?? ? most natural: ?? ?? ?? ?? ??.")
    text("- f(i) ?? i = 0, ..., N-1")
    text("??????????? ?? softmax ? ????, ?? ?? ? communicate.")
    text("? HBM ????, ?? ?? ?? ?? (?? ? SM).")
    text("???????? ? ?? the ?? ?? ??.")
    text("??????????????? SM ??")
    text("? Triton ???????????????????????")

    text("## ??????????")
    text("?????????????")
    text("??????????????????????")
    text("??????????????????????????????")

    text("??????????????")

    text("**Warp**?")
    text("- ???????????? warp??? warp ? 32 ?????")
    text("- ??????? 64 ???????? 2 ? warp?")
    text("| TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT | TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT |", verbatim=True)
    text("- ?? warp ????????? SM ?????????????")
    text("- ???????? warp ???????????????? A?else B??????????????")
    text("| AAAAAAAAA....................... |", verbatim=True)
    text("| .........BBBBBBBBBBBBBBBBBBBBBBB |", verbatim=True)
    text("- SM ???? warp????????????????? warp ? HBM ???????")

    text("**?Warp????**?")
    text("- ?? ?? ?? ?? ?? 0 ?? 255 ???.")
    text("- The ?? ??? ?? ??, the fewer ?? ?? ? ?? ? an SM (? occupancy).")
    text("- ? occupancy isn't necessarily ? ?? ?? ?? ? doing ?? ??.")
    text("- ??: ?? coarsening (?? ?? ?? multiple ??).")
    text("- ??: ?? ? ? 64 ??, ?? ?? 160 ???, SM ? 65536 ???")

    # ?? we want ? ??
    num_threads_per_block = 128
    num_registers_per_thread = 160

    # ?? ?? offers
    max_registers = 65536  # Registers allowed per SM
    max_warps = 64         # Concurrent warps allowed per SM

    # ?? ???? ?? at once
    assert num_registers_per_thread <= 255
    num_registers_per_block = num_threads_per_block * num_registers_per_thread  # @inspect num_registers_per_block
    num_blocks = max_registers // num_registers_per_block  # Limited by registers @inspect num_blocks
    num_warps = num_blocks * num_threads_per_block / 32  # @inspect num_warps
    occupancy = num_warps / max_warps  # @inspect occupancy

    text("**bank ??**???????")
    text("- ???????? 32 ? bank???? 4 ???")
    text("B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31", verbatim=True)
    text("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", verbatim=True)
    text("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", verbatim=True)
    text("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", verbatim=True)
    text("- ?? ??, ?? bank ?? ? ? ?? ?? ?? ?? (?? ? the ?? ?? ??).")
    text("- ??????????? bank????????bank ????")
    text("- ?? case ??: ?? ?? ?? ? spans ?? bank; 32 ?? ?? ?? ? ?? ? 32-way bank conflict!")
    text("- Unavoidable: ?? doing matmul A @ B, ?? ? ? A ?? ? ? B")
    text("- ??????? swizzling ??????????? ? xor col???? bank ???")

    text("**??????**?HBM??")
    text("- warp ?? 32 ????? HBM ?????????? 128 ???????????")
    text("M00 M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 M13 M14 M15 M16 M17 M18 M19 M20 M21 M22 M23 M24 M25 M26 M27 M28 M29 M30 M31", verbatim=True)
    text("M32 M33 M34 M35 M36 M37 M38 M39 M40 M41 M42 M43 M44 M45 M46 M47 M48 M49 M50 M51 M52 M53 M54 M55 M56 M57 M58 M59 M60 M61 M62 M63", verbatim=True)
    text("- ?? case: ?? coalescing, ?? ?? ?? the ?? cache line (32 ?? x 4 ?? = 128 ??).")

    text("**? occupancy**:")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2019/06/pasted-image-0.png", width=400)
    text("- ?? ? ?? ? SMs ? waves.")
    text("- B200 ? 148 SMs, ?? we launch 160 ?? ?, ?? wave ? 148 ?, ? wave ? 12 ?.")
    text("- Wave quantization problem: last wave ? fewer ?? ?, leaving ?? SMs idle (? ? occupancy).")
    text("- Solution: ? ?? ? ?? ? divide # SMs.")

    text("???")
    text("- Programming ??: ?? (HBM) -> ?? ? (?? ??) -> ?? (???)")
    text("- Details ? ?? (warps, bank conflicts, ?? coalescing, occupancy) determine ??")


def benchmarking_and_profiling():
    text("?????")
    text("1. ???? ?? ???? your ??")
    text("2. ? changes")
    text("3. ???? ?? ???? your ?? ??")

    benchmarking()   # How long does it take?
    profiling()      # Where time is being spent?

    text("???????????????")


def benchmarking():
    text("???? measures the wall-clock ?? ? performing ?? ??.")
    text("?????????????????????????????????")

    text("???????")
    text("- comparing ?? implementations (?? ? ???), ??")
    text("- ?? ?? ?? scales (e.g., ? ??).")

    text("??? ?? [`torch.utils.benchmark`](https://pytorch.org/tutorials/recipes/recipes/benchmark.html).")
    text("???????????????????")

    # ???? ????
    matmul = run_operation2(dim=1024, operation=lambda a, b: a @ b)
    result = benchmark(matmul)  # @inspect result

    # ?? how timing scales ? ??
    results = {}
    for dim in [256, 512, 1024, 2048, 4096, 8192]:
         results[dim] = benchmark(run_operation2(dim=dim, operation=lambda a, b: a @ b))  # @inspect results @stepover

    text("????????????????????????")


def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    """Benchmark `func` by running it `num_trials`.  Return the average time."""
    # ??: ?? times ?? ? ?? due ? compilation, etc.
    # Since ??? ?? the ?? multiple times, the timing that matters ? steady state.
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # ?? it ?? real ??!
    times: list[float] = [] # @inspect times
    for trial in range(num_trials):  # Do it multiple times to capture variance
        # ?? CUDA events ?? accurate GPU timing (avoid capturing CPU overhead)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()  # Start timing
        run()  # Actually perform computation
        end_event.record()  # End timing

        torch.cuda.synchronize()  # Wait for CUDA threads to finish

        times.append((start_event.elapsed_time(end_event)))  # @inspect times

    mean_time = mean(times)   # @inspect mean_time @stepover
    return mean_time


def profiling():
    text("??????????????????????????")
    text("??????????????????????????")

    text("PyTorch ? a ??-? [profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).")
    text("????????? nsight ???????")

    text("## add(dim=2048)")
    add_profile = profile(run_operation2(dim=2048, operation=lambda a, b: a + b))
    text(add_profile, verbatim=True)

    text("## matmul(dim=2048)")
    matmul_profile = profile(run_operation2(dim=2048, operation=lambda a, b: a @ b)) # @stepover
    text(matmul_profile, verbatim=True)

    text("## matmul(dim=128)")
    matmul_profile = profile(run_operation2(dim=128, operation=lambda a, b: a @ b)) # @stepover
    text(matmul_profile, verbatim=True)

    text("???")
    text("- ??? ?? ?? CUDA ?? ? ??? ?? ?? (the long names).")
    text("- ?? CUDA ?? ? invoked depending ? the tensor ??.")

    text("CUDA ??????????????????")
    text("??: cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bi...")
    text("- cutlass: NVIDIA's CUDA library ?? linear algebra")
    text("- sm100: ??? the NVIDIA Blackwell ?? (B200)")
    text("- f32: float32")
    text("- 64x64x16: tile ?? (?? ? this later)")


def profile(run: Callable, num_warmups: int = 1):
    # ??
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    # ?? the ?? ? the profiler
    with torch.profiler.profile(activities=[ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        torch.cuda.synchronize()

    # Print ? ?
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=100,
                                      row_limit=10)

    # Append ? profiles.txt
    with open("var/profiles.txt", "a") as f:
        f.write(f"Profile at {time.ctime()}:\n")
        f.write(table)
        f.write("\n\n")
    return table


def naive_vs_builtin_vs_compiled_gelu():
    text("?? [GeLU ????] ???????????(https://pytorch.org/docs/stable/generated/torch.nn.GELU.html).")

    x = torch.tensor([1.])  # @inspect x

    # 1. ?? naively ???? ? PyTorch (non-fused)
    y1 = naive_gelu(x)  # @inspect y1

    # 2. Built-? PyTorch ?? (fused)
    y2 = builtin_gelu(x)  # @inspect y2
    check_equal_1d(naive_gelu, builtin_gelu)  # Check it works

    # 3. ?? PyTorch compiler ? the naive ??
    compiled_gelu = torch.compile(naive_gelu)  # @stepover
    y3 = compiled_gelu(x)  # @inspect y3 @stepover
    check_equal_1d(naive_gelu, compiled_gelu)  # Check it works (compilation shouldn't change semantics) @stepover

    # Benchmarking
    naive_time = benchmark(run_operation1(dim=16384, operation=naive_gelu)) # @inspect naive_time @stepover
    builtin_time = benchmark(run_operation1(dim=16384, operation=builtin_gelu)) # @inspect builtin_time @stepover
    compiled_time = benchmark(run_operation1(dim=16384, operation=compiled_gelu)) # @inspect compiled_time @stepover
    text("??????????????")

    text("??????????????????????????")

    text("## naive_gelu")
    naive_gelu_profile = profile(run_operation1(dim=16384, operation=naive_gelu))  # @stepover
    text(naive_gelu_profile, verbatim=True)

    text("## builtin_gelu")
    builtin_gelu_profile = profile(run_operation1(dim=16384, operation=builtin_gelu))  # @stepover
    text(builtin_gelu_profile, verbatim=True)

    text("## compiled_gelu")
    compiled_gelu_profile = profile(run_operation1(dim=16384, operation=compiled_gelu))  # @stepover
    text(compiled_gelu_profile, verbatim=True)

    text("?????")
    text("- ????? multiple ??, ?? ?? reads/writes ?/? HBM (**no fusion**).")
    text("- ?????????? ?? ?? (**?? fusion**), ?? ?? ? HBM, ?? ?? ? HBM.")
    text("- ??????? Triton ???")


def triton_introduction():
    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=600)

    text("? CUDA?? NVIDIA ???????????????????")
    text("- ??? ??-grained control")
    text("- ??? ?? ? manage ?? things (e.g., ?? ??)")

    text("? Triton?? OpenAI ????????????????????")
    text("- ???????? (?? ?? getting started)")
    text("- ????? ?? ?? ?? ?? ??, operate ? it, ?? ? ? ?? ??")


def triton_gelu_example():
    text("??? GeLU ? Triton ???")

    x = torch.randn(8192, device=cuda_if_available())
    y = triton_gelu(x)

    check_equal_1d(triton_gelu, naive_gelu)  # Check for correctness @stepover

    text("Triton ???? PTX??????????? GPU ?????")

    text("?????? Triton ??? PTX ???")
    link(get_local_url("var/triton_gelu-ptx.txt"))

    text("???")
    text("- ld.??.* ?? st.??.* reads ?? writes ? ?? ??")
    text("- %ctaid.x ? ? index, %tid.x ? ?? index")
    text("- %f* ? floating point ???, %r* ? integer ???")
    text("- ?? ?? ?? 8 ?? at the ?? ?? (?? coarsening)")


def triton_gelu(x: torch.Tensor):
    # ????
    assert x.is_cuda
    assert x.is_contiguous()

    # ??????
    y = torch.empty_like(x)

    # ???? (?? divided ?? ?)
    # | T T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |
    # |    ? 0      |    ? 1      |     ? 2      |    ? 3     |
    num_elements = x.numel()  # @inspect num_elements
    BLOCK_SIZE = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)  # @inspect num_blocks

    # ????
    kernel = triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    # ?? PTX??????
    output_ptx("triton_gelu", kernel)  # @stepover

    return y


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # ????? `x_ptr`
    # ????? `y_ptr`

    # | T T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |
    # |    ? 0      |    ? 1      |     ? 2      |    ? 3     |

    pid = tl.program_id(axis=0)      # Identifies the block
    start = pid * BLOCK_SIZE         # ???????

    # Indices ?? this ?? ? ?? operate
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # Don't ??/?? past the end ? the tensor
    mask = offsets < num_elements

    # ??
    x = tl.load(x_ptr + offsets, mask=mask)

    # Approx gelu ? 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # ?? (tl.tanh doesn't exist, ?? tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # ??
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_softmax_example():
    text("?????????? Triton ??????? (e.g., GeLU).")
    text("??????????????")

    text("??????? Triton ??? softmax ???"), link("https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html")

    text("?????softmax ???????????????")
    text("?????????????????")
    text("[0 0 0]      =>   [1/3 1/3 1/3]", verbatim=True)
    text("[1 1 -inf]        [1/2 1/2 0  ]", verbatim=True)

    text("?????????????????")
    x = torch.tensor([
        [5., 5, 5],
        [0, 0, 100],
    ], device=cuda_if_available())
    y1 = naive_softmax(x) # @inspect y1

    text("????? Triton ???")
    image("images/triton-softmax.png", width=600)
    y2 = triton_softmax(x)  # @inspect y2

    # ?? our implementations ? ??
    check_equal_2d(pytorch_softmax, naive_softmax) # @stepover
    check_equal_2d(pytorch_softmax, triton_softmax) # @stepover


def naive_softmax(x: torch.Tensor):
    # M: ?? ? ?, N: ?? ? ?
    M, N = x.shape

    # ?? the max ? ?? ? (MN reads, M writes)
    x_max = x.max(dim=1)[0]

    # Subtract off the max (MN + M reads, MN writes)
    x = x - x_max[:, None]

    # Exponentiate (MN reads, MN writes)
    numerator = torch.exp(x)

    # ?? normalization ?? (MN reads, M writes)
    denominator = numerator.sum(dim=1)

    # Normalize (MN reads, MN writes)
    y = numerator / denominator[:, None]

    # ??: 5MN + M reads, 3MN + 2M writes
    # ???, ?? ? MN reads, MN writes (speedup ? 4x!)
    return y


def triton_softmax(x: torch.Tensor):
    # ??????
    y = torch.empty_like(x)

    # ????
    M, N = x.shape                          # Number of rows x number of columns
    block_size = triton.next_power_of_2(N)  # Each block contains all the columns
    num_blocks = M                          # Each block is a row

    # Launch ??
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )

    return y


@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # ?? ?? ? independently
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # ?? ? global ??
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # ??
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # ?? back ? global ??
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)


def triton_row_sum_example():
    text("? softmax ????????????????????????? Triton ????")
    text("???????????????")
    text("???4096 ??????? 1024??")

    text("???")
    text("- Break ? ? ?? tiles (4 ? the ?? ??)")
    text("- ?? ?? iterates ?? tiles ?? accumulates a sum")
    text("- ? final reduction (sum) ?? accumulators ? ?? ?? (?? ?? ? warp shuffles)")

    text("?????????????????? softmax??")
    x = torch.tensor([[1., 2, 3, 4], [5, 6, 7, 8]], device=cuda_if_available())  # @inspect x
    y1 = builtin_row_sum(x)  # @inspect y1

    image("images/triton-row-sum.png", width=600)

    y2 = triton_row_sum(x)  # @inspect y2


def builtin_row_sum(x: torch.Tensor):
    return x.sum(dim=1)


def triton_row_sum(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)  # Which row are we processing?

    # Accumulator ?? ?? ??
    # One ?: T1 T2 T3 T4 | T1 T2 T3 T4 | T1 T2 T3 T4 (N = 12, ?_?? = 4)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over tiles
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x

    # Final reduction ? ?_?? (?? ??) ? a ??
    result = tl.sum(acc, axis=0)

    tl.store(out_ptr + row, result)


def triton_matmul_relu_example():
    text("?????????????")
    a = torch.randn(1024, 1024, device=cuda_if_available())
    b = torch.randn(1024, 1024, device=cuda_if_available())
    c = naive_matmul_relu(a, b)

    text("?????????????")

    text("|        k                  n                     ", verbatim=True)
    text("|   [ A1 A2 A3 ]       [ B1 B2 B3 ]   [ C1 C2 C3 ]", verbatim=True)
    text("| m [ A4 A5 A6 ]  *  k [ B4 B5 B6 ] = [ C4 C5 C6 ]", verbatim=True)
    text("|   [ A7 A8 A9 ]       [ B7 B8 B9 ]   [ C7 C8 C9 ]", verbatim=True)

    text("**Naive ??:**")
    text("?????? (m, n)?")
    text("??? k?")
    text("- ?? A[m, k] ?? B[k, n] ? HBM.")
    text("- Multiply ?? accumulate.")
    text("????? HBM ?? C[m, n]?")

    text("???M K N ????M N ???")
    text("?????O(1)")

    text("?? C4 ? C5 ??? A4?A5?A6?")
    text("???? HBM ???? A4?A5?A6????????")
    text("?????????????")

    text("**Idealized ??:**")
    text("- ?? ?? ? A ?? B ?? ?? ??, ?? ?? C.")
    text("- ?? we ?? M K + K N reads ?? M N writes.")
    text("- This yields the idealized O(N) arithmetic intensity ? ??.")
    text("- ??, A ?? B ? ?? ? ?? ? fit ? ?? ??.")

    text("**Tiling:**")

    image("images/gemm_tiled.png", width=600)
    text("???????? C ?????????????")
    text("?? C ?????????")
    text("??????A ?????B ??????")
    text("- ?? the corresponding A tile ?? B tile ? HBM ?? ?? ??.")
    text("- ?? ???? ? the tiles.")
    text("- Accumulate ?? the partial sum (? ?? ??).")
    text("??????? HBM?")

    text("?????O(tile_??)?")

    text("?????")
    text("- ??, you ?? ? ?? an elementwise ??? ??.")
    text("- ??: GeLU(A @ B)")
    text("- Solution: ?? fusion!")

    text("**??.**")

    text("??????????????????")
    x = torch.tensor([[0., 1, 2, 3], [4, 5, 6, 7]])  # @inspect x
    stride_row, stride_col = x.stride()  # @inspect stride_row stride_col
    row = 1
    col = 2
    index = row * stride_row + col * stride_col  # @inspect index

    # ?? c = a @ b
    c = triton_matmul_relu(a, b)


def naive_matmul_relu(x: torch.Tensor, y: torch.Tensor):
    # Matmul followed ?? ReLU
    return torch.nn.functional.relu(x @ y)


def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    # A ? M x K, B ? K x N
    M, K = a.shape
    K, N = b.shape

    # ??????
    c = torch.empty((M, N), device=a.device)

    # ????
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    return c


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,    # ?? c = a @ b
    M, N, K,                # a is M x K, b is K x N, c is M x N
    stride_am, stride_ak,   # How to navigate a
    stride_bk, stride_bn,   # How to navigate b
    stride_cm, stride_cn,   # How to navigate c
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # We ? working ? the (m, n)-th tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Indices
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Row indices of a [BLOCK_M]
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Column indices of b [BLOCK_N]
    indices_k = tl.arange(0, BLOCK_K)                    # Row indices of a = column indices of b [BLOCK_K]

    # Initial ?? ? pointers ? a ?? b
    a_ptrs = a_ptr + indices_m[:, None] * stride_am + indices_k[None, :] * stride_ak  # [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + indices_k[:, None] * stride_bk + indices_n[None, :] * stride_bn  # [BLOCK_K, BLOCK_N]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Move along ? tiles ? a, ? tiles ? b
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(indices_m[:, None] < M) & (indices_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(indices_k[:, None] + k < K) & (indices_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak  # Advance to the next row tile of a
        b_ptrs += BLOCK_K * stride_bk  # Advance to the next column tile of b

    # ?? ??? ?? (e.g., ReLU)
    acc = tl.maximum(acc, 0.0)

    # ?? ?? tile
    c_ptrs = c_ptr + indices_m[:, None] * stride_cm + indices_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(indices_m[:, None] < M) & (indices_n[None, :] < N))


############################################################

def run_operation1(dim: int, operation: Callable) -> Callable:
    # ??: ?? one random dim x dim matrices
    x = torch.randn(dim, dim, device=cuda_if_available())
    # ?? a ?? ? ?? the ??
    return lambda : operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    # ??: ?? two random dim x dim matrices
    x = torch.randn(dim, dim, device=cuda_if_available())
    y = torch.randn(dim, dim, device=cuda_if_available())
    # ?? a ?? ? ?? the ??
    return lambda : operation(x, y)


def naive_gelu(x: torch.Tensor):
    # tanh approximation ? the gelu ??? ??
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


def builtin_gelu(x: torch.Tensor):
    # PyTorch's built-? GeLU ? the tanh approximation
    return torch.nn.functional.gelu(x, approximate="tanh")


def pytorch_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)


def check_equal_1d(f1, f2):
    x = torch.randn(2048, device=cuda_if_available())
    y1 = f1(x)  # @stepover
    y2 = f2(x)  # @stepover
    assert torch.allclose(y1, y2, atol=1e-6)


def check_equal_2d(f1, f2):
    x = torch.randn(2048, 2048, device=cuda_if_available())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def check_equal_2d_2d(f1, f2):
    x1 = torch.randn(2048, 2048, device=cuda_if_available())
    x2 = torch.randn(2048, 2048, device=cuda_if_available())
    y1 = f1(x1, x2)
    y2 = f2(x1, x2)
    assert torch.allclose(y1, y2, atol=1e-6)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def output_ptx(name: str, kernel):
    """Print out the PTX code generated by Triton for the given `kernel`."""
    ptx_path = f"var/{name}-ptx.txt"
    with open(ptx_path, "w") as f:
        ptx = kernel.asm["ptx"]
        f.write(ptx)


if __name__ == "__main__":
    main()
