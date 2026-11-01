import torch
import time
import math
import sys
import os
from inspect import isfunction
from typing import Callable
from torch import nn, tensor
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from edtrace import text, image, link
from gpu_util import cuda_if_available
from lecture_util import article_link

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda: None  # CUDA ???????????

def main():
    text("# Lecture 7: ??")
    text("??: ?? GPU ????")
    text("??: ?? GPU ?????")
    image("images/gpu-node-overview.png", width=700)

    text("????????**??**?????????????/???**??**??")
    text("??????????????????")

    text("???????")
    text("- ????? GPU? L1 cache / ?? ?? (fastest)")
    text("- ????? GPU? HBM")
    text("- ????? GPU? NVLink/NVSwitch")
    text("- ????? GPU? Infiniband/Ethernet (slowest)")

    text("???????/????????")
    text("???????/???? GPU/???????")

    text("?????? GPU?")
    text("1. ?????????? + ?? + ?????????? GPU?")
    text("2. ??????? GPU??? FLOPs???????")

    # ?? you execute ?? ??? (python lecture_07.py), it ?? multiprocessing, ?? ?? ?? ? ?? ?? (??).
    # ??, ?? you trace ?? (python -m edtrace.execute -m lecture_07), we turn off multiprocessing.
    link(title="stdout for this lecture", url="var/traces/lecture_07_stdout.txt")

    text("### ? 1 ????????/???????")
    collective_operations()    # Programming model
    hardware()                 # Hardware: how GPUs are connected
    torch_distributed()        # How this is implemented in NCCL/PyTorch
    benchmarking()             # Measure actual NCCL bandwidth

    text("### ? 2 ????????")
    text("?????? MLP ???????????????")
    text("?????MLP ? Transformer ???????????????????")

    data_parallelism()         # Cut up along the batch dimension
    tensor_parallelism()       # Cut up along the width dimension
    pipeline_parallelism()     # Cut up along the depth dimension

    text("??????")
    text("- ??/????")
    text("- ??????????????")
    text("- ???????????????????????")
    text("- Jax/TPUs: ?? define the ??, the sharding strategy, ?? the Jax compiler handles the rest "), link(title="levanter", url="https://crfm.stanford.edu/2023/06/16/levanter-1_0-release.html")
    text("- ????? PyTorch???????????????????")

    text("### ??")
    text("- ???????? ?? (??), tensor/expert (??), pipeline (??), ?? (??)")
    text("- ????? DDP (??-??), FSDP/ZeRO (??-gather + ??-scatter)")
    text("- ????? ???????? (e.g., NVLink)")
    text("- ?????? ??????????????????????")
    text("- ??**????**????**??**????????? GPU ???????**??**")
    text("- ?????????????????????????????????")


def collective_operations():
    text("**??????**?????????????? "), article_link("https://en.wikipedia.org/wiki/Collective_operation")
    text("- ????? 20 ?? 80 ??????????????????")
    text("- *Collective* ??? ? you specify a general ?? pattern ?? ?? ??.")
    text("- ???????????????????")

    text("**??**?")
    image("images/ranks.png", width=500)
    text("- **Rank**: a particular ??/GPU (e.g., 0, 1, 2, 3)")
    text("- **World ??**???????? 4?")

    text("???")
    text("- Broadcast, scatter, gather, ?? (foundations)")
    text("- ??-gather???-scatter???-?? (workhorse)")
    text("- ??-?-????? MoE?")

    text("**Broadcast**?? rank 0 ???????? rank")
    # ??
    rank0 = tensor([0., 1, 2, 3])

    # ??
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([0., 1, 2, 3])
    rank2 = tensor([0., 1, 2, 3])
    rank3 = tensor([0., 1, 2, 3])

    text("????????? rank 0 loads initial ??? ?? broadcasts ? ?? ranks")

    text("**Scatter**?? rank 0 ????????? rank")
    # ??
    rank0 = tensor([0., 1, 2, 3])

    # ??
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    text("???????????? ??-scatter")

    text("**Gather**???? rank ?????? rank 0?? scatter ???")
    # ??
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # ??
    rank0 = tensor([0., 1, 2, 3])

    text("???????????? ??-gather")

    text("**??**???? rank ?????? rank 0??????????? sum?min?max?")
    # ??
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # ??
    rank0 = tensor([6.])  # Sum of all ranks (0 + 1 + 2 + 3)

    text("???????????? ??-??")

    text("**??-gather**???? rank ?? gather????? rank 0")
    # ??
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # ??
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([0., 1, 2, 3])
    rank2 = tensor([0., 1, 2, 3])
    rank3 = tensor([0., 1, 2, 3])

    text("????? ?? rank holds ?? shard, gather ? ?? ?? ?? ?? ????")

    text("**??-scatter**???????? ????????")
    # ??
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([1., 2, 3, 4])
    rank2 = tensor([2., 3, 4, 5])
    rank3 = tensor([3., 4, 5, 6])

    # ??
    rank0 = tensor([6.])  # Sum along dim 0 (0 + 1 + 2 + 3)
    rank1 = tensor([10.]) # Sum along dim 1 (1 + 2 + 3 + 4)
    rank2 = tensor([14.]) # Sum along dim 2 (2 + 3 + 4 + 5)
    rank3 = tensor([18.]) # Sum along dim 3 (3 + 4 + 5 + 6)

    text("??????????????????????????????")

    text("**??-??** = ??-scatter + ??-gather")
    # ??
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([1., 2, 3, 4])
    rank2 = tensor([2., 3, 4, 5])
    rank3 = tensor([3., 4, 5, 6])

    # ??
    rank0 = tensor([6., 10, 14, 18])
    rank1 = tensor([6., 10, 14, 18])
    rank2 = tensor([6., 10, 14, 18])
    rank3 = tensor([6., 10, 14, 18])

    text("????? ?? ????, sum ?? ? ?? ?? shards, ?? replicate ?? ??")
    text("? ??-?? ??? ??-scatter + ??-gather ?????????? ZeRO/FSDP?")

    text("**??-?-??**??? rank ????? rank ???????????")
    # ??
    rank0 = tensor([0., 1, 2, 3])      # send  0 to rank 0,  1 to rank 1,  2 to rank 2,  3 to rank 3
    rank1 = tensor([4., 5, 6, 7])      # send  4 to rank 0,  5 to rank 1,  6 to rank 2,  7 to rank 3
    rank2 = tensor([8., 9, 10, 11])    # send  8 to rank 0,  9 to rank 1, 10 to rank 2, 11 to rank 3
    rank3 = tensor([12., 13, 14, 15])  # send 12 to rank 0, 13 to rank 1, 14 to rank 2, 15 to rank 3

    # ??
    rank0 = tensor([0, 4, 8, 12])
    rank1 = tensor([1, 5, 9, 13])
    rank2 = tensor([2, 6, 10, 14])
    rank3 = tensor([3, 7, 11, 15])

    text("?????")
    text("- ?? ?? MoEs: ?? rank ? ?? ? ?? ?? subset ? experts; ?? ? route ?? ? experts")
    text("- ?? balanced splits, ??-?-?? looks ?? transpose")
    text("- ? handles unbalanced splits (?? ?? splits ? ? ?? balanced ?? ??)")

    text("??????????")
    text("- ??: performs ?? associative/commutative ?? (sum, min, max)")
    text("- Scatter ? inverse ? gather")
    text("- ??: ??? destination ? ?? ??")


def hardware():
    text("???????????")
    image("https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs42774-021-00098-3/MediaObjects/42774_2021_98_Fig1_HTML.png?as=webp", width=500)
    text("- GPUs ? ?? node communicate via a PCI(e) bus (v7.0, 16 lanes => 242 GB/s) "), article_link("https://en.wikipedia.org/wiki/PCI_Express")
    text("- GPUs ? ?? nodes communicate via Ethernet (~200 MB/s)")

    text("?????????????")
    image("images/gpu-node-overview.png", width=700)

    text("?????")
    text("- 8 GPUs ? node, connected ?? NVLink ? an NVSwitch (B200s' NVLink 5.0 gets 1.8 TB/s; HBM ? 8 TB/s)")
    text("- 256 nodes ? pod, connected ?? Infiniband (via PCIe -> HCA / Infiniband NIC -> Infiniband cable) (~0.05 TB/s)")
    text("- N pods ? cluster / datacenter, connected ?? Ethernet (via PCIe -> CPU)")

    text("?? CPU?")
    text("- Ethernet ?? passing ?? the CPU (copying ?? ? ?? socket buffer, ?? TCP packets, copy ? NIC ring buffer)")
    text("- Remote Direct ?? ?? (RDMA): ?? ?? GPU ? ??? ??/?? ??? GPU's ?? ??? involving the CPU")
    text("- Infiniband supports RDMA, ?? standard Ethernet ? ?")

    text("???")
    text("- GB200/GB300 NVL72: 8 GPUs ? tray, 9 trays ? rack -> 72 GPUs ? ?? NVLink domain")
    text("- RDMA ?? Converged Ethernet (RoCE): Ethernet bypasses CPU, similar ?? cheaper/weaker ? Infiniband, used ?? Meta")

    text("### NVIDIA Collective ?? Library (NCCL)")
    text("NCCL translates collective ?? ?? ?-level packets ? ? sent ?? GPUs. "), link(title="talk", url="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/")
    text("- Detects topology ? ?? (e.g., ?? ? nodes, switches, NVLink/PCIe)")
    text("- Optimizes the path ?? GPUs")
    text("- Launches GPU ?? ? send/receive ??")


def torch_distributed():
    text("PyTorch ??? library (`torch.distributed`) "), link(title="documentation", url="https://pytorch.org/docs/stable/distributed.html")
    text("- Provides clean interface ?? collective ?? (e.g., `all_gather_into_tensor`)")
    text("- Supports multiple backends ?? ?? ??: gloo (CPU), nccl (GPU)")
    text("- ? supports ??-level algorithms (e.g., `FullyShardedDataParallel`) [? used ? this course]")

    text("????????????")
    spawn(collective_operations_main, world_size=4)


def collective_operations_main(rank: int, world_size: int):  # @inspect rank world_size
    """This function is running asynchronously for each process (rank = 0, ..., world_size - 1)."""
    setup(rank, world_size)

    ### ??-?? (dist = torch.???)
    dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)

    data = tensor([0., 1, 2, 3], device=cuda_if_available(rank)) + rank  # Both input and output

    print(f"Rank {rank} [before all-reduce]: {data}", flush=True)
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    print(f"Rank {rank} [after all-reduce]: {data}", flush=True)

    ### ??-scatter
    dist.barrier()

    input = torch.arange(world_size, dtype=torch.float32, device=cuda_if_available(rank)) + rank  # ??
    output = torch.empty(1, device=cuda_if_available(rank))  # Allocate output

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

    ### ??-gather
    dist.barrier()

    input = output  # ?? is the output of reduce-scatter
    output = torch.empty(world_size, device=cuda_if_available(rank))  # Allocate output

    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)

    text("?????-?? = ??-scatter + ??-gather?")

    cleanup()


def benchmarking():
    text("????????")

    # ??-??
    spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)

    # ??-scatter
    spawn(reduce_scatter, world_size=4, num_elements=100 * 1024**2)

    text("?????")
    link(title="How to reason about collective operations", url="https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce")
    link(title="Sample benchmarking code", url="https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py")


def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)  # @stepover

    # ????
    data = torch.randn(num_elements, device=cuda_if_available(rank))

    # ??
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here

    # ?? ??-??
    start_time = time.time()
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)  # @stepover

    # ?? the effective bandwidth
    dist.barrier()
    size_bytes = data.element_size() * data.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send + receive, world_size-1 steps in all-reduce
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    # Notes:
    # - Effective bandwidth ~ 2 * ??_bytes / ??_duration
    # - Independent ? world_??
    # - Independent ? topology (ring ? tree)

    cleanup()  # @stepover


def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)  # @stepover

    # ???????
    input = torch.randn(world_size, num_elements, device=cuda_if_available(rank))  # Each rank has a matrix
    output = torch.empty(num_elements, device=cuda_if_available(rank))

    # ??
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here

    # ?? ??-scatter
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)  # @stepover

    # ?? the effective bandwidth
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # How much data in the input
    sent_bytes = data_bytes * (world_size - 1)  # How much needs to be sent (no 2x here)
    total_duration = world_size * duration  # Total time for transmission
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    # Notes:
    # - ??-?? = ??-scatter + ??-gather
    # - ??-?? moves 2x the ?? ? 2x the ?? compared ? ??-scatter, ?? similar bandwidth

    cleanup()  # @stepover


def data_parallelism():
    image("images/data-parallelism.png", width=300)
    text("????? ?? rank gets a slice ? the ??")

    data = generate_sample_data()
    spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)

    text("?????")
    text("- ?? ? ?? ?? ranks (computed ? ?? ??)")
    text("- ?? ? ??-reduced ? ? the ?? ?? ranks")
    text("- ??, ?? remain the ?? ?? ranks")

    text("???FSDP/ZeRO??? ??-gather ? ??-scatter??????????????")


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


def data_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)  # @stepover

    # ?? the slice ? ?? ?? this rank (???, ?? rank ?? ?? ? its own ??)
    # --- B0 ---
    # --- B1 ---
    # --- B2 ---
    # --- B3 ---
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size @stepover
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(cuda_if_available(rank))

    # ?? MLP ?? params[0], ..., params[num_? - 1] (?? rank ? ?? ??)
    params = [get_init_params(num_dim, num_dim, rank) for layer in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state

    for step in range(num_steps):
        # ????
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # ????
        loss.backward()

        # Sync ?? ?? ???? (? difference ?? standard ?? ?? DDP)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # ????
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[layer]) for layer in range(num_layers)]}", flush=True)  # @stepover

    cleanup()  # @stepover


def tensor_parallelism():
    image("images/tensor-parallelism.png", width=300)
    text("????? ?? rank gets part ? ?? ?, transfer ?? ??/???")

    data = generate_sample_data()
    spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)


def tensor_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int):
    setup(rank, world_size)  # @stepover

    data = data.to(cuda_if_available(rank))  # All ranks get the data (batch_size x num_dim)
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_num_dim = int_divide(num_dim, world_size)  # Shard `num_dim`  @inspect local_num_dim @stepover

    # ?? ?? (?? rank gets 1/world_?? ? the ??)
    #  |  |  |  |
    # W0 W1 W2 W3
    #  |  |  |  |
    params = [get_init_params(num_dim, local_num_dim, rank) for layer in range(num_layers)]

    # ????
    x = data
    for layer in range(num_layers):
        # ????? (??_?? x local_num_dim)
        x = x @ params[layer]  # Note: this is only on a slice of the parameters
        x = F.gelu(x)

        # Allocate ?? ?? ??? (world_?? x ??_?? x local_num_dim)
        activations = [torch.empty(batch_size, local_num_dim, device=cuda_if_available(rank)) for _ in range(world_size)]

        # ?? ??-gather ?????
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # Concatenate them ? ?? ??_?? x num_dim
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)  # @stepover

    # ?????????

    cleanup()  # @stepover


def pipeline_parallelism():
    image("images/pipeline-parallelism.png", width=300)
    text("????? ?? rank gets subset ? ?, transfer ?? ??/???")

    data = generate_sample_data()
    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)


def pipeline_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)  # @stepover

    # ?? ?? the ??
    data = data.to(cuda_if_available(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim

    # ?? ? ?
    local_num_layers = int_divide(num_layers, world_size)  # @inspect local_num_layers @stepover

    # ?? rank gets a subset ? ?
    local_params = [get_init_params(num_dim, num_dim, rank) for layer in range(local_num_layers)]  # @stepover

    # ????

    # Break ? ?? micro ?? ? minimize the bubble
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size @stepover
    if rank == 0:
        # The ??
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # Allocate ?? ?? ???
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=cuda_if_available(rank)) for _ in range(num_micro_batches)]

    for x in micro_batches:
        # ?? ??? ? previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # ?? ? assigned ? this rank
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # Send ? the ??? rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)  # @stepover
            dist.send(tensor=x, dst=rank + 1)

    text("????????????????????")

    # ?????????

    cleanup()  # @stepover

############################################################

def setup(rank: int, world_size: int):
    """Initializes the distributed environment (called at start of process)."""
    # Specify ?? master lives (rank 0), used ? coordinate (actual ?? goes ?? NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Cleans up the distributed environment (called at end of process)."""
    torch.distributed.destroy_process_group()


class DisableDistributed:
    """
    Context manager that temporarily disables distributed functions (replaces with no-ops).
    This is for when we're tracing the lecture, since we can't trace through
    multiprocessing, so we just want to run the function directly without
    distributed communication.
    """
    def __enter__(self):
        self.old_functions = {}
        for name in dir(dist):
            value = getattr(dist, name, None)
            if isfunction(value):
                self.old_functions[name] = value
                setattr(dist, name, lambda *args, **kwargs: None)

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self.old_functions:
            setattr(dist, name, self.old_functions[name])


def spawn(func: Callable, world_size: int, *args, **kwargs):
    """
    Launches `world_size` processes that each calls `func` on world_size, args, kwargs.
    Note: if we are being traced (inside edtrace), we just run the function directly without multiprocessing and disable distributed functions.
    """
    # ??: ?? kwargs ? ? the same order ?? ?? main needs
    if not sys.gettrace():
        # ?? the normal ?? path ?? multiprocessing
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)
    else:
        # If we're being traced (?? edtrace), just ?? the ?? ???.
        with DisableDistributed():  # @stepover
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)


def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    """Create parameters and put them on the `rank`-th GPU."""
    torch.random.manual_seed(0)  # ??????
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=cuda_if_available(rank)) / math.sqrt(num_outputs))


def int_divide(a: int, b: int):
    """Return a / b and throw an error if there's a remainder."""
    assert a % b == 0
    return a // b


def summarize_tensor(tensor: tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"


def render_duration(duration: float) -> str:
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"


if __name__ == "__main__":
    main()
