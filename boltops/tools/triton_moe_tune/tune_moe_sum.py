import unittest
import itertools
import gc
import torch
import triton
import triton.language as tl
from typing import List

from boltops.fused_moe.triton import triton_moe_sum
from boltops.utility.perf_utils import perftest


def torch_moe_sum_ref(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
    """参考实现：对 [M, top_k, N] 在 top_k 维求和，与 vLLM ops.moe_sum 语义一致。"""
    torch.sum(input_tensor, dim=1, out=output_tensor)

TOP_K_SIZE = 8
N_SIZE = 4096

class TestMoeSum(unittest.TestCase):
    # 测试配置
    DTYPES = [torch.half]
    # M = [32]                # 序列长度
    M = [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 116383, 16384, 32767, 32768, 65535, 65536, 65537]
    N = [N_SIZE]              # 隐藏维度
    TOP_KS = [TOP_K_SIZE]            # 专家数量
    SEEDS = [0]             # 随机种子

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _test_moe_sum(self, M: int, N: int, top_k: int, dtype: torch.dtype, seed: int):
        print(f"\nTesting: Batch={M}, N={N}, TOP_K={top_k}, dtype={dtype}")

        torch.manual_seed(seed)

        # 创建测试数据
        input_tensor = torch.randn((M, top_k, N), dtype=dtype, device="cuda")
        output_tensor = torch.zeros((M, N), dtype=dtype, device="cuda")
        ref_output = torch.zeros((M, N), dtype=dtype, device="cuda")

        # 计算参考结果
        torch_moe_sum_ref(input_tensor, ref_output)

        # 使用我们的triton实现
        triton_moe_sum(input_tensor, output_tensor)

        # 验证结果
        max_diff = torch.max(torch.abs(output_tensor - ref_output))
        rel_diff = torch.mean(torch.abs(output_tensor - ref_output)) / torch.mean(torch.abs(ref_output))

        print(f"Max absolute difference: {max_diff}")
        print(f"Mean relative difference: {rel_diff}")

        # 检查结果是否在可接受的误差范围内
        if dtype == torch.float32:
            threshold = 1e-5
        else:
            threshold = 1e-2

        self.assertTrue(
            rel_diff < threshold,
            f"Relative difference {rel_diff} exceeds threshold {threshold}"
        )

    def test_moe_sum(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.TOP_KS,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                top_k=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                self._test_moe_sum(*params)

# 性能测试配置
moe_sum_configs = [
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[
            # 测试不同batch size
            *[2**i for i in range(0, 17)],  # 从1到 65504
            32767,
            65535,
            65537
        ],
        line_arg='implementation',  # 改为比较不同实现
        line_vals=['torch', 'triton'],  # 两种实现方式
        line_names=['torch.sum(dim=1)', 'triton_moe_sum'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='Time(ms)',
        xlabel='Batch Size (M)',
        plot_name='MoE Sum Performance Comparison(ms)',
        args={
            'N': N_SIZE,
            'TOP_K': TOP_K_SIZE,
            'dtype': torch.float16,
            'device': 'cuda',
            'seed': 0
        }
    )
]


@perftest(num_warmup=1, num_iters=11, testGraph=True)
def moe_sum_benchmark(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    implementation: str,
):
    if implementation == "torch":
        torch_moe_sum_ref(input_tensor, output_tensor)
    else:
        triton_moe_sum(input_tensor, output_tensor)
    return output_tensor


@triton.testing.perf_report(moe_sum_configs)
def bench_moe_sum(
    M: int,
    N: int,
    TOP_K: int,
    dtype: torch.dtype,
    implementation: str,  # 新增参数用于选择实现方式
    device: str = "cuda",
    seed: int = 0
):
    """MOE sum操作的性能测试

    Args:
        M: batch size
        N: hidden dimension
        TOP_K: number of experts
        dtype: data type
        implementation: 实现方式 ('torch' 或 'triton')
        device: device to run on
        seed: random seed
    """
    warmup = 25
    rep = 100

    print(f"\nBenchmark: M={M}, N={N}, TOP_K={TOP_K}, dtype={dtype}, implementation={implementation}")

    torch.manual_seed(seed)

    # 准备测试数据
    input_tensor = torch.randn((M, TOP_K, N), dtype=dtype, device=device)
    output_tensor = torch.zeros((M, N), dtype=dtype, device=device)

    # 打印内存使用情况
    print("\nMemory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / (1024**2):.2f} MB")

    # 测量性能（统一走 perftest + testGraph=True）
    _, avg_us = moe_sum_benchmark(input_tensor, output_tensor, implementation)
    ms = avg_us / 1000.0

    # 计算理论吞吐量
    gb_per_sec = lambda ms: 2 * input_tensor.nelement() * input_tensor.element_size() / (ms * 1e6)
    print(f"Throughput: {gb_per_sec(ms):.2f} GB/s")

    return ms

if __name__ == "__main__":
    # # 运行单元测试
    unittest.main(verbosity=2)

    # 运行性能测试
    bench_moe_sum.run(print_data=True)
