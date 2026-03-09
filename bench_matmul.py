import random
from collections import defaultdict
import minitorch
import time
import sys
import numpy as np
import numba

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
HAS_CUDA = hasattr(numba, "cuda") and numba.cuda.is_available()
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps) if HAS_CUDA else None


def run_matmul(backend, size=16) -> None:
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    if HAS_CUDA and GPUBackend is not None:
        run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.perf_counter()
            run_matmul(FastTensorBackend, size)
            end_fast = time.perf_counter()
            fast_times.append(end_fast - start_fast)

            if HAS_CUDA and GPUBackend is not None:
                start_gpu = time.perf_counter()
                run_matmul(GPUBackend, size)
                try:
                    numba.cuda.synchronize()
                except Exception:
                    pass
                end_gpu = time.perf_counter()
                gpu_times.append(end_gpu - start_gpu)

        times[size]["fast"] = float(np.mean(fast_times))
        if gpu_times:
            times[size]["gpu"] = float(np.mean(gpu_times))
        print(times[size])

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")