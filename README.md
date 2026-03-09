# MiniTorch

A from-scratch implementation of a deep learning library, built purely in Python with NumPy and Numba for acceleration. This project covers the full stack of a modern deep learning framework — from basic math operators to GPU-accelerated tensor operations, automatic differentiation, and neural network layers.

## Project Architecture

The library is organized into five progressive modules, each building on the previous:

```
minitorch/
├── operators.py          # Module 0 — Core mathematical operators
├── module.py             # Module 0 — Neural network Module/Parameter system
├── datasets.py           # Module 0 — 2D classification datasets
├── testing.py            # Module 0 — Shared test utilities (MathTest)
│
├── autodiff.py           # Module 1 — Autodifferentiation engine
├── scalar.py             # Module 1 — Scalar variable with autodiff
├── scalar_functions.py   # Module 1 — Differentiable scalar functions
├── optim.py              # Module 1 — SGD optimizer
│
├── tensor_data.py        # Module 2 — Tensor storage, shapes, strides, indexing
├── tensor_ops.py         # Module 2 — SimpleOps backend (map/zip/reduce)
├── tensor_functions.py   # Module 2 — Differentiable tensor functions
├── tensor.py             # Module 2 — Tensor class with autodiff
│
├── fast_ops.py           # Module 3 — Numba JIT-compiled parallel ops
├── cuda_ops.py           # Module 3 — CUDA GPU kernels
│
├── nn.py                 # Module 4 — Pooling, softmax, dropout, max
└── fast_conv.py          # Module 4 — 1D/2D convolution (Numba)
```

## Module Details

### Module 0: Fundamentals

- **`operators.py`** — Scalar math functions (`add`, `mul`, `sigmoid`, `relu`, `log`, `exp`, etc.) and higher-order functions (`map`, `zipWith`, `reduce`). These serve as the atomic building blocks for all computations.
- **`module.py`** — PyTorch-style `Module` and `Parameter` classes. Modules form a tree structure that manages parameters, supports `train`/`eval` modes, and enables recursive parameter collection.
- **`datasets.py`** — 2D point classification datasets (`simple`, `diag`, `split`, `xor`, `circle`, `spiral`) for testing and visualization.

### Module 1: Autodifferentiation

- **`autodiff.py`** — The core autodiff engine: `central_difference` for numerical gradients, `topological_sort` for graph traversal, `backpropagate` for reverse-mode AD, and `Context` for storing forward-pass data.
- **`scalar.py`** — `Scalar` class that wraps a float value with a computation graph, supporting automatic gradient computation via `backward()`.
- **`scalar_functions.py`** — Differentiable function implementations (`Add`, `Mul`, `Sigmoid`, `ReLU`, `Log`, `Exp`, `LT`, `EQ`) with both `forward` and `backward` methods.
- **`optim.py`** — `SGD` optimizer that updates parameters using computed gradients.

### Module 2: Tensors

- **`tensor_data.py`** — Low-level tensor storage with NumPy arrays. Implements strided indexing, broadcasting (`shape_broadcast`, `broadcast_index`), and dimension permutation.
- **`tensor_ops.py`** — The `TensorBackend` abstraction and `SimpleOps` reference implementation. Provides `tensor_map`, `tensor_zip`, `tensor_reduce` as higher-order operations.
- **`tensor_functions.py`** — Tensor-level differentiable functions (`Neg`, `Add`, `Mul`, `Sigmoid`, `ReLU`, `Log`, `Exp`, `MatMul`, `Sum`, `Permute`, `View`) plus helper constructors (`zeros`, `rand`, `tensor`).
- **`tensor.py`** — `Tensor` class with operator overloading, autodiff support, shape manipulation, and backend dispatch.

### Module 3: Parallel Computing

- **`fast_ops.py`** — Numba `@njit`-compiled versions of `tensor_map`, `tensor_zip`, `tensor_reduce`, and `tensor_matrix_multiply` with `prange` parallelism and stride-alignment optimizations.
- **`cuda_ops.py`** — CUDA GPU kernels for map, zip, reduce, and tiled matrix multiplication using shared memory.

### Module 4: Neural Networks

- **`nn.py`** — Higher-level neural network operations: `avgpool2d`, `maxpool2d`, `softmax`, `logsoftmax`, `dropout`, `argmax`, and a differentiable `Max` function.
- **`fast_conv.py`** — Numba-compiled 1D and 2D convolution functions with corresponding autograd `Function` classes (`Conv1dFun`, `Conv2dFun`).

## Installation

```bash
# Clone and install
git clone <repo-url>
cd minitorch
pip install -e .

# With development tools
pip install -e ".[dev]"

# With CUDA support
pip install -e ".[cuda]"

# With Streamlit app dependencies
pip install -e ".[app]"
```

## Running Tests

Tests are organized by module and marked with pytest markers:

```bash
# Run all tests
pytest tests/

# Run by module
pytest tests/test_operators.py tests/test_module.py       # Module 0
pytest tests/test_autodiff.py tests/test_scalar.py        # Module 1
pytest tests/test_tensor_data.py tests/test_tensor.py     # Module 2
pytest tests/test_tensor_general.py                       # Module 3
pytest tests/test_nn.py tests/test_conv.py                # Module 4

# Run by task marker
pytest -m task0_1
pytest -m task2_3
```

## Interactive Apps

The `project/` directory contains Streamlit-based interactive demos:

```bash
# 2D classifier visualization
streamlit run project/app.py

# Sentiment analysis
streamlit run project/run_sentiment.py

# MNIST digit recognition
streamlit run project/run_mnist_interface.py
```

## Key Design Principles

1. **Everything from scratch** — No PyTorch or TensorFlow under the hood. Only NumPy for storage and Numba for JIT compilation.
2. **Progressive complexity** — Each module builds on the previous, mirroring how real frameworks are constructed.
3. **Multiple backends** — The same tensor operations run on `SimpleOps` (pure Python), `FastOps` (Numba CPU), or `CudaOps` (GPU).
4. **Full autodiff** — Both scalar and tensor autodiff with proper broadcasting support in the backward pass.
