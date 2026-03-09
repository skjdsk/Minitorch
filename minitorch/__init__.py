"""MiniTorch - A minimal deep learning library for educational purposes.

This library implements a from-scratch deep learning framework covering:
  - Module 0: Mathematical operators and module system
  - Module 1: Autodifferentiation with scalars
  - Module 2: Tensor operations and tensor autodiff
  - Module 3: Parallel computing with Numba/CUDA backends
  - Module 4: Neural network layers and convolutions
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401

# Module 0: Core operators, module system, datasets
from .module import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403

# Module 1: Autodiff, scalars, optimizers
from .autodiff import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403

# Module 2: Tensors
from .tensor_data import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403

# Module 3: Fast backends (Numba + CUDA)
from .fast_ops import *  # noqa: F401,F403
from .cuda_ops import *  # noqa: F401,F403
from . import fast_ops, cuda_ops  # noqa: F401

# Module 4: Neural network layers and convolutions
from .nn import *  # noqa: F401,F403
from .fast_conv import *  # noqa: F401,F403
