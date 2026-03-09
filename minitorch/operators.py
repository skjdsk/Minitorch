"""Collection of the core mathematical operators used throughout the code base.

This module implements fundamental mathematical operations that serve as building blocks
for neural network computations in MiniTorch.

NOTE: The `task0_1` tests will not fully pass until you complete `task0_3`.
Some tests depend on higher-order functions implemented in the later task.
"""

# =============================================================================
# Task 0.1: Mathematical Operators
# =============================================================================
import math
from typing import Callable, List, TypeVar


def mul(x: float, y: float) -> float:
    """Multiply two numbers and return the product."""
    return x * y


def id(x: float) -> float:
    """Identity function: return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers and return the sum."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Return 1.0 if x < y else 0.0"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return 1.0 if x == y else 0.0"""
    return 1.0 if x == y else 0.0


def is_close(x: float, y: float, tol: float = 1e-2) -> float:
    """Return 1.0 if |x - y| < tol else 0.0"""
    return 1.0 if abs(x - y) < tol else 0.0


def max(x: float, y: float) -> float:
    """Return the larger of two numbers."""
    if x > y:
        return x
    elif y > x:
        return y
    else:
        return x


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid: 1 / (1 + exp(-x)) with sign-aware form."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def relu(x: float) -> float:
    """Rectified Linear Unit: max(0, x)."""
    if x > 0:
        return x
    else:
        return 0


def log(x: float) -> float:
    """Natural logarithm of x."""
    return math.log(x)


def exp(x: float) -> float:
    """Exponential e**x."""
    return math.exp(x)


def inv(x: float) -> float:
    """Reciprocal 1/x."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Backward for log: d * (1/x)."""
    return d / x


def inv_back(x: float, d: float) -> float:
    """Backward for inv: d * (-1/x^2)."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Backward for ReLU: return d if x>0 else 0."""
    if x > 0:
        return d
    else:
        return 0


# =============================================================================
# Task 0.3: Higher-Order Functions
# =============================================================================

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def map(fn: Callable[[T], U], iterable: List[T]) -> List[U]:
    """Apply fn to each element of iterable and return a new list."""
    result = []
    for x in iterable:
        result.append(fn(x))
    return result


def zipWith(fn: Callable[[T, U], V], list1: List[T], list2: List[U]) -> List[V]:
    """Combine two lists elementwise using fn and return a new list."""
    result = []
    for x, y in zip(list1, list2):
        result.append(fn(x, y))
    return result


def reduce(fn: Callable[[T, T], T], iterable: List[T], initial_value: T) -> T:
    """Reduce iterable to a single value by repeatedly applying fn."""
    result = initial_value
    for x in iterable:
        result = fn(result, x)
    return result


def negList(lst: list[float]) -> list[float]:
    """Negate all elements in a list."""
    return map(neg, lst)


def addLists(lst1: list[float], lst2: list[float]) -> list[float]:
    """Add corresponding elements from two lists."""
    return zipWith(add, lst1, lst2)


def sum(lst: list[float]) -> float:
    """Sum all elements in a list."""
    return reduce(add, lst, 0.0)


def prod(lst: list[float]) -> float:
    """Calculate product of all elements in a list."""
    return reduce(mul, lst, 1.0)
