from __future__ import annotations

from typing import TYPE_CHECKING
from abc import abstractmethod

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    """Turn a singleton tuple into a value"""
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Internal wrapper: ensure backward returns a tuple of floats."""
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Internal wrapper: dispatch to subclass forward."""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        """Apply this ScalarFunction to the given values and return a new `Scalar`.

        Steps:
            1) Extract raw floats from input Scalars (or wrap Python numbers as Scalars).
            2) Create a `Context` and run `forward` to get the float result.
            3) Build a new `Scalar` with history `(last_fn=cls, ctx, inputs=scalars)`.

        Args:
            *vals: Scalar-like inputs (Scalars, ints, or floats).

        Returns:
            Scalar: Result scalar carrying the computation history.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, (float, int)), "Expected return type float got %s" % (type(c))
        c = float(c)

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)

    @staticmethod
    @abstractmethod
    def forward(ctx: Context, *args: float) -> float:
        """Compute the forward value for this function (subclasses must implement)."""
        ...

    @staticmethod
    @abstractmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute input gradients given upstream gradient `d_output` (must implement)."""
        ...


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return a + b."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """d/da = 1 * d_output, d/db = 1 * d_output."""
        return d_output, d_output


class Log(ScalarFunction):
    r"""Log function $f(x) = \,\log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return log(a) and save `a` for backward."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """d/da log(a) = 1/a; use operators.log_back for numerical stability."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


### To implement for Task 1.2 and 1.4 ###
# Look at the above classes for examples on how to implement the forward and backward functions
# Use the operators.py file from Module 0


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return a * b and save (a, b) for backward."""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """d/da = b * d_output, d/db = a * d_output."""
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return 1/a and save `a` for backward."""
        ctx.save_for_backward(a)
        return 1.0 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """d/da (1/a) = -1/a^2."""
        (a,) = ctx.saved_values
        return d_output * (-1.0 / (a * a))


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return -a."""
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """d/da (-a) = -1."""
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function σ(a) = 1 / (1 + exp(-a))"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return sigmoid(a) and save it for backward."""
        s = 1.0 / (1.0 + operators.exp(-a))
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """d/da σ(a) = σ(a) * (1 - σ(a))."""
        (s,) = ctx.saved_values
        return d_output * s * (1.0 - s)


class ReLU(ScalarFunction):
    """ReLU function f(a) = max(0, a)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return max(0, a) and save `a` for backward."""
        ctx.save_for_backward(a)
        return a if a > 0.0 else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Gradient is 1 if a>0 else 0."""
        (a,) = ctx.saved_values
        return d_output * (1 if a > 0.0 else 0.0)


class Exp(ScalarFunction):
    """Exp function f(a) = e^a"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Return exp(a) and save the output for backward."""
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """d/da exp(a) = exp(a) (re-use saved output)."""
        (out,) = ctx.saved_values
        return d_output * out


class LT(ScalarFunction):
    """Less-than function $f(x,y) = 1.0$ if x < y else $0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return 1.0 if a < b else 0.0 (no saved values needed)."""
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Non-differentiable almost everywhere; return zeros."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x,y) = 1.0$ if x == y else $0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return 1.0 if a == b else 0.0 (no saved values needed)."""
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Non-differentiable almost everywhere; return zeros."""
        return 0.0, 0.0
