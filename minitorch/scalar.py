from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union, List

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn: The last Function that was called.
        ctx: The context for that Function.
        inputs: The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence["Scalar"] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward
# Use what you defined in scalar_functions.py

_var_count = 0


class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> "Scalar":
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> "Scalar":
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> "Scalar":
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> "Scalar":
        """Return self + b."""
        return Add.apply(self, b)

    def __bool__(self) -> bool:
        return bool(self.data)

    def __float__(self) -> float:
        return float(self.data)

    def __lt__(self, b: ScalarLike) -> "Scalar":
        """Return 1.0 if self < b else 0.0 as a Scalar."""
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> "Scalar":
        """Return 1.0 if self > b else 0.0 as a Scalar."""
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> "Scalar":  # type: ignore[override]
        """Return 1.0 if self == b else 0.0 as a Scalar."""
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike) -> "Scalar":
        """Return self - b."""
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> "Scalar":
        """Return -self."""
        return Neg.apply(self)

    def __radd__(self, b: ScalarLike) -> "Scalar":
        return self + b

    def __rmul__(self, b: ScalarLike) -> "Scalar":
        return self * b

    def log(self) -> "Scalar":
        """Return the natural logarithm of this scalar."""
        return Log.apply(self)

    def exp(self) -> "Scalar":
        """Return e**self as a scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> "Scalar":
        """Return sigmoid(self) = 1 / (1 + exp(-self))."""
        return Sigmoid.apply(self)

    def relu(self) -> "Scalar":
        """Return relu(self) = max(self, 0)."""
        return ReLU.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the incoming derivative `x` onto this variable's gradient.

        Should only be called during autodifferentiation on leaf variables.

        Args:
            x: Value to be accumulated into the derivative of this variable.

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        """True if this variable is created by the user (no `last_fn`)."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable does not track history / require gradients."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Immediate parent variables that produced this variable."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply local chain rule at this node.

        Given the upstream derivative `d_output` (∂L/∂this), return an iterable of
        (parent, ∂L/∂parent) pairs computed via this node's `last_fn._backward`.

        Args:
            d_output: Upstream derivative w.r.t. this node.

        Returns:
            Iterable of (parent, parent_grad) pairs for non-constant parents.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        grads = h.last_fn._backward(h.ctx, d_output)
        out: List[Tuple[Variable, Any]] = []
        for parent, g in zip(h.inputs, grads):
            if not parent.is_constant():
                out.append((parent, g))
        return out

    def backward(self, d_output: Optional[float] = None) -> None:
        """Run reverse-mode autodiff to fill derivatives for all leaves in the history.

        Args:
            d_output: Starting derivative to backpropagate through the graph.
                Typically omitted, in which case it defaults to 1.0.

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function by comparing gradients
    from backprop against central differences.

    Args:
        f: A function from n-Scalar inputs to a single Scalar output.
        *scalars: The input Scalar arguments to test.

    Raises:
        AssertionError: If any gradient differs from the central-difference
            estimate beyond the specified tolerances.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([s.data for s in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([s.data for s in scalars]), x.derivative, i, check.data),
        )
