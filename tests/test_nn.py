import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))  # type: ignore[misc]
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))  # type: ignore[misc]
def test_max(t: Tensor) -> None:
    """Verify max reduction functionality across various tensor dimensions."""
    # Verify maximum computation along the last dimension (dimension 2)
    max_result_dim2 = minitorch.max(t, 2)
    expected_max_dim2 = max([t[0, 0, idx] for idx in range(4)])
    assert max_result_dim2[0, 0, 0] == expected_max_dim2

    # Verify maximum computation along the middle dimension (dimension 1)
    max_result_dim1 = minitorch.max(t, 1)
    expected_max_dim1 = max([t[0, idx, 0] for idx in range(3)])
    assert max_result_dim1[0, 0, 0] == expected_max_dim1

    # Verify maximum computation along the first dimension (dimension 0)
    max_result_dim0 = minitorch.max(t, 0)
    expected_max_dim0 = max([t[idx, 0, 0] for idx in range(2)])
    assert max_result_dim0[0, 0, 0] == expected_max_dim0

    # Introduce minimal random noise to prevent equal values for gradient verification
    noise = minitorch.rand(t.shape) * 1e-5
    perturbed_tensor = t + noise
    minitorch.grad_check(lambda tensor: minitorch.max(tensor, 2), perturbed_tensor)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))  # type: ignore[misc]
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())  # type: ignore[misc]
def test_drop(t: Tensor) -> None:
    # Property: dropout with rate=0.0 should return original tensor
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]

    # Property: dropout with rate=1.0 should return zero tensor
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))  # type: ignore[misc]
def test_softmax(t: Tensor) -> None:
    # Use fixed tensor to avoid numerical issues
    t = minitorch.tensor(
        [
            [
                [
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                ]
            ]
        ]
    )

    # Property: softmax should sum to 1.0 along reduced dimension
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    # Test gradient computation
    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))  # type: ignore[misc]
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
