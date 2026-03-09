from typing import Tuple

from .tensor import Tensor
from .tensor_functions import Function, rand
from . import operators
from .autodiff import Context
from .fast_ops import FastOps


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    tile_h = height // kh
    tile_w = width // kw

    input = input.contiguous()
    input = input.view(batch, channel, tile_h, kh, tile_w, kw)
    input = input.permute(0, 1, 2, 4, 3, 5)

    input = input.contiguous()
    input = input.view(batch, channel, tile_h, tile_w, kw * kh)

    return input, tile_h, tile_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    batch, channel, height, width = input.shape
    input, tile_h, tile_w = tile(input, kernel)
    input = input.mean(4)
    input = input.view(batch, channel, tile_h, tile_w)
    return input


max_reduce = FastOps.reduce(operators.max, -1e9)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        # Extract the dimension index as an integer from the tensor
        dimension = int(dim[0])
        # Store the input tensor and dimension for use in backward pass
        ctx.save_for_backward(input, dimension)
        # Compute and return the maximum values along the specified dimension
        return max_reduce(input, dimension)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        # Retrieve the saved input tensor and dimension from forward pass
        input_tensor, dimension = ctx.saved_values
        # Create a mask indicating positions of maximum values using argmax
        max_mask = argmax(input_tensor, dimension)
        # Propagate gradient only to positions where maximum values occurred
        return (grad_output * max_mask, 0.0)


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a specified dimension.

    Args:
        input: Input tensor to find maximum values from
        dim: Dimension along which to compute the maximum

    Returns:
        Tensor containing maximum values along the specified dimension

    """
    # Convert dimension to tensor format and apply the Max function
    dimension_tensor = input._ensure_tensor(dim)
    return Max.apply(input, dimension_tensor)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
        input : input tensor
        dim : dimension to apply argmax

    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    # Find the maximum values along the specified dimension
    maximum_values = max_reduce(input, dim)
    # Create a binary mask: 1 where input equals max, 0 elsewhere
    one_hot_mask = maximum_values == input
    return one_hot_mask


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the softmax as a tensor.

    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor

    """
    # Compute exponential of each element
    exp_values = input.exp()
    # Sum exponential values along the specified dimension
    normalization_factor = exp_values.sum(dim)
    # Normalize by dividing each exponential by the sum
    normalized_output = exp_values / normalization_factor
    return normalized_output


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor

    """
    # Find maximum values along dimension for numerical stability
    max_values = max(input, dim)
    # Subtract max to prevent overflow (log-sum-exp trick)
    shifted_input = input - max_values
    # Compute exponential of shifted values
    exp_shifted = shifted_input.exp()
    # Sum the exponentials along the dimension
    sum_exp = exp_shifted.sum(dim)
    # Take logarithm of the sum
    log_sum_exp = sum_exp.log()
    # Compute final log-softmax: input - log(sum(exp)) - max
    log_softmax_output = input - log_sum_exp - max_values
    return log_softmax_output


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    # Extract tensor dimensions
    batch_size, num_channels, img_height, img_width = input.shape
    # Reshape input into tiles for pooling operation
    tiled_input, tile_height, tile_width = tile(input, kernel)
    # Apply max pooling along the last dimension (kernel elements)
    pooled_tiles = max(tiled_input, 4)
    # Reshape back to original spatial dimensions with reduced size
    pooled_output = pooled_tiles.view(batch_size, num_channels, tile_height, tile_width)
    return pooled_output


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: input tensor
        rate: probability [0, 1) of dropping out each position
        ignore: skip dropout, i.e. do nothing at all

    Returns:
    -------
        tensor with random positions dropped out

    """
    # If dropout is disabled, return input unchanged
    if ignore:
        return input

    # Generate random values in [0, 1) with same shape as input
    random_values = rand(input.shape, input.backend)
    # Create binary mask: True where value > rate (keep), False otherwise (drop)
    keep_mask = random_values > rate
    # Apply mask to zero out dropped positions
    output = keep_mask * input
    return output
