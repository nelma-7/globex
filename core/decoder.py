"""Custom classes for algorithm implementation, based on the garage framework.
"""

"""
MLPDecoder class:
    Creates a simple MLP decoder
"""
import akro
import numpy as np

from garage import InOutSpec
from garage.torch.modules import MLPModule

class MLPDecoder(MLPModule):
    """This MLP network decodes latent context into some output vector
    Conceptually, the output vector could be a state, action, or reward
    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module): Activation
            function for intermediate dense layer(s). It should return a
            torch.Tensor.Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable or torch.nn.Module): Activation
            function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    @property
    def spec(self):
        """garage.InOutSpec: Input and output space."""
        input_space = akro.Box(-np.inf, np.inf, self._input_dim)
        output_space = akro.Box(-np.inf, np.inf, self._output_dim)
        return InOutSpec(input_space, output_space)

    @property
    def input_dim(self):
        """int: Dimension of the decoder input."""
        return self._input_dim

    @property
    def output_dim(self):
        """int: Dimension of the decider output (embedding)."""
        return self._output_dim
