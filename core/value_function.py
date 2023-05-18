"""Custom classes for algorithm implementation, based on the garage framework.
"""
import torch

from garage.torch.modules import MLPModule
from torch import nn

class ContinuousMLPValueFunction(MLPModule):
    """Implements a continuous MLP value function V(s).
    """

    def __init__(self, env_spec, **kwargs):
        """Initialize class with multiple attributes.
        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.
        """
        self._env_spec = env_spec
        self._obs_dim = env_spec.input_space.flat_dim
        self._output_dim = env_spec.output_space.flat_dim

        MLPModule.__init__(self,
                           input_dim=self._obs_dim,
                           output_dim=self._output_dim,
                           **kwargs)

    # pylint: disable=arguments-differ
    def forward(self, observations, actions):
        """Return Q-value(s).
        Args:
            observations (np.ndarray): observations.
            actions (np.ndarray): actions.
        Returns:
            torch.Tensor: Output value
        """
        return super().forward(torch.cat([observations, actions], 1))
    

from torch.nn import functional as F
from garage.torch.value_functions.value_function import ValueFunction

class MLPValueFunction(ValueFunction):
    """ MLP Value Function.
    Creates a value function based on a simple MLP.
    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): The name of the value function.
    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='MLPValueFunction'):
        super().__init__(env_spec, name)

        input_dim = env_spec.input_space.flat_dim
        output_dim = 1

        self.module = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

    def compute_loss(self, obs, returns, use_huber_loss):
        r"""Compute mean value of loss. Adapted from Zintgraf et al (2021)
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).
        """
        value = self.module(obs)
        
        if use_huber_loss:
            loss = F.smooth_l1_loss(value, returns, reduction='mean')
        else:
            loss = 0.5 * (returns - value).pow(2).mean()

        return loss

    # pylint: disable=arguments-differ
    def forward(self, obs):
        r"""Predict value based on paths.
        """
        return self.module(obs).squeeze(-1)
