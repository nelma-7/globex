"""Custom classes for algorithm implementation, based on the garage framework.
"""
import akro
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from garage import InOutSpec
from garage.np.embeddings import Encoder
from garage.torch import NonLinearity
from garage.torch import global_device
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule

class MultiHeadedMLPEncoder(MultiHeadedMLPModule, Encoder):
    """This MLP network encodes context of RL tasks.
        Context is stored in the terms of observation, action, and reward, and this
        network uses an MLP module for encoding it.
        Args:
            n_heads (int): number of output heads
            input_dim (int) : Dimension of the network input.
            output_dims (int or list or tuple): Dimension of the network output.
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
        """int: Dimension of the encoder input."""
        return self._input_dim

    @property
    def output_dim(self):
        """int: Dimension of the encoder output (embedding)."""
        return self._output_dim

    def reset(self, do_resets=None):
        """Reset the encoder.
        This is effective only to recurrent encoder. do_resets is effective
        only to vectoried encoder.
        For a vectorized encoder, do_resets is an array of boolean indicating
        which internal states to be reset. The length of do_resets should be
        equal to the length of inputs.
        Args:
            do_resets (numpy.ndarray): Bool array indicating which states
                to be reset.
        """    

class GRURecurrentEncoder(nn.Module, Encoder):
    """GRU Recurrent Encoder. Can input either an individual timestep or a full trajectory.
    Updated to support truncated backprop through time (via detach_every param)
        
    Args:
        input_dim (int): Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output. For variBAD, this will be the dim of the latent space
        num_recurrent_layers (int): Number of recurrent GRU layers. Default is 1 (based on original implementation)
        hidden_size (int): Size of hidden layer. This has historically been tuned (eg. 64 for gridworld; 128 for mujoco)
        hidden_nonlinearity (callable or torch.nn.Module or list or tuple):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearities (callable or torch.nn.Module or list or tuple):
            Activation function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_head
        mu_w_inits / var_w_inits (callable or list or tuple): Initializer function for
            the weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        mu_b_inits / var_b_inits (callable or list or tuple): Initializer function for
            the bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 input_dim,
                 output_dims,
                 hidden_size,
                 default_batch_size=1,
                 num_recurrent_layers=1,
                 layers_before_gru=(),
                 layers_after_gru=(),
                 gru_batch_first=False,
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 mu_w_inits=nn.init.xavier_normal_,
                 mu_b_inits=None,
                 var_w_inits=nn.init.xavier_normal_,
                 var_b_inits=None,
                 layer_normalization=False):
        super().__init__()

        self._input_dim = input_dim 
        self._output_dim = int(output_dims / 2) # Divide by 2 here as we have different FC layers for mean/std
        self.hidden_size = hidden_size
        self.num_recurrent_layers = num_recurrent_layers
        self._default_batch_size = default_batch_size #useful for denoting dims later

        # Initialise hidden state 
        self._hidden_state = torch.zeros((self.num_recurrent_layers, default_batch_size, 
                            self.hidden_size), requires_grad=True).to(global_device())
        
        # Initialise fc layers before gru (typically empty, but here for flexibility)
        self._fc_before_gru = nn.ModuleList()
        curr_input_dim = input_dim
        for i in range(len(layers_before_gru)):
            hidden_layers = nn.Sequential()
            if layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                         nn.LayerNorm(curr_input_dim))

            linear_layer = nn.Linear(curr_input_dim, layers_before_gru[i])
            curr_input_dim = layers_before_gru[i]

            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)

            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))

            self._fc_before_gru.append(hidden_layers)
            
        # Initialise GRU. Note that the hidden state has dimensions (num_recurrent_layers, batch_size, hidden_size)
        self.gru = nn.GRU(input_size=curr_input_dim,
                          hidden_size=hidden_size,
                          num_layers=num_recurrent_layers,
                          batch_first=gru_batch_first)

        # Initialise fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self._fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            hidden_layers = nn.Sequential()
            if layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                         nn.LayerNorm(curr_input_dim))

            linear_layer = nn.Linear(curr_input_dim, layers_after_gru[i])
            curr_input_dim = layers_after_gru[i]

            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)

            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))
                                         
            self._fc_before_gru.append(hidden_layers)
        
        # Initialise output layers
        self.fc_mu = nn.Linear(curr_input_dim, self._output_dim) 
        self.fc_var = nn.Linear(curr_input_dim, self._output_dim) 

        mu_w_inits(self.fc_mu.weight)
        if mu_b_inits is not None: #if inits are None, then the bias will be uniform (see pytorch nn.Linear docs)
            mu_b_inits(self.fc_mu.bias)
        var_w_inits(self.fc_var.weight)
        if var_b_inits is not None:
            var_b_inits(self.fc_var.bias)

    @property
    def spec(self):
        """garage.InOutSpec: Input and output space."""
        input_space = akro.Box(-np.inf, np.inf, self._input_dim)
        output_space = akro.Box(-np.inf, np.inf, self._output_dim)
        return InOutSpec(input_space, output_space)

    @property
    def input_dim(self):
        """int: Dimension of the encoder input."""
        return self._input_dim

    @property
    def output_dim(self):        
        """int: Dimension of the encoder output (embedding)."""
        return self._output_dim
    
    # pylint: disable=arguments-differ
    def forward(self, input_val, return_prior=False, detach_every = None):
        """Forward method.
        Forwards the inputs through the Linear/GRU layers and returns latent means, latent logvars, and the base output of nn.GRU
        
        Args:
            input_val (torch.Tensor): Input values with (N, S, input_dim) shape, where
                                N = num_tasks, S = seq_len
            return_prior (bool): Determines whether we will return the prior or not
                                Note that with variBAD, return_prior=True only when training the VAE on a full trajectory
        Returns:
            List[torch.Tensor]: Output values
        """
        x = input_val
        
        # This function will reset the hidden state and ensure our hidden state has the correct dims
        if return_prior: 
            prior_latent_mean, prior_latent_var= self.reset_and_get_prior(x.shape[1])

        for layer in self._fc_before_gru:
            x = layer(x)
        
        #a bit of a hack to make sure devices are controlled properly
        if self._hidden_state.device != global_device():
            self._hidden_state=self._hidden_state.to(global_device())

        if detach_every is None:
            output, self._hidden_state = self.gru(x, self._hidden_state)
        else:
            output = []
            for i in range(int(np.ceil(x.shape[1] / detach_every))): # x has dims (num_tasks, seq_len/batch_size, input_dim)
                curr_input = x[:,i*detach_every:i*detach_every+detach_every,:]  # pytorch caps if we overflow, nice
                curr_output, self._hidden_state = self.gru(curr_input, self._hidden_state)
                output.append(curr_output)
                # detach hidden state; useful for BPTT when sequences are very long
                self._hidden_state = self._hidden_state.detach()
            output = torch.cat(output, dim=1)
        
        for layer in self._fc_after_gru:
            output = layer(output)

        latent_mean = self.fc_mu(output)
        latent_var = self.fc_var(output)
        
        if return_prior:
            latent_mean = torch.cat([prior_latent_mean, latent_mean])
            latent_var = torch.cat([prior_latent_var, latent_var])

        return latent_mean, latent_var
    
    def reset_and_get_prior(self, batch_size):
        """Resets the hidden state of the GRU encoder, 
            and returns a prior that is generated by passing a hidden state of 0s through our FC layers after the GRU
            
            This copies what happens in variBAD, but we will ignore it for the new algo
        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.
        Returns:
            List[torch.Tensor]: Output values
        """

        #creates a "prior" resetting the hidden state, then forwarding it through the FC layers only
        self._hidden_state = torch.zeros((self.num_recurrent_layers, batch_size, self.hidden_size), requires_grad=True).to(global_device())
        h = self._hidden_state

        # forward through fully connected layers after GRU
        for i in range(len(self._fc_after_gru)):
            h = F.relu(self._fc_after_gru[i](h))

        # outputs
        latent_mean = self.fc_mu(h)
        latent_var = self.fc_var(h)

        return latent_mean, latent_var
    
    def reset(self, batch_size = None, do_resets=None):
        """Reset the encoder. Note that this function is not actually used at the moment
        This is effective only to recurrent encoder. 
        batch_size is a helpful input to ensure the hidden state has the right batch size.
            This is important as batch sizes may change when training policy, compared to training the VAE
        For a vectorized encoder, do_resets is an array of boolean indicating
        which internal states to be reset. The length of do_resets should be
        equal to the length of inputs.
        Args:
            do_resets (numpy.ndarray): Bool array indicating which states
                to be reset.
        """ 
        if batch_size is None:
            batch_size = self._default_batch_size

        self._hidden_state = torch.zeros((self.num_recurrent_layers, batch_size, self.hidden_size), requires_grad=True).to(global_device())
    
        return self._hidden_state       

    @property
    def hidden_state(self):
        """Return hidden state.
        Returns:
            torch.Tensor: Hidden state weights, with shape :math:`(D, N, H)`.
                D is the number of hidden layers (in this case, 2). N is batch_size (ie num_tasks in PEARL). H is the size of the hidden layer.
        """
        return self._hidden_state 