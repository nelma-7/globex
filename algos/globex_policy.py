"""Custom policy based on the garage framework.
Implements a context-conditioned policy that has a global and local component.

The global context is sampled at the beginning of an episode and stays unchanged
Local context is sampled at each timestep

Final version for submitted code
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from garage.torch import global_device, product_of_gaussians

# pylint: disable=attribute-defined-outside-init
# pylint does not recognize attributes initialized as buffers in constructor
class LocalGlobalContextualPolicy(nn.Module):
    """A policy that outputs actions based on observation and both local & global latent context.
    
    This class tracks and controls both latent contexts, while providing methods to properly interact with the encoder and decoders
    Args: 
        latent_dim (int): Latent context variable dimension.
        global_encoder (garage.torch.embeddings.ContextEncoder): global context encoder.
        local_encoder (garage.torch.embeddings.ContextEncoder): local context encoder.
        transition_decoder (torch.nn.Module): transition reconstruction decoder
        reward_decoder (torch.nn.Module): reward decoder for global encoder
        state_decoder (torch.nn.Module): state decoder for global encoder
        global_recurrent_encoder (bool): Whether the global encoder is recurrent or not
        local_recurrent_encoder (bool): Whether the local encoder is recurrent or not
        policy (garage.torch.policies.Policy): Inner policy network used
        
        tanh_actions_before_sampling (bool): Whether to tanh transform the mean of the distribution (note: this is used in variBAD but not PEARL)
        use_next_obs (bool): True if next observation is used in context for distinguishing tasks; false otherwise.
        disable_local_encoder (bool): True if local encoder is not being used
        disable_global_encoder (bool): True if global encoder is not being used
        sample_embedding (bool): Whether to pass latent_z to the policy if True, or distribution params (latent_mean/var) if False (True by default)
        min_std (float): Minimum std for encoder distribution
    """

    def __init__(self, latent_dim, global_encoder, local_encoder, transition_decoder, reward_decoder, state_decoder, global_recurrent_encoder, local_recurrent_encoder,
                 policy, tanh_actions_before_sampling=False, use_next_obs = True,
                 disable_local_encoder=False, disable_global_encoder=False, local_kl_normal_prior = True, sample_global_embedding=True, sample_local_embedding=True, min_std=1e-6):
        super().__init__()
        self._latent_dim = latent_dim
        self._global_encoder = global_encoder
        self._local_encoder = local_encoder
        self._transition_decoder = transition_decoder
        self._reward_decoder = reward_decoder
        self._state_decoder = state_decoder
        self._policy = policy
        self._global_recurrent_encoder = global_recurrent_encoder
        self._local_recurrent_encoder = local_recurrent_encoder
        self._tanh_actions_before_sampling = tanh_actions_before_sampling
        self._use_next_obs = use_next_obs
        self._disable_local_encoder = disable_local_encoder
        self._disable_global_encoder = disable_global_encoder
        self._local_kl_normal_prior = local_kl_normal_prior
        self._sample_global_embedding = sample_global_embedding
        self._sample_local_embedding = sample_local_embedding
        self._min_std = min_std
        
        self.prev_local_means = None
        self.prev_local_vars = None

        # set "minimum std" to the value where softplus(x) = min_std
        self._min_std = torch.Tensor([min_std]).log().to(global_device()) if min_std is not None else None

        # This will create the following variables, which represent a posterior over tasks
        #   z_global, global_z_means, global_z_vars
        #   z_local, local_z_means, local_z_vars
        self.reset_global_belief()
        self.reset_local_belief()

    def reset_global_belief(self, num_tasks=1, batch_size=1):
        """Resets global belief to a N(0,1) prior with initial shape (global_batch_size, latent_dim). 
        May need to tinker with this if we ever use a recurrent global encoder
        """
        # mostly relevant if we have a recurrent encoder. Should not do anything else otherwise
        if self._global_recurrent_encoder:
            self._global_encoder.reset(batch_size=batch_size)

        #reset distribution over z to the prior
        mu = torch.zeros(num_tasks, self._latent_dim).to(global_device())
        var = torch.ones(num_tasks, self._latent_dim).to(global_device())
            
        if self._tanh_actions_before_sampling:
            self.global_z_means = torch.tanh(mu)
        else:
            self.global_z_means = mu 
        self.global_z_vars = var
        self.sample_global_from_belief()
        

    def reset_local_belief(self, batch_size=1, num_tasks=1):
        """Sample local belief to a prior generated by forwarding 0-context through the encoder with initial shape (num_tasks, batch_size, latent_dim). 
        seq_len should only be greater than 1 when using a recurrent encoder
        """
        if self._local_recurrent_encoder:
            self._local_encoder.reset(batch_size=batch_size)

        if self._local_recurrent_encoder:
            input_dim = self._local_encoder.gru.input_size
            mu, sigma_squared = self._local_encoder.forward(torch.zeros(num_tasks, batch_size, input_dim).to(global_device()))
        else:
            input_dim = self._local_encoder._layers[0][0].in_features # a bit hacky but should work
            output = self._local_encoder.forward(torch.zeros(num_tasks, batch_size, input_dim).to(global_device()))
            mu = output[...,:self._latent_dim]
            sigma_squared = output[...,self._latent_dim:]
        
        # Dims are (num_tasks, seq_len, latent_dim)
        if self._tanh_actions_before_sampling:
            self.local_z_means = torch.tanh(mu)
        else: 
            self.local_z_means = mu 
        if self._min_std is not None:
            sigma_squared = sigma_squared.clamp(min=self._min_std.item())
        self.local_z_vars =  F.softplus(sigma_squared)
                
        # Keep history of previous context if we are using a prior of previous belief instead of normal
        self.prev_local_context = None
        
        self.sample_local_from_belief()
        

    def sample_global_from_belief(self):
        """Sample z using distributions from current means and variances. 
        """
        posteriors = [
            torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(
                torch.unbind(self.global_z_means), torch.unbind(self.global_z_vars))
        ]
        z_global = [d.rsample() for d in posteriors]
        self.z_global = torch.stack(z_global)

    def sample_local_from_belief(self): 
        """Sample z using distributions from current means and variances. 
        """
        posteriors = [
            torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(
                torch.unbind(self.local_z_means), torch.unbind(self.local_z_vars))
        ]
        z_local = [d.rsample() for d in posteriors]
        self.z_local = torch.stack(z_local)

    def infer_global_posterior(self, context, detach_every=None):
        r"""Compute :math:`q(z \| c)` as a function of input context and sample new global z.
        Args:
            context (torch.Tensor): Context values, with shape
                :math:`(X, N, C)`. X is the number of tasks. N is batch size. C
                is the combined size of observation, action, reward, and next
                observation if next observation is used in context. Otherwise,
                C is the combined size of observation, action, and reward.
        """
        if self._global_recurrent_encoder:
            mu, sigma_squared = self._global_encoder.forward(context, detach_every=detach_every)
        else:
            output = self._global_encoder.forward(context)
            mu = output[...,:self._latent_dim]
            sigma_squared = output[...,self._latent_dim:]
        
        mu = mu.view(context.size(0), -1, self._latent_dim)
        sigma_squared = sigma_squared.view(context.size(0), -1, self._latent_dim)

        #transforms
        if self._tanh_actions_before_sampling:
            mu = torch.tanh(mu)
        if self._min_std is not None:
            sigma_squared = sigma_squared.clamp(min=self._min_std.item())
        sigma_squared = F.softplus(sigma_squared)

        if not self._global_recurrent_encoder: # Creating a permutation-invariant product of gaussians as per PEARL paper
            z_params = [
                product_of_gaussians(m, s)
                for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))
            ]
            self.global_z_means = torch.stack([p[0] for p in z_params]) # This should ensure that global_z_means.shape = (num_tasks, latent_dim)
            self.global_z_vars = torch.stack([p[1] for p in z_params])
        else:
            self.global_z_means = mu[:,-1,:] # This should ensure that global_z_means.shape = (num_tasks, latent_dim)
            self.global_z_vars = sigma_squared[:,-1,:]
            
        self.sample_global_from_belief()

    def infer_local_posterior(self, context, detach_every=None, save_prev_dist=False, prior_reset_inds=None):
        r"""Compute :math:`q(z \| c)` as a function of input context and sample new z.
        Args:
            context (torch.Tensor): Context values, with shape
                :math:`(X, N, C)`. X is the number of tasks. N is batch_size (NOT CONTEXT_BATCH_SIZE). C
                is the combined size of observation, action, reward, and next
                observation if next observation is used in context. Otherwise,
                C is the combined size of observation, action, and reward.
        """
        # Don't forget to reset the encoder hidden state (if RNN) before calling this where appropriate.
        # It is not hard-coded into here because there are instances where we may want to persist the hidden state inbetween calls (eg. rolling out)
        #if not self._local_kl_normal_prior: # if needed: save prior for calculating KL loss later
        #    self.prev_local_z_means = self.local_z_means
        #    self.prev_local_z_vars = self.local_z_vars
        
        if self._local_recurrent_encoder:
            mu, sigma_squared = self._local_encoder.forward(context, detach_every=detach_every)
        else:
            output = self._local_encoder.forward(context)
            mu = output[...,:self._latent_dim]
            sigma_squared = output[...,self._latent_dim:]
        
        # Dims are (num_tasks, seq_len, latent_dim)
        if self._tanh_actions_before_sampling:
            self.local_z_means = torch.tanh(mu)
        else: 
            self.local_z_means = mu 
        if self._min_std is not None:
            sigma_squared = sigma_squared.clamp(min=self._min_std.item())
        self.local_z_vars =  F.softplus(sigma_squared)
        
         # Update 11/04/2021 - save prior local context
        if save_prev_dist:  
            self.prev_local_means = self.local_z_means
            self.prev_local_vars = self.local_z_vars
        
        self.sample_local_from_belief()

    def forward(self, obs, context, detach_every=None):
        """Get actions and probs from policy, given obs and (context, local_context).
        We will derive z from context.
        Args:
            obs (torch.Tensor): Observation values, with shape
                :math:`(X, N, O)`. X is the sequence length. N is batch size. O
                 is the size of the flattened observation space.
            context (torch.Tensor): Context values, with shape
                :math:`(X, N, C)`. X is the sequence length. N is batch size. C
                is the combined size of observation, action, reward, and next
                observation if next observation is used in context. Otherwise,
                C is the combined size of observation, action, and reward.
        Returns:
            tuple:
                * torch.Tensor: Predicted action values.
                * np.ndarray: Mean of distribution.
                * np.ndarray: Log std of distribution.
                * torch.Tensor: Log likelihood of distribution.
                * torch.Tensor: Sampled values from distribution before
                    applying tanh transformation.
            torch.Tensor: z values, with shape :math:`(N, L)`. N is batch size.
                L is the latent dimension.
        """
        # Resize obs and task_z to prepare for forward pass through policy network
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        global_context, local_context = context 
        
        if not self._disable_global_encoder:
            # Calculate global_z from global context
            self.infer_global_posterior(global_context, detach_every=detach_every)
            self.sample_global_from_belief()
            
            #if self._sample_global_embedding:
            #    global_z = self.z_global # Here, global_z has dims (t, latent_dim)
            #else:
            #    global_z = torch.cat([self.global_z_means, self.global_z_vars], dim=-1) # Otherwide, dims (t, latent_dim*2)
            #global_z_reshaped = [z.repeat(b, 1) for z in global_z]
            #global_z_reshaped = torch.cat(global_z_reshaped, dim=0) # Dims: (t*b, latent_dim or latent_dim*2)
        else:
            global_z_reshaped = None
        
        if not self._disable_local_encoder:
            # Calculate local_z from local context
            self.infer_local_posterior(local_context, detach_every=detach_every)
            self.sample_local_from_belief()
            #if self._sample_local_embedding:
            #    local_z = self.z_local # Here, local_z has dims (t, b, latent_dim)
            #else:
            #    local_z = torch.cat([self.local_z_means, self.local_z_vars], dim=-1) # Otherwide, dims (t, b, latent_dim*2)
        else:
            local_z_reshaped = None 

        # Finally, get correct z to input into policy. 
        z_reshaped, global_z_reshaped, local_z_reshaped = self._get_z_for_policy()
        
        # Run policy, get log probs and new actions
        obs_z = torch.cat([obs, z_reshaped.detach()], dim=1)
        dist = self._policy(obs_z)[0]
        pre_tanh, actions = dist.rsample_with_pre_tanh_value() # actions has shape (task * batch_size, action_space)
        log_pi = dist.log_prob(value=actions, pre_tanh_value=pre_tanh)
        log_pi = log_pi.unsqueeze(1)
        mean = dist.mean.to('cpu').detach().numpy()
        log_std = (dist.variance**.5).log().to('cpu').detach().numpy()
        # z_reshaped, global_z_reshaped, local_z_reshaped all have dims (t*b, latent_dim) for respective z
        return (actions, mean, log_std, log_pi, pre_tanh, dist), z_reshaped, (global_z_reshaped, local_z_reshaped)

    def get_action(self, obs, context):
        """Sample action from the policy, conditioned on the task embedding. Context is only passed through for later training purposes
        Args:
            obs (torch.Tensor): Observation values, with shape :math:`(1, O)`.
                O is the size of the flattened observation space.
        Returns:
            torch.Tensor: Output action value, with shape :math:`(1, A)`.
                A is the size of the flattened action space.
            dict:
                * latent_z: latent z samples
                * mean: np.ndarray[float]: Mean of the ACTION distribution.
                * log_std: np.ndarray[float]: Standard deviation of logarithmic values
                    of the ACTION distribution.
        """
        if context is None: #zero-fill context if it's None. Should only occur if local encoder is disabled 
            context = torch.zeros_like(self.z_global.squeeze()).to(global_device())
        
        """
        # Reshape inputs. z here has dims (batch_size, latent_dim*2) where batchsize=1 during rollouts
        if self._sample_global_embedding:
            global_z = self.z_global.view(1,-1)
        else: # If we are not sampling embeddings, we are using the params of the normal dist for the policy
            global_z = torch.cat([self.global_z_means.view(1, -1), self.global_z_vars.view(1, -1)], dim=-1)
        if self._sample_local_embedding:
            local_z = self.z_local.view(1,-1)
        else: # If we are not sampling embeddings, we are using the params of the normal dist for the policy
            local_z = torch.cat([self.local_z_means.view(1, -1), self.local_z_vars.view(1, -1)], dim=-1)
        
        if self._disable_local_encoder:
            z = global_z
        elif self._disable_global_encoder:
            z = local_z
        else:
            z = torch.cat([global_z, local_z], dim=-1).view(1,-1)
        """
        
        z, _, _ = self._get_z_for_policy(is_rollout=True)
        
        # Run obs and z through policy
        obs = torch.as_tensor(obs[None], device=global_device()).float()
        obs_in = torch.cat([obs, z], dim=-1)
        action, info = self._policy.get_action(obs_in)
        
        # Update 11/04/2023
        # Here, if prev_local_context is None (after resets only) then we replace prev_local_context with 0s. 
        # It ultimately should not matter as we remove it anyway later
        if self.prev_local_context is None:
            self.prev_local_context = torch.zeros_like(context)
        
        # Indicator variable to make things much, much easier later
        if torch.eq(self.prev_local_context.to(global_device()), torch.zeros_like(context).to(global_device())).all():
            prior_reset_ind = torch.Tensor([1])
        else:
            prior_reset_ind = torch.Tensor([0])

        # Add relevant info to buffer in case it's useful for training
        info['global_z'] = self.z_global.squeeze().detach().cpu().numpy()
        if not self._disable_local_encoder:
            info['local_z'] = self.z_local.squeeze().detach().cpu().numpy()
            info['local_context'] = context.detach().cpu().numpy()
            info['prev_local_context'] = self.prev_local_context.detach().cpu().numpy()
            info['prior_reset_ind'] = prior_reset_ind.cpu().numpy()
        else: #pass through a tensor of 0s if we are disabling the local encoder
            info['local_z'] = torch.zeros_like(self.z_global.squeeze()).detach().cpu().numpy()
            info['local_context'] = torch.zeros_like(self.z_global.squeeze()).detach().cpu().numpy()
            info['prev_local_context'] = torch.zeros_like(self.z_global.squeeze()).detach().cpu().numpy()
            info['prior_reset_ind'] = prior_reset_ind.cpu().numpy()

        self.prev_local_context = context # Update 11/04/2023
        
        return action, info

    def compute_kl_div(self, global_kl, local_kl, dones):
        r"""Compute :math:`KL(q(z_t|c_t) \| p(z), where p(z) is a N(0,I) prior`. 
        NOTE: assumes we have run infer_global_posterior() and infer_local_posterior() with relevant context first!!
        
        prev_local_z_mean/var only used if our local prior is NOT N(0,1)
        Returns:
            float: :math:`KL(q(z|c) \| p(z))`.
        """
        global_prior = torch.distributions.Normal(torch.zeros(self._latent_dim).to(global_device()), 
                                                    torch.ones(self._latent_dim).to(global_device()))
        if global_kl:
            global_posterior = [
            torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(
                torch.unbind(self.global_z_means), torch.unbind(self.global_z_vars))
            ]
            global_kl_divs = [
            torch.distributions.kl.kl_divergence(post, global_prior)
            for post in global_posterior
            ]
            global_kl_div = torch.sum(torch.stack(global_kl_divs))
        else: 
            global_kl_div = torch.Tensor([0]).to(global_device())

        if local_kl: # To replicate the same calc as in global_kl, we will flatten the local_z params to get a (t*b, latent_dim) shape 
            t, b, _ = self.local_z_means.size()
            local_mu = self.local_z_means.view(t * b, -1)
            local_var = self.local_z_vars.view(t * b, -1)
            local_post = torch.distributions.Normal(local_mu, torch.sqrt(local_var))
            
            if self._local_kl_normal_prior:
                local_prior = torch.distributions.Normal(torch.zeros(self._latent_dim).to(global_device()), 
                                                    torch.ones(self._latent_dim).to(global_device()))        
                
                # This code should be equivalent to above global_kl calc but significantly more efficient
                local_kl_divs = torch.distributions.kl.kl_divergence(local_post, local_prior)            
                local_kl_div = torch.sum(local_kl_divs)
            else:
                if self.prev_local_means is not None and self.prev_local_vars is not None:
                    prev_mu = self.prev_local_means
                    prev_var = self.prev_local_vars
                else:
                    # First, get inds where dones occurred. Only relevant when we have multiple runs in a batch 
                    inds = torch.where(dones[:,:-1,:].squeeze()) #filter out final step because we will be adding 1 later
                
                    # Then, shift local_z_means/vars and insert a N(0,1) dist at t=0 to create the prior
                    prev_mu = torch.cat([torch.zeros_like(self.local_z_means[:,0,:]).unsqueeze(1).to(global_device())
                                    ,self.local_z_means[:,:-1,:]], dim=1)
                    prev_var = torch.cat([torch.ones_like(self.local_z_vars[:,0,:]).unsqueeze(1).to(global_device())
                                    ,self.local_z_vars[:,:-1,:]], dim=1)
                    
                    # And also replace the prior with N(0,1) when we reset the env
                    prev_mu[inds[0],inds[1]+1,:] = torch.zeros_like(self.local_z_means[0,0,:]).to(global_device())
                    prev_var[inds[0],inds[1]+1,:] = torch.ones_like(self.local_z_vars[0,0,:]).to(global_device())
                
                local_prior = torch.distributions.Normal(prev_mu.view(t * b, -1), torch.sqrt(prev_var.view(t * b, -1)))
                local_kl_divs = torch.distributions.kl.kl_divergence(local_post, local_prior)            
                local_kl_div = torch.sum(local_kl_divs)
        else:
            local_kl_div = torch.Tensor([0]).to(global_device())

        return (global_kl_div, local_kl_div)

    # Helper function to make code above readable
    def _get_z_for_policy(self, is_rollout=False):
        """Returns the z that should be used for the policy, based on given inputs.
        self.infer_global_posterior() and self.infer_local_posterior() must be run separately
        is_rollout simplifies calcs to make this fn more efficient during rollout
        Returns z in long form (batch_size, dims), along with global_z and local_z separately (in policy-ready format)
        """
        # z_global and its params have dim size (_, latent_dim)
        assert len(self.z_global.shape) == 2
        # z_local and its params have dim size (_, _, latent_dim)
        assert len (self.z_local.shape) == 3
        
        t, b, _ = self.z_local.size()
        
        if self._sample_global_embedding:
            global_z = self.z_global
        else: # If we are not sampling embeddings, we are using the params of the normal dist for the policy
            global_z = torch.cat([self.global_z_means, self.global_z_vars], dim=-1)
        if not is_rollout:
            global_z = [z.repeat(b, 1) for z in global_z]
            global_z = torch.cat(global_z, dim=0) # Dims: (t*b, latent_dim or latent_dim*2)
            
        if self._sample_local_embedding:
            local_z = self.z_local.view(t*b,-1)
        else: # If we are not sampling embeddings, we are using the params of the normal dist for the policy
            local_z = torch.cat([self.local_z_means.view(t*b,-1), self.local_z_vars.view(t*b,-1)], dim=-1)
        
        if self._disable_local_encoder:
            z = global_z
        elif self._disable_global_encoder:
            z = local_z
        else:
            z = torch.cat([global_z, local_z], dim=-1)
        
        return z, global_z, local_z

    @property
    def networks(self):
        """Return context_encoder and policy.

        Returns:
            list: Encoder and policy networks.

        """
        #if self._local_recurrent_encoder:
        #    return [self._global_encoder, self._local_encoder,
        #        self._transition_decoder, self._reward_decoder, self._state_decoder,
        #        self._policy, self._local_encoder._hidden_state]
        #else:
        return [self._global_encoder, self._local_encoder,
                self._transition_decoder, self._reward_decoder, self._state_decoder,
                self._policy]

    @property
    def context(self):
        """Return context.

        Returns:
            torch.Tensor: Context values, with shape :math:`(X, N, C)`.
                X is the number of tasks. N is batch size. C is the combined
                size of observation, action, reward, and next observation if
                next observation is used in context. Otherwise, C is the
                combined size of observation, action, and reward.

        """
        return self._global_context
