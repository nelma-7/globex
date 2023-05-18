"""Global-Local Embeddings for Contextual meta-RL
Uses garage package 
Final version for submitted code
"""

import copy

import akro
from dowel import logger
from dowel import tabular
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from garage import EnvSpec, InOutSpec, StepType, EpisodeBatch, log_multitask_performance
from garage.np.algos import MetaRLAlgorithm
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker
from garage.torch import global_device
from garage.torch.embeddings import MLPEncoder
from garage.experiment.deterministic import get_seed, set_seed
from garage.sampler import _apply_env_update

#custom classes
from decoder import MLPDecoder
from encoder import GRURecurrentEncoder
from globex_policy import LocalGlobalContextualPolicy 
from evaluation import customMetaEvaluator
from mi_estimators import CLUB

class GLOBEX(MetaRLAlgorithm):
    r"""A new algorithm; utilises two encoders to separate local and global effects

    Args:
        env(list[Environment]): Batch of sampled environment updates(EnvUpdate), 
            which, when invoked on environments, will configure them with new tasks.
        inner_policy (garage.torch.policies.Policy): Policy.
        qf (torch.nn.Module): Q-function (used in SAC calculations)
        vf (torch.nn.Module): Value function.
        sampler (garage.sampler.Sampler): Sampler, controls how to sample tasks
        test_env_sampler (garage.experiment.SetTaskSampler): Sampler for test tasks
        global_recurrent_encoder (bool): Whether global encoder is recurrent
        local_recurrent_encoder (bool): Whether local encoder is recurrent
        optimizer_class (type): Type of optimizer for training networks.
        discount (float): RL discount factor.
        reward_scale (int): Reward scale.
        
        # Training params
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int or None): Number of tasks for testing.
        latent_dim (int): Size of latent context vector.
        encoder_hidden_sizes (list[int]): Output dimension of dense layer(s) of the context encoder.
        decoder_hidden_sizes (list[int]): Output dimension of dense layer(s) of context decoder(s).
        num_iter_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before training.
        num_tasks_sample (int): Number of random tasks to obtain data for each iteration.
        num_steps_prior (int): Number of transitions to obtain per task with z~prior
        num_steps_posterior (int): Number of transitions to obtain per task with z~posterior (add to context buffer)
        num_extra_rl_steps_posterior (int): Number of transitions to obtain per task with z~posterior (do not add to context buffer)
        batch_size (int): Number of transitions in RL batch.
        num_tasks_sample (int): Number of train tasks to obtain data from each epoch
        meta_batch_size (int): The number of tasks sampled per training epoch
        
        # Policy (SAC) params
        policy_lr (float): Policy learning rate.
        qf_lr (float): Q-function learning rate.
        vf_lr (float): Value function learning rate.
        policy_mean_reg_coeff (float): Policy mean regulation weight.
        policy_std_reg_coeff (float): Policy std regulation weight.
        policy_pre_activation_coeff (float): Policy pre-activation weight.
        soft_target_tau (float): Interpolation parameter for doing the soft target update.
        replay_buffer_size (int): Maximum samples in replay buffer.
        policy_max_grad_norm (float): Maximum gradient update for policy
        
        # Local/Global encoding params
        context_lr (float): Inference network learning rate.
        context_batch_size (int): number of transitions sampled in global & local encoder training 
        context_tbptt_size (int): number of transitions for truncated backprop through time. Only relevant for recurrent encoders.
        context_buffer_size (int): Size of replay buffer for local & global encoder 
        global_kl_lambda (float): KL lambda value for global encoder.
        local_kl_lambda (float): KL lambda for local encoder. We also scale this down by context_batch_size 
            so that global_kl_loss and local_kl_loss is at roughly the same scale
        reward_loss_coefficient (float): Coefficient for reward loss
        state_loss_coefficient (float): Coefficient for state loss
        transition_reconstruction_coefficient (float): Coefficient for transition reconstruction loss
        mi_loss_coefficient (float): Coefficient for MI loss
        encoder_max_grad_norm (float): Maximum gradient update for encoders
        decoder_max_grad_norm (float): Maximum gradient update for decoders
        encoder_min_std (float): Minimum std for latent variables
                        
        # OTHER HYPERPARAMS
        global_nonlinearity (torch.nn.functional): nonlinearity used for global encoder
        local_nonlinearity (torch.nn.functional): nonlinearity used for global encoder
        decoder_nonlinearity (torch.nn.functional): nonlinearity used for global encoder
        epochs_per_eval (int): Number of epochs per eval run
        n_exploration_eps (int): Number of exploration episodes before updating global context 
        n_test_episodes (int): Number of exploitation episodes after updating global context
            
        # ABLATION PARAMETERS
        # Note that this should be identical to PEARL if decode_state, decode_reward = False and policy_loss_through_enc, disable_local_encoder = True
        use_next_obs_in_context (bool): Whether to use next_obs in context
        decode_state (bool): Whether to decode state in global encoder training
        decode_reward (bool): Whether to decode reward in global encoder training
        policy_loss_through_enc (bool): Whether to pass SAC Q-function loss through encoders during training (PEARL-style)
        disable_local_encoder (bool): Whether to entirely disable local encoder, and use global encoder only
        disable_global_encoder (bool): Whether to entirely disable global encoder, and use local encoder only        
    """

    # pylint: disable=too-many-statements
    def __init__(
            self,
            env,
            inner_policy,
            qf,
            vf,
            sampler,
            test_env_sampler,
            global_recurrent_encoder,
            local_recurrent_encoder,
            optimizer_class=torch.optim.Adam,
            discount=0.99,
            reward_scale=5.0,

            # Training params. Defaults taken from PEARL implementation
            num_train_tasks=100,
            num_test_tasks=100,
            latent_dim=5,
            encoder_hidden_sizes=(200,200,200),
            decoder_hidden_sizes=(200,200,200),
            num_iter_per_epoch=2000,
            num_initial_steps=2000, 
            num_steps_prior=400,
            num_steps_posterior=0,
            num_extra_rl_steps_posterior=600, 
            batch_size=256,
            num_tasks_sample=5, #Number of train tasks to obtain data from each epoch
            meta_batch_size=16, #Number of train tasks to update per training iteration. There are num_iter_per_epoch training iterations per epoch (as we train off-policy)
            
            #SAC Policy params
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            policy_mean_reg_coeff=1E-3,
            policy_std_reg_coeff=1E-3,
            policy_pre_activation_coeff=0.,
            soft_target_tau=0.005,
            replay_buffer_size=100000,
            policy_max_grad_norm = None, # By default, we don't cap grad norms
            
            # LOCAL/GLOBAL ENCODING PARAMS
            context_lr=3E-4,
            context_batch_size=100,
            context_tbptt_size=None,
            context_buffer_size=100000,
            global_kl_lambda=.1,
            local_kl_lambda=.05,
            local_kl_normal_prior=True, #NEW: True=N(0,1) prior for local KL, False=prev transaction prior
            reward_loss_coefficient=1,
            state_loss_coefficient=1,
            transition_reconstruction_coefficient=1,
            mi_loss_coefficient=0.1,
            global_max_grad_norm = None,
            local_max_grad_norm = None,
            decoder_max_grad_norm = None,
            mi_max_grad_norm = None,
            encoder_min_std = 1e-10,
            sample_global_embedding = True, #whether to sample global_z (True), or pass the parameters of the normal dist to the policy (False)
            sample_local_embedding = True, #whether to sample local_z (True), or pass the parameters of the normal dist to the policy (False)
            
            # OTHER HYPERPARAMS
            global_nonlinearity = nn.ReLU,
            local_nonlinearity = nn.ReLU,
            decoder_nonlinearity = nn.ReLU,
            epochs_per_eval=1, # Num of epochs per eval run
            n_exploration_eps=2, #using oyster defaults
            n_test_episodes=1,
            save_grads=False, #saves a list of grad_norms that applies to each iteration
            
            # ABLATION PARAMETERS
            # Note that this should be identical to PEARL if decode_state, decode_reward = False and policy_loss_through_enc, disable_local_encoder = True
            use_next_obs_in_context=True, # True by default as it only makes sense for local context
            decode_state=False, # Whether to decode state in global encoder training
            decode_reward=True, # Whether to decode reward in global encoder training
            policy_loss_through_global=False,
            policy_loss_through_local=False,
            pearl_validation=False, #for one quick check haha
            disable_local_encoder=False,
            disable_global_encoder=False
            ):
        
        self._env = env
        self._qf1 = qf
        self._qf2 = copy.deepcopy(qf)
        self._vf = vf
        self._sampler = sampler
        self._discount = discount
        self._reward_scale = reward_scale
        # Ablation Params
        if disable_local_encoder and disable_global_encoder:
            raise ValueError('you cannot disable both global and local encoders')
        self._use_next_obs_in_context = use_next_obs_in_context
        self._policy_loss_through_local = policy_loss_through_local
        self._policy_loss_through_global = policy_loss_through_global
        self._disable_local_encoder = disable_local_encoder
        self._disable_global_encoder = disable_global_encoder
        self._pearl_validation = pearl_validation
        self._save_grads = save_grads
        # Training Params
        self._num_train_tasks = num_train_tasks
        self._latent_dim = latent_dim
        self._num_iter_per_epoch = num_iter_per_epoch
        self._num_initial_steps = num_initial_steps
        self._num_steps_prior = num_steps_prior
        self._num_steps_posterior = num_steps_posterior
        self._num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self._batch_size = batch_size
        # Sampling Params
        self._num_tasks_sample = num_tasks_sample
        self._meta_batch_size = meta_batch_size
        self._num_test_tasks = num_test_tasks
        if num_test_tasks is None:
            num_test_tasks = test_env_sampler.n_tasks
        if num_test_tasks is None:
            raise ValueError('num_test_tasks must be provided if '
                             'test_env_sampler.n_tasks is None')
        # Policy Params
        self._policy_mean_reg_coeff = policy_mean_reg_coeff
        self._policy_std_reg_coeff = policy_std_reg_coeff
        self._policy_pre_activation_coeff = policy_pre_activation_coeff
        self._soft_target_tau = soft_target_tau
        self._replay_buffer_size = replay_buffer_size
        self._policy_max_grad_norm = policy_max_grad_norm
        self.target_vf = copy.deepcopy(self._vf)
        self.vf_criterion = torch.nn.MSELoss()
        # Encoding Params
        self._global_recurrent_encoder = global_recurrent_encoder
        self._local_recurrent_encoder = local_recurrent_encoder
        self._decode_state = decode_state
        self._decode_reward = decode_reward
        self._local_kl_normal_prior = local_kl_normal_prior
        self._context_tbptt_size = context_tbptt_size
        self._context_batch_size = context_batch_size
        self._context_buffer_size = context_buffer_size
        self._global_kl_lambda = global_kl_lambda
        self._local_kl_lambda = local_kl_lambda / context_batch_size
        self._reward_loss_coefficient = reward_loss_coefficient
        self._state_loss_coefficient = state_loss_coefficient
        self._transition_reconstruction_coefficient = transition_reconstruction_coefficient
        self._mi_loss_coefficient = mi_loss_coefficient
        self._global_max_grad_norm = global_max_grad_norm
        self._local_max_grad_norm = local_max_grad_norm
        self._decoder_max_grad_norm = decoder_max_grad_norm
        self._mi_max_grad_norm = mi_max_grad_norm
        self._encoder_min_std = encoder_min_std
        self._sample_global_embedding = sample_global_embedding
        self._sample_local_embedding = sample_local_embedding
        # Quick error check for simplicity in calculations when using recurrent encoder, given how everything is set up
        if (context_tbptt_size is not None) and (context_batch_size % context_tbptt_size != 0 or batch_size % context_tbptt_size != 0):
            raise ValueError('if doing truncated backprop through time, context_tbptt must be a factor of both context_batch_size and batch_size')
        self._task_idx = None
        self._single_env = env[0]()
        self.max_episode_length = self._single_env.spec.max_episode_length
        self._is_resuming = False

        # Set up meta evaluator class
        worker_args = dict(deterministic=True, disable_local_encoder=disable_local_encoder,
                           use_next_obs_in_context=use_next_obs_in_context)
        self._epochs_per_eval = epochs_per_eval
        self._evaluator = customMetaEvaluator(test_task_sampler=test_env_sampler,
                                        worker_class=CustomWorker,
                                        worker_args=worker_args,
                                        n_test_tasks=num_test_tasks,
                                        n_exploration_eps=n_exploration_eps,
                                        n_test_episodes=n_test_episodes) 

        # Set up encoders
        encoder_spec = self.get_env_spec(self._single_env, latent_dim,
                                         'encoder', use_next_obs_in_context=use_next_obs_in_context, 
                                         disable_local_encoder=disable_local_encoder, disable_global_encoder=disable_global_encoder)
        encoder_in_dim = int(np.prod(encoder_spec.input_space.shape))
        encoder_out_dim = int(np.prod(encoder_spec.output_space.shape))
        if self._global_recurrent_encoder:
            global_encoder = GRURecurrentEncoder(input_dim=encoder_in_dim,
                                        output_dims=latent_dim,
                                        hidden_nonlinearity=global_nonlinearity,
                                        hidden_size=encoder_hidden_sizes[0])
        else: 
            global_encoder = MLPEncoder(input_dim=encoder_in_dim,
                                            output_dim=encoder_out_dim,
                                            hidden_nonlinearity=global_nonlinearity,
                                            hidden_sizes=encoder_hidden_sizes)
        
        if self._local_recurrent_encoder:
            local_encoder = GRURecurrentEncoder(input_dim=encoder_in_dim,
                                            output_dims=encoder_out_dim,
                                            hidden_nonlinearity=local_nonlinearity,
                                            hidden_size=encoder_hidden_sizes[0])
        else:
            local_encoder = MLPEncoder(input_dim=encoder_in_dim,
                                            output_dim=encoder_out_dim,
                                            hidden_nonlinearity=local_nonlinearity,
                                            hidden_sizes=encoder_hidden_sizes)

        # Set up decoders (transition reconstruction, reward/state global decoder)
        transition_decoder_spec = self.get_env_spec(self._single_env, latent_dim,
                                         'transition_decoder', use_next_obs_in_context=use_next_obs_in_context, 
                                         disable_local_encoder=disable_local_encoder, disable_global_encoder=disable_global_encoder)
        transition_in_dim = int(np.prod(transition_decoder_spec.input_space.shape))
        transition_out_dim = int(np.prod(transition_decoder_spec.output_space.shape))
        transition_decoder = MLPDecoder(input_dim=transition_in_dim,
                                        output_dim=transition_out_dim,
                                        hidden_nonlinearity=decoder_nonlinearity,
                                        hidden_sizes=decoder_hidden_sizes)
        reward_decoder_spec = self.get_env_spec(self._single_env, latent_dim,
                                            'reward_decoder', use_next_obs_in_context=use_next_obs_in_context, 
                                            disable_local_encoder=disable_local_encoder, disable_global_encoder=disable_global_encoder)
        reward_in_dim = int(np.prod(reward_decoder_spec.input_space.shape))
        reward_out_dim = int(np.prod(reward_decoder_spec.output_space.shape))
        reward_decoder = MLPDecoder(input_dim=reward_in_dim,
                                        output_dim=reward_out_dim,
                                        hidden_nonlinearity=decoder_nonlinearity,
                                        hidden_sizes=decoder_hidden_sizes)
        state_decoder_spec = self.get_env_spec(self._single_env, latent_dim,
                                         'state_decoder', use_next_obs_in_context=use_next_obs_in_context, 
                                         disable_local_encoder=disable_local_encoder, disable_global_encoder=disable_global_encoder)
        state_in_dim = int(np.prod(state_decoder_spec.input_space.shape))
        state_out_dim = int(np.prod(state_decoder_spec.output_space.shape))
        state_decoder = MLPDecoder(input_dim=state_in_dim,
                                        output_dim=state_out_dim,
                                        hidden_nonlinearity=decoder_nonlinearity,
                                        hidden_sizes=decoder_hidden_sizes)
        
        # MI estimator
        self.mi_estimator = CLUB(latent_dim, latent_dim, hidden_size=64) #hardcode for now
        
        # Instantiate policy
        self._policy = LocalGlobalContextualPolicy(
            latent_dim=latent_dim,
            global_encoder=global_encoder,
            local_encoder=local_encoder,
            transition_decoder=transition_decoder,
            reward_decoder=reward_decoder,
            state_decoder=state_decoder,
            global_recurrent_encoder=global_recurrent_encoder,
            local_recurrent_encoder=local_recurrent_encoder,
            policy=inner_policy,
            use_next_obs=use_next_obs_in_context,
            min_std=encoder_min_std,
            disable_global_encoder=disable_global_encoder,
            disable_local_encoder=disable_local_encoder,
            sample_global_embedding=sample_global_embedding,
            sample_local_embedding=sample_local_embedding,
            local_kl_normal_prior=local_kl_normal_prior)

        # buffer for training RL update
        self._replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }
        self._context_replay_buffers = {
            i: PathBuffer(context_buffer_size)
            for i in range(num_train_tasks)
        }
        
        # Optimisers. Note that self._policy._networks = 
        # [self._global_encoder, self._local_encoder, self._transition_decoder, self._state_decoder, self._reward_decoder, self._policy]
        self.policy_optimizer = optimizer_class(
            self._policy.networks[5].parameters(),
            lr=policy_lr,
        )
        # Qf and vf optimisers for SAC
        self.qf1_optimizer = optimizer_class(
            self._qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self._qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self._vf.parameters(),
            lr=vf_lr,
        )
        
        # Local/Global encoding optimisers. Separated as we have multiple loss functions that will affect different networks
        dual_encoder_params = []
        if not disable_global_encoder:
            dual_encoder_params += [*self._policy.networks[0].parameters()] #global encoder
        if not disable_local_encoder:
            dual_encoder_params += [*self._policy.networks[1].parameters(), #local encoder
                                    *self._policy.networks[2].parameters()] #transition decoder
        if decode_reward:
            dual_encoder_params += [*self._policy.networks[3].parameters()] #reward decoder
        if decode_state:
            dual_encoder_params += [*self._policy.networks[4].parameters()] #state decoder
            
        self.local_global_optimizer = optimizer_class(
            dual_encoder_params,
            lr=context_lr
        )
        
        self.mi_optimizer = optimizer_class(
            self.mi_estimator.parameters(),
            lr=context_lr 
        )

        #debugging!
        self.q_grad_norms = []
        self.policy_grad_norms = []
        self.local_enc_grad_norms = []
        self.global_enc_grad_norms = []
        self.transition_dec_grad_norms = []
        self.mi_grad_norms = []

    def train(self, trainer):
        """Obtain samples, train, and evaluate for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`Trainer..step_epochs()`, which provides services
                such as snapshotting and sampler control.

        """
        for _ in trainer.step_epochs():
            epoch = trainer.step_itr 
            logger.log('Gathering samples...')
            # Obtain initial set of samples from all train tasks
            if epoch == 0 or self._is_resuming:
                for idx in range(self._num_train_tasks):
                    self._task_idx = idx
                    self._obtain_samples(trainer, epoch,
                                         self._num_initial_steps)
                    self._is_resuming = False

            # Log paths for train error
            train_paths = []
            prior_paths, posterior_paths, extra_paths = [], [], []
            for i in range(self._num_tasks_sample):
                idx = np.random.randint(self._num_train_tasks)
                self._task_idx = idx
                self._context_replay_buffers[idx].clear()
                #self._replay_buffers[idx].clear() #UPDATE - see what happens if we keep replay buffer mostly "on-policy"
                # obtain samples with z ~ prior
                if self._num_steps_prior > 0:
                    prior_paths = self._obtain_samples(trainer, epoch, self._num_steps_prior)
                # obtain samples with z ~ posterior AND add to context replay buffer
                if self._num_steps_posterior > 0:
                    posterior_paths = self._obtain_samples(trainer, epoch, self._num_steps_posterior, update_posterior=True, add_to_global_enc=True)
                # obtain samples with z ~ posterior BUT DO NOT add to context replay buffer
                if self._num_extra_rl_steps_posterior > 0:
                    extra_paths = self._obtain_samples(trainer, epoch, self._num_extra_rl_steps_posterior, update_posterior=True, add_to_global_enc=False)
                train_paths.extend(posterior_paths + extra_paths) #do not gather prior as it's all exploration paths
            
            if trainer.step_itr % self._epochs_per_eval == 0:
                train_paths = EpisodeBatch.from_list(self._single_env.spec,
                                                    train_paths)
                log_multitask_performance(self._evaluator._eval_itr, train_paths, self._discount)

            logger.log('Training...')
            # sample train tasks and optimize networks
            self._train_once(epoch=trainer.step_itr)

            if trainer.step_itr % self._epochs_per_eval == 0: #Only eval each epochs_per_eval iteration for performance
                logger.log('Evaluating...')
                # evaluate
                if not self._disable_global_encoder:
                    self._policy.reset_global_belief()
                if not self._disable_local_encoder:
                    self._policy.reset_local_belief(1)
                self._evaluator.evaluate(self)
            else:
                logger.log('Skipping Evaluation (only eval every ' + str(self._epochs_per_eval) + ' epochs)...')
            trainer.step_itr += 1

    def _train_once(self, epoch):
        """Perform one iteration of training."""
        for a in range(self._num_iter_per_epoch):
            indices = np.random.choice(range(self._num_train_tasks),
                                       self._meta_batch_size)
                
            total_autoencoder_loss, reward_reconstruction_loss, state_reconstruction_loss, transition_loss, mi_loss, global_kl_loss, local_kl_loss, qf_loss, vf_loss, policy_loss, \
            policy_mean, policy_log_std, log_pi, global_enc_mean, global_enc_vars, local_enc_mean, local_enc_vars = self._optimize_policy(indices)    
            """
            #inline print statements for small-scale debugging. Comment out when actually running things!!
            print("===============RUN",a,"===============")
            print("Policy Loss:", policy_loss)
            print("QF Loss:", qf_loss)
            print("total autoencoder loss:", total_autoencoder_loss)
            print("reward_reconstruction_loss:", reward_reconstruction_loss)
            print('transition_loss:', transition_loss)
            print('mi_loss:', mi_loss)
            print('global_kl_loss:', global_kl_loss) 
            print('local_kl_loss:', local_kl_loss)
            """
        # Save grads as well!
        if self._save_grads:
            q_grad_mean = torch.hstack(self.q_grad_norms[-self._num_iter_per_epoch:]).mean()
            policy_grad_mean = torch.hstack(self.policy_grad_norms[-self._num_iter_per_epoch:]).mean()
            local_enc_grad_mean = torch.hstack(self.local_enc_grad_norms[-self._num_iter_per_epoch:]).mean()
            global_enc_grad_mean = torch.hstack(self.global_enc_grad_norms[-self._num_iter_per_epoch:]).mean()
            transition_dec_grad_mean = torch.hstack(self.transition_dec_grad_norms[-self._num_iter_per_epoch:]).mean()
            mi_grad_mean = torch.hstack(self.mi_grad_norms[-self._num_iter_per_epoch:]).mean()

        # Log performance
        if epoch % self._epochs_per_eval == 0:
            with tabular.prefix('Losses'):
                tabular.record('/QFunctionLoss', qf_loss.item())
                tabular.record('/ValueFunctionLoss', vf_loss.item())
                tabular.record('/PolicyLoss', policy_loss.item())
                tabular.record('/TotalLocalGlobalLoss', total_autoencoder_loss.item())
                if not self._disable_global_encoder:
                    tabular.record('/GlobalKLLoss', global_kl_loss.item())
                    tabular.record('/GlobalKLLossWithCoef', (global_kl_loss*self._global_kl_lambda).item())
                if self._decode_reward:
                    tabular.record('/GlobalRewardLoss', reward_reconstruction_loss.item())
                    tabular.record('/GlobalRewardLossWithCoef', (reward_reconstruction_loss*self._reward_loss_coefficient).item())
                if self._decode_state:
                    tabular.record('/GlobalStateLoss', (state_reconstruction_loss*self._state_loss_coefficient).item())
                if not self._disable_local_encoder:
                    tabular.record('/TransitionLoss', transition_loss.item())
                    tabular.record('/TransitionLossWithCoef', (transition_loss*self._transition_reconstruction_coefficient).item())
                    tabular.record('/CounterfactualMILoss', mi_loss.item())
                    tabular.record('/CounterfactualMILossWithCoef', (mi_loss*self._mi_loss_coefficient).item())
                    tabular.record('/LocalKLLoss', local_kl_loss.item())
                    tabular.record('/LocalKLLossWithCoef', (local_kl_loss*self._local_kl_lambda).item())
            with tabular.prefix('Policy'):
                tabular.record('/ActionLogProb', log_pi.item())
                tabular.record('/PolicyMean', policy_mean.item())
                tabular.record('/PolicyStd', policy_log_std.item())
            with tabular.prefix('Encoder'):
                if not self._disable_global_encoder:
                    tabular.record('/GlobalEncoderMean', global_enc_mean.item())
                    tabular.record('/GlobalEncoderVar', global_enc_vars.item())
                if not self._disable_local_encoder:
                    tabular.record('/LocalEncoderMean', local_enc_mean.item())
                    tabular.record('/LocalEncoderVar', local_enc_vars.item())
            if self._save_grads:
                with tabular.prefix('Gradients'):
                    tabular.record('/QGradNorm', q_grad_mean.item())
                    tabular.record('/PolicyGradNorm', policy_grad_mean.item())
                    tabular.record('/LocalEncGradNorm', local_enc_grad_mean.item())
                    tabular.record('/GlobalEncGradNorm', global_enc_grad_mean.item())
                    tabular.record('/TransitionDecGradNorm', transition_dec_grad_mean.item())
                    tabular.record('/MIGradNorm', mi_grad_mean.item())

    def _optimize_policy(self, indices):
        """Perform algorithm optimizing.
        
        Unless running ablations, we will train the policy on the RL batch (treating global_z and local_z as inputs),
         and train the global & local encoder together on the context batch
        
        Args:
            indices (list): Tasks used for training.

        """
        ######################################
        ## Get transition and context data, and reset beliefs
        ######################################  
        num_tasks = len(indices)
        obs_context, act_context, rew_context, no_context, dones_context, prior_local_context, prior_reset_inds = self._sample_context(indices) 
        context = torch.cat([obs_context, act_context, rew_context], dim=-1)
        if self._use_next_obs_in_context:
            context = torch.cat([context, no_context], dim=-1)
    
        # clear context and reset belief of policy
        if not self._disable_global_encoder:
            self._policy.reset_global_belief(num_tasks=num_tasks)
        if not self._disable_local_encoder:
            self._policy.reset_local_belief(batch_size=self._batch_size if self._context_tbptt_size is None else self._context_tbptt_size, num_tasks=num_tasks)

        # data shape is (task, batch, feat)
        obs, actions, rewards, next_obs, terms, local_context = self._sample_data(indices)
        
        ######################################
        ## Update Policy (and value, q functions)
        ##  Note: code from garage implementation of PEARL
        ######################################   
        # This code infers posterior for global and local z using the context batch for global_z, and the saved local context for local_z
        policy_outputs, z_reshaped, latent_z = self._policy(obs, context=(context, local_context), detach_every=self._context_tbptt_size) 
        global_z_reshaped, local_z_reshaped = latent_z # This will be in the correct format for inputting into the policy
        new_actions, policy_mean, policy_log_std, log_pi, pre_tanh = policy_outputs[:5]

        # flatten out the task dimension. 
        t, b, _ = obs.size()
        obs_reshaped = obs.view(t * b, -1)
        actions_reshaped = actions.view(t * b, -1)
        rewards_reshaped = rewards.view(t * b, -1)
        rewards_reshaped = rewards_reshaped * self._reward_scale
        next_obs_reshaped = next_obs.view(t * b, -1)
        terms_reshaped = terms.view(t * b, -1)
        
        # Rebuild z_reshaped to detach local_z and global_z where necessary
        if not self._policy_loss_through_local and not self._disable_local_encoder:
            local_z_reshaped = local_z_reshaped.detach()
        if not self._policy_loss_through_global and not self._disable_global_encoder:
            global_z_reshaped = global_z_reshaped.detach()
        if self._disable_local_encoder:
            z_reshaped = global_z_reshaped
        elif self._disable_global_encoder:
            z_reshaped = local_z_reshaped
        else:
            z_reshaped = torch.cat([global_z_reshaped, local_z_reshaped], dim=-1)
            
        # optimize qf and encoder networks
        q1_pred = self._qf1(torch.cat([obs_reshaped, actions_reshaped], dim=1), z_reshaped)
        q2_pred = self._qf2(torch.cat([obs_reshaped, actions_reshaped], dim=1), z_reshaped)
        v_pred = self._vf(obs_reshaped, z_reshaped.detach())

        with torch.no_grad():
            target_v_values = self.target_vf(next_obs_reshaped, z_reshaped)

        # If running policy loss through encoders, get encoder losses here. Otherwise, they will be calculated at the end
        if (self._policy_loss_through_local or self._policy_loss_through_global):
            if not self._disable_local_encoder: # First, re-calculate local z using sampled context
                if not self._local_kl_normal_prior: # Re-calculate prior dist if we are using bayesian filtering prior for local
                    self._policy.reset_local_belief(batch_size=self._context_batch_size if self._context_tbptt_size is None else self._context_tbptt_size, num_tasks=num_tasks)
                    self._policy.infer_local_posterior(prior_local_context, detach_every = self._context_tbptt_size, save_prev_dist=True, prior_reset_inds = prior_reset_inds)
                self._policy.reset_local_belief(batch_size=self._context_batch_size if self._context_tbptt_size is None else self._context_tbptt_size, num_tasks=num_tasks)
                self._policy.infer_local_posterior(context, detach_every = self._context_tbptt_size)
            self.local_global_optimizer.zero_grad()
            total_loss, reward_reconstruction_loss, state_reconstruction_loss, transition_loss, mi_loss, global_kl_loss, local_kl_loss, \
            global_enc_mean, global_enc_vars, local_enc_mean, local_enc_vars = self._encoder_decoder_losses(obs_context, act_context, rew_context, no_context, dones_context) 
            total_loss.backward(retain_graph=True)

        #debug
        #global_grad1 = self._policy._global_encoder._layers[0][0].weight.grad.clone()
        #local_grad1 = self._policy._local_encoder._layers[0][0].weight.grad.clone()
        #tdecoder_grad1 = self._policy._transition_decoder._layers[0][0].weight.grad.clone()
        
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        q_target = rewards_reshaped + (
            1. - terms_reshaped) * self._discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target)**2) + torch.mean(
            (q2_pred - q_target)**2)

        qf_loss.backward()
        if self._save_grads: 
            self.q_grad_norms.append(torch.cat([param.grad.view(-1) for param in self._qf1.parameters()]).norm(2)) 
        #debug
        #global_grad2 = self._policy._global_encoder._layers[0][0].weight.grad.clone()
        #local_grad2 = self._policy._local_encoder._layers[0][0].weight.grad.clone()
        #tdecoder_grad2 = self._policy._transition_decoder._layers[0][0].weight.grad.clone()
        # test: torch.equal(global_grad1, global_grad2) should be false
        # test: torch.equal(local_grad1, local_grad2) should be true
        
        self.qf1_optimizer.step() 
        self.qf2_optimizer.step() 
        if (self._policy_loss_through_local or self._policy_loss_through_global):
            if self._global_max_grad_norm is not None:
                if not self._disable_global_encoder:
                    clip_grad_norm_(self._policy.networks[0].parameters(), self._global_max_grad_norm)
            if self._local_max_grad_norm is not None:
                if not self._disable_local_encoder:
                    clip_grad_norm_(self._policy.networks[1].parameters(), self._local_max_grad_norm)
            if self._decoder_max_grad_norm is not None:
                clip_grad_norm_(self._policy.networks[2].parameters(), self._decoder_max_grad_norm)
                if self._decode_reward:
                    clip_grad_norm_(self._policy.networks[3].parameters(), self._decoder_max_grad_norm)
                if self._decode_state: 
                    clip_grad_norm_(self._policy.networks[4].parameters(), self._decoder_max_grad_norm)
            if self._save_grads:
                self.global_enc_grad_norms.append(torch.cat([param.grad.view(-1) for param in self._policy.networks[0].parameters()]).norm(2)) 
                self.local_enc_grad_norms.append(torch.cat([param.grad.view(-1) for param in self._policy.networks[1].parameters()]).norm(2)) 
                self.transition_dec_grad_norms.append(torch.cat([param.grad.view(-1) for param in self._policy.networks[2].parameters()]).norm(2)) 
            self.local_global_optimizer.step() 
                
        # compute min Q on the new actions. z is always detached here
        q1 = self._qf1(torch.cat([obs_reshaped, new_actions], dim=1), z_reshaped.detach())
        q2 = self._qf2(torch.cat([obs_reshaped, new_actions], dim=1), z_reshaped.detach())
        min_q = torch.min(q1, q2)

        # optimize vf
        v_target = min_q - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss = self._policy_loss(min_q, policy_mean, policy_log_std, log_pi, pre_tanh)
        policy_loss.backward()
        if self._policy_max_grad_norm is not None: #clip actual gradients - helps stabilise training
                clip_grad_norm_(self._policy.networks[5].parameters(), self._policy_max_grad_norm)
        if self._save_grads:
            self.policy_grad_norms.append(torch.cat([param.grad.view(-1) for param in self._policy.networks[5].parameters() if param.grad is not None]).norm(2)) 
        self.policy_optimizer.step()
        
        # UPDATE GLOBAL/LOCAL ENCODERS/DECODERS HERE (unless running grads thru policy)
        if not (self._policy_loss_through_local or self._policy_loss_through_global):
            if not self._disable_local_encoder: # First, re-calculate local z using sampled context
                if not self._local_kl_normal_prior:
                    self._policy.reset_local_belief(batch_size=self._context_batch_size if self._context_tbptt_size is None else self._context_tbptt_size, num_tasks=num_tasks)
                    self._policy.infer_local_posterior(prior_local_context, detach_every = self._context_tbptt_size, save_prev_dist=True, prior_reset_inds = prior_reset_inds)
                self._policy.reset_local_belief(batch_size=self._context_batch_size if self._context_tbptt_size is None else self._context_tbptt_size, num_tasks=num_tasks)
                self._policy.infer_local_posterior(context, detach_every = self._context_tbptt_size)
            self.local_global_optimizer.zero_grad()
            
            total_loss, reward_reconstruction_loss, state_reconstruction_loss, transition_loss, mi_loss, global_kl_loss, local_kl_loss, \
            global_enc_mean, global_enc_vars, local_enc_mean, local_enc_vars = self._encoder_decoder_losses(obs_context, act_context, rew_context, no_context, dones_context)
            total_loss.backward()
            if self._global_max_grad_norm is not None:
                if not self._disable_global_encoder:
                    clip_grad_norm_(self._policy.networks[0].parameters(), self._global_max_grad_norm)
            if self._local_max_grad_norm is not None:
                if not self._disable_local_encoder:
                    clip_grad_norm_(self._policy.networks[1].parameters(), self._local_max_grad_norm)
            if self._decoder_max_grad_norm is not None:
                clip_grad_norm_(self._policy.networks[2].parameters(), self._decoder_max_grad_norm)
                if self._decode_reward:
                    clip_grad_norm_(self._policy.networks[3].parameters(), self._decoder_max_grad_norm)
                if self._decode_state: 
                    clip_grad_norm_(self._policy.networks[4].parameters(), self._decoder_max_grad_norm)
            if self._save_grads:
                self.global_enc_grad_norms.append(torch.cat([param.grad.view(-1) for param in self._policy.networks[0].parameters()]).norm(2)) 
                self.local_enc_grad_norms.append(torch.cat([param.grad.view(-1) for param in self._policy.networks[1].parameters()]).norm(2)) 
                self.transition_dec_grad_norms.append(torch.cat([param.grad.view(-1) for param in self._policy.networks[2].parameters()]).norm(2)) 
            self.local_global_optimizer.step()
        # Return a bunch of logging metrics
        return total_loss, reward_reconstruction_loss, state_reconstruction_loss, transition_loss, mi_loss, global_kl_loss, local_kl_loss, qf_loss, vf_loss, policy_loss, \
                policy_mean.mean(), policy_log_std.mean(), log_pi.mean(), global_enc_mean, global_enc_vars, local_enc_mean, local_enc_vars

    def _policy_loss(self, min_q, policy_mean, policy_log_std, log_pi, pre_tanh):
        """Calculate SAC policy loss. Code from garage implementation of PEARL
        """
        log_policy_target = min_q
        policy_loss = (log_pi - log_policy_target).mean()

        mean_reg_loss = self._policy_mean_reg_coeff * (policy_mean**2).mean()
        std_reg_loss = self._policy_std_reg_coeff * (policy_log_std**2).mean()
        pre_activation_reg_loss = self._policy_pre_activation_coeff * (
            (pre_tanh**2).sum(dim=1).mean())
        policy_reg_loss = (mean_reg_loss + std_reg_loss +
                           pre_activation_reg_loss)
        policy_loss = policy_loss + policy_reg_loss
        return policy_loss
        
    def _encoder_decoder_losses(self, obs, act, rew, next_obs, dones): #NOTE 01/04/2023: CHANGES HAPPENED HERE
        """Single function for encoder-decoder loss. 
            This function will either be called before updating the policy if we are passing RL loss through the encoder,
            or after all policy/qf/vf updates if we are not
            
            Inputs:
            - obs (num_tasks, context_batch_size, obs_dim)
            - act (num_tasks, context_batch_size, act_dim)
            - rew (num_tasks, context_batch_size, batch_size, rew_dim)
            - next_obs (num_tasks, context_batch_size, obs_dim)
            - prev_local_z_mean (TBD)
            - prev_local_z_var (TBD)  
            """
        # Reshape everything to flatten dims
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        act = act.view(t * b, -1)
        rew = rew.view(t * b, -1)
        rew = rew * self._reward_scale
        next_obs = next_obs.view(t * b, -1)
        # Save encoder means/vars for logging purposes
        if self._disable_global_encoder:
            global_enc_mean = torch.Tensor([0]).to(global_device())
            global_enc_vars = torch.Tensor([0]).to(global_device())
        else:
            global_enc_mean = self._policy.global_z_means.mean() 
            global_enc_vars = self._policy.global_z_vars.mean()
        if self._disable_local_encoder:
            local_enc_mean = torch.Tensor([0]).to(global_device())
            local_enc_vars = torch.Tensor([0]).to(global_device())
        else:      
            local_enc_mean = self._policy.local_z_means.mean()
            local_enc_vars = self._policy.local_z_vars.mean()
                
        # optimise our encoder-decoder networks
        # For ablations: if we disable local then we only run global reconstruction + KL (similar, but not exactly PEARL)
        #                if we disable global then we run transition ONLY
        if self._disable_local_encoder:
            reward_reconstruction_loss, state_reconstruction_loss = self._global_reconstruction_loss(self._policy.z_global, obs, act, rew, next_obs)
            global_kl_loss, local_kl_loss = self._policy.compute_kl_div(global_kl=True, local_kl=False, dones=dones) #NOTE 01/04/2023: CHANGES HAPPENED HERE
            transition_loss = torch.Tensor([0]).to(global_device())
            mi_loss = torch.Tensor([0]).to(global_device())
        elif self._disable_global_encoder:
            transition_loss = self._transition_reconstruction_loss(obs, act, rew, next_obs, self._policy.z_global, self._policy.z_local)
            global_kl_loss, local_kl_loss = self._policy.compute_kl_div(global_kl=False, local_kl=True, dones=dones) #NOTE 01/04/2023: CHANGES HAPPENED HERE
            reward_reconstruction_loss = torch.Tensor([0]).to(global_device())
            state_reconstruction_loss = torch.Tensor([0]).to(global_device())
            mi_loss = torch.Tensor([0]).to(global_device())
        else: #regular runs
            reward_reconstruction_loss, state_reconstruction_loss = self._global_reconstruction_loss(self._policy.z_global, obs, act, rew, next_obs)
            transition_loss = self._transition_reconstruction_loss(obs, act, rew, next_obs, self._policy.z_global, self._policy.z_local)  
            global_kl_loss, local_kl_loss = self._policy.compute_kl_div(global_kl=True, local_kl=True, dones=dones) #NOTE 01/04/2023: CHANGES HAPPENED HERE
            mi_loss = self._mutual_information_loss(self._policy.z_global, self._policy.z_local) 

        # Calculate total loss. Note that values will be 0 if they are ablated due to code above
        total_loss = (self._transition_reconstruction_coefficient * transition_loss + 
                      self._reward_loss_coefficient * reward_reconstruction_loss +
                      self._state_loss_coefficient * state_reconstruction_loss +
                      self._global_kl_lambda * global_kl_loss +
                      self._local_kl_lambda * local_kl_loss +    
                      self._mi_loss_coefficient * mi_loss)
                
        return total_loss, reward_reconstruction_loss, state_reconstruction_loss, transition_loss, mi_loss, global_kl_loss, local_kl_loss, \
            global_enc_mean, global_enc_vars, local_enc_mean, local_enc_vars
        
    def _global_reconstruction_loss(self, global_z, obs, actions, reward, next_obs):
        """Calculate global reconstruction loss. 
        
        For now we follow the method of Bing et al (2020) and only try to predict reward/obs at the current timestep

        Args:
            global_z (meta_batch_size, latent_dim): latent embedding from the global encoder
            obs (meta_batch_size * context_batch_size, feature_space): observations
            actions (meta_batch_size * context_batch_size, feature_space): actions
            reward (meta_batch_size * context_batch_size, 1): rewards 
            next_obs: next observations. only used if decode_state=True

        """
        if not (self._decode_reward or self._decode_state): # Efficiency code to skip chunk if needed
            return torch.Tensor([0]).to(global_device()), torch.Tensor([0]).to(global_device())
        # Reshape inputs
        z_reshaped = [z.repeat(self._context_batch_size, 1) for z in global_z]
        z_reshaped = torch.cat(z_reshaped, dim=0)
        
        if self._decode_reward:
            reward_pred = self._policy._reward_decoder(torch.cat([obs, actions, z_reshaped], dim=-1))
            reward_loss = torch.mean((reward_pred - reward)**2)
        else:
            reward_loss = torch.Tensor([0]).to(global_device())

        if self._decode_state:
            state_pred = self._policy._state_decoder(torch.cat([obs, actions, z_reshaped], dim=-1))
            state_loss = torch.mean((state_pred - next_obs)**2)
        else:
            state_loss = torch.Tensor([0]).to(global_device())

        return reward_loss, state_loss


    def _transition_reconstruction_loss(self, obs, actions, rewards, next_obs, global_z, local_z):
        """Calculate reconstruction loss. 

        Args:
            global_z (meta_batch_size, latent_dim): latent embedding from the global encoder
            local_z (meta_batch_size, context_batch_size, latent_dim)
            obs (meta_batch_size*context_batch_size, feature_space): observations. We need to flatten dims before calling this
            actions (meta_batch_size*context_batch_size, feature_space): actions
            reward (meta_batch_size*context_batch_size, 1): rewards 
            next_obs: next observations. only used if decode_state=True
        """
        # Make sure z uses the correct input depending on ablation
        if self._disable_local_encoder: 
            #note that the current code will skip this function if self._disable_local_encoder=True
            t, b, _ = obs.size()        
            global_z = [z.repeat(b, 1) for z in global_z]
            global_z = torch.cat(global_z, dim=0)
            z_reshaped = global_z
        elif self._disable_global_encoder:
            t, b, _ = local_z.size()  
            z_reshaped = local_z.view(t * b, -1)
        else: # Normal runs
            t, b, _ = local_z.size()        
            global_z = [z.repeat(b, 1) for z in global_z]
            global_z = torch.cat(global_z, dim=0)
            local_z = local_z.view(t * b, -1)
            z_reshaped = torch.cat([global_z, local_z], dim=-1)
        
        obs_dim = obs.shape[-1]
        action_dim = actions.shape[-1]
        reward_dim = 1 
        
        # Then, decode z
        decoded = self._policy._transition_decoder(z_reshaped)
        obs_decoded = decoded[..., : obs_dim]
        action_decoded = decoded[..., obs_dim : obs_dim + action_dim]
        reward_decoded = decoded[..., obs_dim+action_dim : obs_dim + action_dim + reward_dim]
        if self._use_next_obs_in_context:
            next_obs_decoded = decoded[..., obs_dim+action_dim+reward_dim : obs_dim + action_dim + reward_dim + obs_dim]

        # Calculate loss
        obs_loss = torch.mean((obs_decoded - obs)**2)
        action_loss = torch.mean((action_decoded-actions)**2)
        reward_loss = torch.mean((reward_decoded-rewards)**2)
        reconstruction_loss = obs_loss + action_loss + reward_loss #TODO: add coefficients? should be ok though
        if self._use_next_obs_in_context:
            next_obs_loss = torch.mean((next_obs_decoded-next_obs)**2)
            reconstruction_loss += next_obs_loss
            
        return reconstruction_loss
        
    def _mutual_information_loss(self, global_z, local_z):
        """UPDATE - calculate MI loss via a variational estimate with vCLUB (Cheng et al 2020)
        
        Args:
            global_z (meta_batch_size, latent_dim)
            local_z (meta_batch_size, context_batch_size, latent_dim)
        """
        # First, resize global, local_z
        t, b, _ = local_z.size()        
        global_z = [z.repeat(b, 1) for z in global_z]
        global_z = torch.cat(global_z, dim=0)
        local_z = local_z.view(t * b, -1)
        
        # First, run a few iterations of descent on the vCLUB estimator 
        #for i in range(3):
        #TODO: If we want to do multiple iterations properly, we will want to be able to sample global_z and local_z without replacing existing values
        self.mi_estimator.train() # Set the MI estimator into training mode
        self.mi_estimator.requires_grad_(True) # Here, we want to calculate gradients
        mi_network_loss = self.mi_estimator.learning_loss(global_z.detach(), local_z.detach()) 
        self.mi_optimizer.zero_grad()
        mi_network_loss.backward()
        if self._mi_max_grad_norm is not None: #try clipping this instead
            clip_grad_norm_(self.mi_estimator.parameters(), self._mi_max_grad_norm)
        if self._save_grads:
            self.mi_grad_norms.append(torch.cat([param.grad.view(-1) for param in self.mi_estimator.parameters()]).norm(2)) 
        self.mi_optimizer.step()
        
        # Next, use the vCLUB estimator to compute MI
        self.mi_estimator.eval() # Set MI estimator into eval mode in case it uses weird layers
        self.mi_estimator.requires_grad_(False) # Do not calculate gradients on the estimator when calculating the MI loss
        mi_loss = self.mi_estimator(global_z, local_z) 
        return mi_loss
    
    def _obtain_samples(self,
                        trainer,
                        itr,
                        num_samples,
                        update_posterior=False,
                        add_to_global_enc=True):
        """Obtain samples for policy training.

        Args:
            trainer (Trainer): Trainer.
            itr (int): Index of iteration (epoch).
            num_samples (int): Number of samples to obtain.
            update_posterior_rate (int): How often (in episodes) to infer
                posterior of policy.
            add_to_enc_buffer (bool): Whether or not to add samples to encoder
                buffer.

        """
        if not self._disable_global_encoder:
            self._policy.reset_global_belief() #reset batchsizes to 1 when gathering samples
        if not self._disable_local_encoder:
            self._policy.reset_local_belief(1)
        total_samples = 0

        returned_paths = []
        if update_posterior: #if we are updating the posterior we want to go one traj at a time
            sample_batchsize = self.max_episode_length
        else:
            sample_batchsize = num_samples
        while total_samples < num_samples:
            paths = trainer.obtain_samples(itr, sample_batchsize,
                                           self._policy,
                                           self._env[self._task_idx])
            total_samples += sum([len(path['rewards']) for path in paths])
            returned_paths.extend(paths)

            for path in paths: 
                p = {
                    'observations':
                    path['observations'],  #"prev_obs" - obs used to obtain action
                    'actions':
                    path['actions'],  # action taken based on observation
                    'rewards':
                    path['rewards'].reshape(-1, 1),  #reward from taking that particular action
                    'next_observations': 
                    path['next_observations'], # "next_obs" - obs obtained as a result of that action
                    'dones':
                    np.array([
                        step_type == StepType.TERMINAL
                        for step_type in path['step_types']
                    ]).reshape(-1, 1),
                    'global_z':
                    path['agent_infos']['global_z'].squeeze(), #global z used across the whole path (will not change)
                    'local_z':
                    path['agent_infos']['local_z'].squeeze(), #local z generated as a function of (o_t, a_t, r_{t-1})
                    #TODO: consider storing a list of previous T transitions to extend local encoder
                    'local_context':
                    path['agent_infos']['local_context'].squeeze(), #context in shape (num_steps, context_dims) is 0 if local encoder is disabled
                    'prev_local_context':
                    path['agent_infos']['prev_local_context'].squeeze(), # previous context in shape (num_steps, context_dims) is 0 if local encoder is disabled
                    'prior_reset_ind':
                    path['agent_infos']['prior_reset_ind'].reshape(-1,1), #1 if prev_local_context is the N(0,1) prior, 0 otherwise
                }
                self._replay_buffers[self._task_idx].add_path(p)

                if add_to_global_enc:
                    self._context_replay_buffers[self._task_idx].add_path(p)

            if update_posterior: #only persist memory across episodes if we are updating the posterior
                obs, act, rew, no, _, _, _ = self._sample_context(self._task_idx, batch_size=self._context_batch_size) #dones do not matter for global context
                context = torch.cat([obs, act, rew], dim=-1)
                if self._use_next_obs_in_context:
                    context = torch.cat([context, no], dim=-1)
                self._policy.infer_global_posterior(context, detach_every = self._context_tbptt_size)
            
        return returned_paths

    def _sample_data(self, indices):
        """Sample batch of training data from a list of tasks.

        Note that if either local encoder is recurrent, then we need to sample full paths in order to calc local encodings correctly
        Global encoder does not matter here, as we will never run a RL batch through the global encoder
        
        Args:
            indices (list): List of task indices to sample from.

        Returns:
            torch.Tensor: Obervations, with shape :math:`(X, N, O^*)` where X
                is the number of tasks. N is batch size.
            torch.Tensor: Actions, with shape :math:`(X, N, A^*)`.
            torch.Tensor: Rewards, with shape :math:`(X, N, 1)`.
            torch.Tensor: Next obervations, with shape :math:`(X, N, O^*)`.
            torch.Tensor: Dones, with shape :math:`(X, N, 1)`.

        """
        # transitions sampled randomly from replay buffer
        initialized = False
        for idx in indices:
            if self._local_recurrent_encoder: # collect FULL paths until you have at least batch_size transitions
                batch = {} # This will be the aggregated batch
                paths = [] # This will collect paths
                l = 0
                while l < self._batch_size:
                    path = self._replay_buffers[idx].sample_path()
                    paths.append(path)
                    l += path['observations'].shape[0]
                for k in path.keys(): # Collect those full paths together into one batch
                    batch[k] = np.vstack([p[k] for p in paths])
            else:
                batch = self._replay_buffers[idx].sample_transitions(
                self._batch_size)
            if not initialized:
                o = batch['observations'][np.newaxis]
                a = batch['actions'][np.newaxis]
                r = batch['rewards'][np.newaxis]
                no = batch['next_observations'][np.newaxis]
                d = batch['dones'][np.newaxis]
                if not self._disable_local_encoder:
                    lc = batch['local_context'][np.newaxis]
                initialized = True
            else:
                o = np.vstack((o, batch['observations'][np.newaxis]))
                a = np.vstack((a, batch['actions'][np.newaxis]))
                r = np.vstack((r, batch['rewards'][np.newaxis]))
                no = np.vstack((no, batch['next_observations'][np.newaxis]))
                d = np.vstack((d, batch['dones'][np.newaxis]))
                if not self._disable_local_encoder:
                    lc = np.vstack((lc, batch['local_context'][np.newaxis]))

        o = torch.as_tensor(o, device=global_device()).float()
        a = torch.as_tensor(a, device=global_device()).float()
        r = torch.as_tensor(r, device=global_device()).float()
        no = torch.as_tensor(no, device=global_device()).float()
        d = torch.as_tensor(d, device=global_device()).float()
        if not self._disable_local_encoder:
            lc = torch.as_tensor(lc, device=global_device()).float()
        else:
            lc = torch.Tensor([0]).to(global_device())
        return o, a, r, no, d, lc

    def _sample_context(self, indices, batch_size=None):
        """Sample batch of context from a list of tasks.
        If either global or local encoders are recurrent then we need to sample full paths instead
        We also sample a full path if our local prior takes the previous local belief (instead of a simple N(0,1) at all times), in order to properly calc KL loss

        Args:
            indices (list): List of task indices to sample from.

        Returns:
            torch.Tensor: Context data, with shape :math:`(X, N, C)`. X is the
                number of tasks. N is batch size. C is the combined size of
                observation, action, reward, and next observation if next
                observation is used in context. Otherwise, C is the combined
                size of observation, action, and reward.

        """
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        if batch_size is None:
            batch_size = self._context_batch_size
        initialized = False
        for idx in indices:
            if self._local_recurrent_encoder or self._global_recurrent_encoder or self._pearl_validation: # collect FULL paths until you have at least batch_size transitions
                batch = {} # This will be the aggregated batch
                paths = [] # This will collect paths
                l = 0
                while l < batch_size:
                    path = self._context_replay_buffers[idx].sample_path()
                    #NOTE 04/04/2023: Changes here (always set final step to done=True so local_kl is correctly calculated later)
                    path['dones'][-1] = True
                    paths.append(path)
                    l += path['observations'].shape[0]
                for k in path.keys():
                    batch[k] = np.vstack([p[k] for p in paths])
            else:
                batch = self._context_replay_buffers[idx].sample_transitions(
                    batch_size) 
            o = batch['observations'][:batch_size,:] #filtering only matters if the environment supports early termination
            a = batch['actions'][:batch_size,:]
            r = batch['rewards'][:batch_size,:]
            no = batch['next_observations'][:batch_size,:]
            d = batch['dones'][:batch_size,:] 
            pc = batch['prev_local_context'][:batch_size,:] 
            id = batch['prior_reset_ind'][:batch_size,:] 

            if not initialized:
                obs = o[np.newaxis]
                action = a[np.newaxis]
                rew = r[np.newaxis]
                next_obs = no[np.newaxis]
                dones = d[np.newaxis] 
                prior_local_context = pc[np.newaxis]
                prior_reset_ind = id[np.newaxis]
                initialized = True
            else:
                obs = np.vstack((obs, o[np.newaxis]))
                action = np.vstack((action, a[np.newaxis]))
                rew = np.vstack((rew, r[np.newaxis]))
                next_obs = np.vstack((next_obs, no[np.newaxis]))
                dones = np.vstack((dones, d[np.newaxis])) 
                prior_local_context = np.vstack((prior_local_context, pc[np.newaxis]))
                prior_reset_ind = np.vstack((prior_reset_ind, id[np.newaxis]))

        obs = torch.as_tensor(obs, device=global_device()).float()
        action = torch.as_tensor(action, device=global_device()).float()
        rew = torch.as_tensor(rew, device=global_device()).float()
        next_obs = torch.as_tensor(next_obs, device=global_device()).float()
        dones = torch.as_tensor(dones, device=global_device()).float()
        prior_reset_ind = torch.as_tensor(prior_reset_ind, device=global_device()).float()
        if self._local_kl_normal_prior: #ignore, return 0s
            prior_local_context = torch.Tensor([0]).to(global_device())
        else:
            prior_local_context = torch.as_tensor(prior_local_context, device=global_device()).float()
            
        return (obs, action, rew, next_obs, dones, prior_local_context, prior_reset_ind) 

    def _update_target_network(self):
        """Update parameters in the target vf network."""
        for target_param, param in zip(self.target_vf.parameters(),
                                       self._vf.parameters()):
            target_param.data.copy_(target_param.data *
                                    (1.0 - self._soft_target_tau) +
                                    param.data * self._soft_target_tau)

    @property
    def policy(self):
        """Return the policy within the model.

        Returns:
            garage.torch.policies.Policy: Policy within the model.

        """
        return self._policy

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return self._policy.networks + [self._policy] + [
            self._qf1, self._qf2, self._vf, self.target_vf
        ] + [self.mi_estimator]

    def get_exploration_policy(self):
        """Return a policy used before adaptation to a specific task.

        Each time it is retrieved, this policy should only be evaluated in one
        task.

        Returns:
            Policy: The policy used to obtain samples that are later used for
                meta-RL adaptation.

        """
        return self._policy

    def adapt_policy(self, exploration_policy, exploration_episodes):
        """Produce a policy adapted for a task.

        Args:
            exploration_policy (Policy): A policy which was returned from
                get_exploration_policy(), and which generated
                exploration_episodes by interacting with an environment.
                The caller may not use this object after passing it into this
                method.
            exploration_episodes (EpisodeBatch): Episodes to which to adapt,
                generated by exploration_policy exploring the
                environment.

        Returns:
            Policy: A policy adapted to the task represented by the
                exploration_episodes.

        """
        total_steps = sum(exploration_episodes.lengths)
        o = exploration_episodes.observations
        a = exploration_episodes.actions
        r = exploration_episodes.rewards.reshape(total_steps, 1)
        no = exploration_episodes.next_observations
        if self._use_next_obs_in_context:
            ctxt = np.hstack((o, a, r, no)).reshape(1, total_steps, -1)
        else:
            ctxt = np.hstack((o, a, r)).reshape(1, total_steps, -1)
        context = torch.as_tensor(ctxt, device=global_device()).float()
        self._policy.infer_global_posterior(context)
        return self._policy

    @classmethod
    def augment_env_spec(cls, env_spec, latent_dim, disable_local_encoder, disable_global_encoder, sample_global_embedding=True, sample_local_embedding=True):
        """Augment environment by a size of latent dimension. Used for qf only.
        Args:
            env_spec (EnvSpec): Environment specs to be augmented.
            latent_dim (int): Latent dimension.

        Returns:
            EnvSpec: Augmented environment specs.
        """
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        # determine actual latent dim
        if disable_global_encoder:
            global_latent_dim = 0
        elif sample_global_embedding:
            global_latent_dim = latent_dim
        else:
            global_latent_dim = latent_dim*2
        if disable_local_encoder:
            local_latent_dim = 0
        elif sample_local_embedding:
            local_latent_dim = latent_dim
        else:
            local_latent_dim = latent_dim*2
        actual_latent_dim = global_latent_dim+local_latent_dim
        aug_obs_dim = obs_dim + actual_latent_dim
        aug_obs = akro.Box(low=-1,
                           high=1,
                           shape=(aug_obs_dim, ),
                           dtype=np.float32)
        aug_act = akro.Box(low=-1,
                           high=1,
                           shape=(action_dim, ),
                           dtype=np.float32)
        return EnvSpec(aug_obs, aug_act)

    @classmethod
    def get_env_spec(cls, env_spec, latent_dim, module, use_next_obs_in_context, disable_local_encoder, disable_global_encoder, 
                     sample_global_embedding=True, sample_local_embedding=True):
        """Get environment specs of encoder with latent dimension.
        Args:
            env_spec (EnvSpec): Environment specification.
            latent_dim (int): Latent dimension.
            module (str): Module to get environment specs for.

        Returns:
            InOutSpec: Module environment specs with latent dimension.
        """
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        # determine actual latent dim for policy/vf. Decoders always take the sampled latents.
        if disable_global_encoder:
            global_latent_dim = 0
        elif sample_global_embedding:
            global_latent_dim = latent_dim
        else:
            global_latent_dim = latent_dim*2
        if disable_local_encoder:
            local_latent_dim = 0
        elif sample_local_embedding:
            local_latent_dim = latent_dim
        else:
            local_latent_dim = latent_dim*2
        actual_latent_dim = global_latent_dim+local_latent_dim
        
        if module == 'encoder': # Encoder takes in context (obs, act, rew) and outputs mean/var params of latent
            in_dim = obs_dim + action_dim + 1
            if use_next_obs_in_context:
                in_dim += obs_dim
            out_dim = latent_dim * 2
        elif module == 'vf': # vf takes in observation + latents and outputs value
            in_dim = obs_dim + actual_latent_dim
            out_dim = 1
        elif module == 'transition_decoder': # transition decoder takes in latents and outputs context
            in_dim = latent_dim*2
            if disable_local_encoder:
                in_dim -= latent_dim
            if disable_global_encoder:
                in_dim -= latent_dim
            out_dim = obs_dim + action_dim + 1
            if use_next_obs_in_context:
                out_dim += obs_dim
        elif module == 'reward_decoder': # reward decoder takes in obs, action and global_z and outputs reward
            in_dim = obs_dim + action_dim + latent_dim 
            out_dim = 1
        elif module == 'state_decoder': # state decoder takes in obs, action and global_z and outputs next_obs
            in_dim = obs_dim + action_dim + latent_dim 
            out_dim = obs_dim
        in_space = akro.Box(low=-1, high=1, shape=(in_dim, ), dtype=np.float32)
        out_space = akro.Box(low=-1,
                             high=1,
                             shape=(out_dim, ),
                             dtype=np.float32)
        spec = InOutSpec(in_space, out_space)
        return spec
    
    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        device = device or global_device()
        for net in self.networks:
            net.to(device)
    
    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        data = self.__dict__.copy()
        del data['_replay_buffers']
        del data['_context_replay_buffers']
        return data

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        self.__dict__.update(state)
        self._replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }

        self._context_replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }
        self._is_resuming = True


class CustomWorker(DefaultWorker):
    """Custom worker for new algo. It will:
        - Store local and global context
        - Sample global context at the beginning of the episode
        - Update local context at each step

    Args:
        seed (int): The seed to use to intialize random number generators. Technically unused
        max_episode_length(int or float): The maximum length of episodes which
            will be sampled. Can be (floating point) infinity.
        worker_number (int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        deterministic (bool): If True, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.
        use_next_obs_in_context (bool): Whether to use next_obs when constructing context
        disable_local_encoder (bool): Ablation parameter

    Attributes:
        agent (Policy or None): The worker's agent.
        env (Environment or None): The worker's environment.
    """
    def __init__(self,
                 *,
                 seed, # Technically unused
                 max_episode_length,
                 worker_number,
                 deterministic=False, # This is set to True when rolling out at test-time
                 use_next_obs_in_context=False,
                 disable_local_encoder=False):
        self._deterministic = deterministic
        self._episode_info = None
        self._disable_local_encoder = disable_local_encoder
        self._use_next_obs_in_context = use_next_obs_in_context
        self.counter = 0 # Dummy counter used when setting worker and env seed. This ensures a "consistent" but reproducible randomness in env initialisation 
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)

    def worker_init(self):
        """Initialize a worker."""
        if self._seed is not None:
            set_seed(self._seed + self._worker_number)
        if self.env is not None and self.env.unwrapped.__class__.__name__ not in ['PointEnv', 'SparsePointEnv']: # a bit hacky, but we won't want to set env seed for basic envs:
            self.env.seed(get_seed() + self.counter)
            self.env.action_space.seed(get_seed() + self.counter)
            self.counter += 1 # we increment this counter to ensure envs change but in a predictable way
            
    def update_env(self, env_update):
        """Overrides the update_env function of superclass. We do this to make sure env seeds are set correctly
        """
        self.env, _ = _apply_env_update(self.env, env_update)
        if self.env.unwrapped.__class__.__name__ not in ['PointEnv', 'SparsePointEnv']: # a bit hacky, but we won't want to set env seed for basic envs
            self.env.seed(get_seed() + self.counter)
            self.env.action_space.seed(get_seed() + self.counter)
            self.counter += 1 # we increment this counter to ensure envs change but in a predictable way
        
    def start_episode(self):
        """Begin a new episode."""
        if not self._disable_local_encoder:
            self.agent.reset_local_belief(1) #because we are running 1 at a time, batch size should always be 1
        self._eps_length = 0
        self._prev_obs, self._episode_info = self.env.reset()
        context_dims = self.env.observation_space.shape[0] + self.env.action_space.shape[0] + 1
        if self._use_next_obs_in_context:
            context_dims += self.env.observation_space.shape[0]
        if not self._disable_local_encoder:
            self._local_context = torch.zeros((1,1,context_dims)) # Ensure local context for first step is just 0s
        else:
            self._local_context = None

    def step_episode(self):
        """Take a single time-step in the current episode.
        Returns:
            bool: True iff the episode is done, either due to the environment
            indicating termination of due to reaching `max_episode_length`.

        """
        if self._eps_length < self._max_episode_length:
            # Get action and step
            a, agent_info = self.agent.get_action(self._prev_obs, self._local_context)
            if self._deterministic:
                a = agent_info['mean']
            es = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._env_steps.append(es)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            self._eps_length += 1
            # Update context
            if not self._disable_local_encoder:
                self._local_context = torch.cat([torch.as_tensor(self._prev_obs[None, None, ...], device=global_device()).float(), #obs (IE obs used to choose action)
                                    torch.as_tensor(es.action[None, None, ...], device=global_device()).float(), 
                                    torch.as_tensor(es.reward, device=global_device()).float().reshape(1,1,-1)], dim=2) 
                if self._use_next_obs_in_context:
                    self._local_context = torch.cat([self._local_context,
                                        torch.as_tensor(es.observation[None, None, ...], device=global_device()).float()] #next_obs (IE obs resulting from action chosen)
                                        , dim=2)
                self.agent.infer_local_posterior(self._local_context)
            else:
                self._local_context = None

            if not es.last:
                self._prev_obs = es.observation
                return False
        self._lengths.append(self._eps_length)
        self._last_observations.append(self._prev_obs)
        return True

    def rollout(self):
        """Sample a single episode of the agent in the environment.

        Returns:
            EpisodeBatch: The collected episode.

        """
        self.agent.sample_global_from_belief()
        self.start_episode()
        while not self.step_episode():
            pass
        return self.collect_episode()
