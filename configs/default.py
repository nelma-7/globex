import torch.nn.functional as F

default_config = dict(
    algo='GLOBEX', # algorithm
    env_name='cheetah-vel', # environment name
    num_epochs=200,
    net_size=300, # number of units per FC layer in policy networks
    encoder_hidden_size=200, # number of units per FC layer in encoder networks
    decoder_hidden_sizes=(200,200),
    max_episode_length=200,
    global_recurrent_encoder=False,
    local_recurrent_encoder=False,
    policy_nonlinearity = F.relu,
    vf_nonlinearity = F.relu,
    qf_nonlinearity = F.relu,
    algo_params=dict(
        num_train_tasks=100, 
        num_test_tasks=30,
        latent_dim=5, # dimension of the latent context vector
        discount=0.99,
        reward_scale=5.0,
        use_next_obs_in_context=False,
        # Training params
        num_iter_per_epoch=2000,
        num_initial_steps=2000, 
        num_steps_prior=400,
        num_steps_posterior=0,
        num_extra_rl_steps_posterior=600, 
        batch_size=256,
        num_tasks_sample=5,
        meta_batch_size=16,
        # SAC Policy params
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        policy_mean_reg_coeff=1E-3,
        policy_std_reg_coeff=1E-3,
        policy_pre_activation_coeff=0.,
        soft_target_tau=0.005,
        replay_buffer_size=1000000,
        policy_max_grad_norm = None, 
        # Global-Local Embedding params
        context_lr=3E-4,
        context_batch_size=100,
        context_tbptt_size=None,
        context_buffer_size=100000,
        global_kl_lambda=.01,
        local_kl_lambda=.01,
        local_kl_normal_prior=True, # True=N(0,1) prior for local KL, False=prev transaction prior
        reward_loss_coefficient=1,
        state_loss_coefficient=1,
        transition_reconstruction_coefficient=1,
        mi_loss_coefficient=0.1,
        global_max_grad_norm = None,
        local_max_grad_norm = None,
        decoder_max_grad_norm = None,
        mi_max_grad_norm = None,
        encoder_min_std = 1e-10,
        sample_global_embedding = True, # Whether to sample global_z (True), or pass the parameters of the normal dist to the policy (False)
        sample_local_embedding = True, # Whether to sample local_z (True), or pass the parameters of the normal dist to the policy (False)
        # Other
        global_nonlinearity = F.relu,
        local_nonlinearity = F.relu,
        decoder_nonlinearity = F.relu,
        epochs_per_eval=1, # Num epochs per eval run
        n_exploration_eps=2, # Num exploration eps before adaptation via global embedding
        n_test_episodes=1,
        save_grads=False, # saves a list of avg grad norms each iteration; for debugging purposes
        # Ablations
        decode_state=False, # Whether to decode state in global encoder training
        decode_reward=False, # Whether to decode reward in global encoder training
        policy_loss_through_global=True, 
        policy_loss_through_local=False,
        disable_local_encoder=False,
        disable_global_encoder=False,
    ),
    pearl_params=dict( #default params for PEARL - follows cheetah-vel params
        num_train_tasks=100,
        num_test_tasks=30,
        latent_dim=5,
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3E-4,
        policy_mean_reg_coeff=1E-3,
        policy_std_reg_coeff=1E-3,
        policy_pre_activation_coeff=0.,
        soft_target_tau=0.005,
        kl_lambda=.1,
        use_information_bottleneck=True,
        use_next_obs_in_context=False,
        meta_batch_size=16,
        num_iter_per_epoch=2000,
        num_initial_steps=2000,
        num_tasks_sample=5,
        num_steps_prior=400,
        num_steps_posterior=0,
        num_extra_rl_steps_posterior=600,
        batch_size=256,
        context_batch_size=100,
        context_mini_batch_size=100,
        discount=0.99,
        replay_buffer_size=1000000,
        reward_scale=5.0,
        update_post_train=1,
        epochs_per_eval=1, # Num of epochs per eval run
        n_exploration_eps=2, 
        n_test_episodes=1,                     
    ),
    util_params=dict(
        use_gpu=True,
        gpu_id=0,
    )
)