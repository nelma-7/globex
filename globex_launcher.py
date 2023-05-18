"""
GLOBEX launcher
"""
from garage.envs import GymEnv, normalize
from garage.experiment import Snapshotter
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

from globex import GLOBEX #new algo
from value_function import ContinuousMLPValueFunction

def globex_experiment(env_class, variant):
    """Train GLOBEX.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        SEE ALGO FOR REST OF ARGUMENTS
        use_gpu (bool): Whether or not to use GPU for training.
    """
    set_seed(seed)
    trainer = Trainer(ctxt)

    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size, encoder_hidden_size)

    # Create multi-task environment and sample tasks
    base_env = env_class() # The base env is to ensure we are sampling tasks with the same seed!
    base_env.seed(seed)
    base_env.action_space.seed(seed)

    env_sampler = SetTaskSampler(
        env_class,
        env=base_env,
        wrapper=lambda env, _: normalize(
            GymEnv(env, max_episode_length=max_episode_length)))

    env = env_sampler.sample(num_train_tasks) # env[0]() will produce an environment
    initial_env = env[0]()

    test_env_sampler = SetTaskSampler(
        env_class,
        env=base_env,
        wrapper=lambda env, _: normalize(
            GymEnv(env, max_episode_length=max_episode_length)))

    # Instantiate networks
    # Augmented env dims - in = action space, out = obs space + latent dim
    augmented_env = GLOBEX.augment_env_spec(initial_env, latent_size, disable_local_encoder=disable_local_encoder, disable_global_encoder=disable_global_encoder,
                                                sample_global_embedding=sample_global_embedding, sample_local_embedding=sample_local_embedding)
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size],
                                hidden_nonlinearity = qf_nonlinearity)

    vf_env = GLOBEX.get_env_spec(initial_env, latent_size, 'vf', use_next_obs_in_context=use_next_obs_in_context, 
                                            disable_local_encoder=disable_local_encoder, disable_global_encoder=disable_global_encoder,
                                                sample_global_embedding=sample_global_embedding, sample_local_embedding=sample_local_embedding)

    vf = ContinuousMLPValueFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size],
                                hidden_nonlinearity = vf_nonlinearity)

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size],
        hidden_nonlinearity = policy_nonlinearity)

    worker_args = dict(deterministic=False, disable_local_encoder=disable_local_encoder,
                        use_next_obs_in_context=use_next_obs_in_context)
    
    sampler = LocalSampler(agents=None,  #either localsampler or raysampler. RaySampler may not work well on GPUs
                            envs=initial_env,
                            max_episode_length=env[0]().spec.max_episode_length,
                            n_workers=1,
                            worker_args=worker_args,
                            worker_class=al.CustomWorker)   

    algo = GLOBEX(
        env=env,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        sampler=sampler,
        global_recurrent_encoder=global_recurrent_encoder,
        local_recurrent_encoder=local_recurrent_encoder,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        latent_dim=latent_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        decoder_hidden_sizes=decoder_hidden_sizes,
        batch_size=batch_size,
        test_env_sampler=test_env_sampler,
        num_steps_prior=num_steps_prior,
        num_steps_posterior_in_enc=num_steps_posterior_in_enc,
        num_steps_posterior=num_steps_posterior,
        stochastic_training=stochastic_training,
        context_batch_size=context_batch_size,
        context_tbptt_size=context_tbptt_size,
        num_initial_steps=num_initial_steps,
        num_iter_per_epoch=num_iter_per_epoch,
        num_tasks_sample=num_tasks_sample,
        meta_batch_size=meta_batch_size,
        global_kl_lambda=global_kl_lambda,
        local_kl_lambda=local_kl_lambda,
        reward_loss_coefficient=reward_loss_coefficient,
        state_loss_coefficient=state_loss_coefficient,
        transition_reconstruction_coefficient=transition_reconstruction_coefficient,
        mi_loss_coefficient=mi_loss_coefficient,
        discount=discount,
        reward_scale=reward_scale,
        policy_lr=policy_lr,
        qf_lr=qf_lr,
        vf_lr=vf_lr,
        context_lr=context_lr,
        global_nonlinearity=global_nonlinearity,
        local_nonlinearity=local_nonlinearity,
        decoder_nonlinearity=decoder_nonlinearity,
        n_exploration_eps=n_exploration_eps,
        n_test_episodes=n_test_episodes,
        epochs_per_eval=epochs_per_eval,
        save_grads = save_grads,
        policy_max_grad_norm = policy_max_grad_norm, 
        global_max_grad_norm = global_max_grad_norm,
        local_max_grad_norm = local_max_grad_norm,
        decoder_max_grad_norm = decoder_max_grad_norm,
        mi_max_grad_norm = mi_max_grad_norm,
        decode_state=decode_state,
        decode_reward=decode_reward,
        policy_loss_through_global=policy_loss_through_global,
        policy_loss_through_local=policy_loss_through_local,
        sample_global_embedding=sample_global_embedding,
        sample_local_embedding=sample_local_embedding,
        use_next_obs_in_context=use_next_obs_in_context,
        local_kl_normal_prior=local_kl_normal_prior,
        disable_local_encoder=disable_local_encoder,
        disable_global_encoder=disable_global_encoder
    )

    set_gpu_mode(use_gpu, gpu_id=gpu_id)
    if use_gpu:
        algo.to()

    trainer.setup(algo=algo, env=initial_env)

    trainer.train(n_epochs=num_epochs, batch_size=batch_size)

pearl_half_cheetah_dir()