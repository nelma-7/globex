"""
PEARL launcher. Uses the garage implementation of PEARL

"""
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

from algos.pearl import PEARL, PEARLWorker 
from algos.context_conditioned_policy import ContextConditionedPolicy

def pearl_experiment(ctxt, env_class, is_gym, variant):
    """Train PEARL, using the garage package.
    Args (contained within variant):
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        num_epochs (int): Number of training epochs.
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int): Number of tasks to use for testing.
        latent_size (int): Size of latent context vector.
        encoder_hidden_size (int): Output dimension of dense layer of the
            context encoder.
        net_size (int): Output dimension of a dense layer of Q-function and
            value function.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        max_episode_length (int): Maximum episode length.
        reward_scale (int): Reward scale.
        use_gpu (bool): Whether or not to use GPU for training.
    """
    set_seed(variant['seed'])
    ctxt = None
    trainer = Trainer(ctxt)

    variant["encoder_hidden_sizes"] = (variant["encoder_hidden_size"], variant["encoder_hidden_size"], variant["encoder_hidden_size"])

    # Create multi-task environment and sample tasks
    base_env = env_class() # The base env is to ensure we are sampling tasks with the same seed!
    base_env.seed(variant['seed'])
    base_env.action_space.seed(variant['seed'])

    if is_gym:
        env_sampler = SetTaskSampler(
            env_class,
            env=base_env,
            wrapper=lambda env, _: normalize(
                GymEnv(env, max_episode_length=variant["max_episode_length"])))
        test_env_sampler = SetTaskSampler(
        env_class,
        env=base_env,
        wrapper=lambda env, _: normalize(
            GymEnv(env, max_episode_length=variant["max_episode_length"])))
    else: # an example of an env where is_gym=False is PointRobot
        env_sampler = SetTaskSampler(
            env_class,
            env=base_env,
            wrapper=lambda env, _: normalize(env))
        test_env_sampler = SetTaskSampler(
        env_class,
        env=base_env,
        wrapper=lambda env, _: normalize(env))

    env = env_sampler.sample(variant["pearl_params"]["num_train_tasks"]) # env[0]() will produce an environment
    initial_env = env[0]()

    # instantiate networks
    hidden_sizes = [variant["net_size"],variant["net_size"],variant["net_size"]]
    augmented_env = PEARL.augment_env_spec(initial_env, variant["pearl_params"]["latent_dim"])
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=hidden_sizes)

    vf_env = PEARL.get_env_spec(initial_env, variant["pearl_params"]["latent_dim"], 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=hidden_sizes)

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=hidden_sizes)
    
    sampler = LocalSampler(agents=None,
                           envs=initial_env, 
                           max_episode_length=initial_env.spec.max_episode_length,
                           n_workers=1,
                           worker_class=PEARLWorker)

    algo = PEARL(
        env=env,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        sampler=sampler,
        test_env_sampler=test_env_sampler,
        encoder_hidden_sizes=variant["encoder_hidden_sizes"],
        **variant["pearl_params"] 
    )

    set_gpu_mode(variant["util_params"]["use_gpu"], gpu_id=variant["util_params"]["gpu_id"])
    if variant["util_params"]["use_gpu"]:
        algo.to()

    trainer.setup(algo=algo, env=initial_env)

    trainer.train(n_epochs=variant["num_epochs"], batch_size=variant["pearl_params"]["batch_size"])
