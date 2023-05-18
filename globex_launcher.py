"""
GLOBEX launcher
"""
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

from algos.globex import GLOBEX, CustomWorker #new algo
from core.value_function import ContinuousMLPValueFunction

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
    set_seed(variant['seed'])
    ctxt = None
    trainer = Trainer(ctxt)

    variant["encoder_hidden_sizes"] = (variant["encoder_hidden_size"], variant["encoder_hidden_size"], variant["encoder_hidden_size"])

    # Create multi-task environment and sample tasks
    base_env = env_class() # The base env is to ensure we are sampling tasks with the same seed!
    base_env.seed(variant['seed'])
    base_env.action_space.seed(variant['seed'])

    env_sampler = SetTaskSampler(
        env_class,
        env=base_env,
        wrapper=lambda env, _: normalize(
            GymEnv(env, max_episode_length=variant["max_episode_length"])))

    env = env_sampler.sample(variant["num_train_tasks"]) # env[0]() will produce an environment
    initial_env = env[0]()

    test_env_sampler = SetTaskSampler(
        env_class,
        env=base_env,
        wrapper=lambda env, _: normalize(
            GymEnv(env, max_episode_length=variant["max_episode_length"])))

    # Instantiate networks
    hidden_sizes = [variant["net_size"],variant["net_size"],variant["net_size"]]
    # Augmented env dims - in = action space, out = obs space + latent dim
    augmented_env = GLOBEX.augment_env_spec(initial_env, variant["algo_params"]["latent_size"], 
                                            disable_local_encoder=variant["algo_params"]["disable_local_encoder"], 
                                            disable_global_encoder=variant["algo_params"]["disable_global_encoder"],
                                            sample_global_embedding=variant["algo_params"]["sample_global_embedding"], 
                                            sample_local_embedding=variant["algo_params"]["sample_local_embedding"])
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=hidden_sizes,
                                hidden_nonlinearity = variant["qf_nonlinearity"])

    vf_env = GLOBEX.get_env_spec(initial_env, variant["algo_params"]["latent_size"], 'vf', 
                                 use_next_obs_in_context=variant["algo_params"]["use_next_obs_in_context"], 
                                disable_local_encoder=variant["algo_params"]["disable_local_encoder"], 
                                disable_global_encoder=variant["algo_params"]["disable_global_encoder"],
                                sample_global_embedding=variant["algo_params"]["sample_global_embedding"], 
                                sample_local_embedding=variant["algo_params"]["sample_local_embedding"])

    vf = ContinuousMLPValueFunction(env_spec=vf_env,
                                hidden_sizes=hidden_sizes,
                                hidden_nonlinearity = variant["vf_nonlinearity"])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=hidden_sizes,
        hidden_nonlinearity = variant["policy_nonlinearity"])

    worker_args = dict(deterministic=False, disable_local_encoder=variant["algo_params"]["disable_local_encoder"],
                        use_next_obs_in_context=variant["algo_params"]["use_next_obs_in_context"])
    
    sampler = LocalSampler(agents=None,  
                            envs=initial_env,
                            max_episode_length=initial_env.spec.max_episode_length,
                            n_workers=1,
                            worker_args=worker_args,
                            worker_class=CustomWorker)   

    algo = GLOBEX(
        env=env,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        sampler=sampler,
        test_env_sampler=test_env_sampler,
        global_recurrent_encoder=variant["global_recurrent_encoder"],
        local_recurrent_encoder=variant["local_recurrent_encoder"],
        encoder_hidden_sizes=variant["encoder_hidden_sizes"],
        decoder_hidden_sizes=variant["decoder_hidden_sizes"], 
        **variant["algo_params"] # TODO try kwargs???
    )

    set_gpu_mode(variant["util_params"]["use_gpu"], gpu_id=variant["util_params"]["gpu_id"])
    if variant["util_params"]["use_gpu"]:
        algo.to()

    trainer.setup(algo=algo, env=initial_env)

    trainer.train(n_epochs=variant["num_epochs"], batch_size=variant["algo_params"]["batch_size"])
