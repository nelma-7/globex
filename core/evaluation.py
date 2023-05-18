"""Evaluator which tests Meta-RL algorithms on test environments."""

from dowel import logger, tabular

from garage import EpisodeBatch, log_multitask_performance
from garage.experiment.deterministic import get_seed
from garage.sampler import DefaultWorker, LocalSampler, WorkerFactory
from garage.experiment import MetaEvaluator


class customMetaEvaluator(MetaEvaluator):
    """Evaluates Meta-RL algorithms on test environments.
    Args:
        test_task_sampler (TaskSampler): Sampler for test
            tasks. To demonstrate the effectiveness of a meta-learning method,
            these should be different from the training tasks.
        is_sampler (Bool): Confirms whether the test_task_sampler is actually a TaskSampler
            If False, we take it as the RESULT of the test task sampler
        deterministic_eval (Bool): Determines whether we set environment seed when instantiating envs
            If True, we will change env seed each meta-test iteration, but in a consistent way (adding eval_itr to seed)
            If False, meta-test env instantiation will be completely random
        n_test_tasks (int or None): Number of test tasks to sample each time
            evaluation is performed. Note that tasks are sampled "without
            replacement". If None, is set to `test_task_sampler.n_tasks`.
        n_exploration_eps (int): Number of episodes to gather from the
            exploration policy before requesting the meta algorithm to produce
            an adapted policy.
        n_test_episodes (int): Number of episodes to use for each adapted
            policy. The adapted policy should forget previous episodes when
            `.reset()` is called.
        prefix (str): Prefix to use when logging. Defaults to MetaTest. For
            example, this results in logging the key 'MetaTest/SuccessRate'.
            If not set to `MetaTest`, it should probably be set to `MetaTrain`.
        test_task_names (list[str]): List of task names to test. Should be in
            an order consistent with the `task_id` env_info, if that is
            present.
        worker_class (type): Type of worker the Sampler should use.
        worker_args (dict or None): Additional arguments that should be
            passed to the worker.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 *args,
                 test_task_sampler,
                 deterministic_eval=True, 
                 is_sampler=True,
                 max_episode_length=None,
                 n_exploration_eps=0,
                 n_test_tasks=None,
                 n_test_episodes=1,
                 prefix='MetaTest',
                 test_task_names=None,
                 worker_class=DefaultWorker,
                 worker_args=None):
        self._test_task_sampler = test_task_sampler
        self._worker_class = worker_class
        self._deterministic_eval = deterministic_eval
        if worker_args is None:
            self._worker_args = {}
        else:
            self._worker_args = worker_args
        if n_test_tasks is None:
            n_test_tasks = test_task_sampler.n_tasks
        self._n_test_tasks = n_test_tasks
        self._n_test_episodes = n_test_episodes
        self._n_exploration_eps = n_exploration_eps
        self._eval_itr = 0
        self._prefix = prefix
        self._test_task_names = test_task_names
        self._test_sampler = None
        self._train_sampler = None
        self._max_episode_length = max_episode_length
        self._is_sampler = is_sampler
        

    def eval_train(self, algo, train_envs, prefix='MetaTrain'):
        """Evaluate the Meta-RL algorithm on all train tasks. CURRENTLY UNUSED as it's much more efficient to use sampled training runs 
        Args:
            algo (MetaRLAlgorithm): The algorithm to evaluate.
        """
        if test_episodes_per_task is None:
            test_episodes_per_task = self._n_test_episodes
        
        adapted_episodes = []
        logger.log('Evaluating training tasks...')
        if self._train_sampler is None:
            single_env = train_envs[0]()
            self._max_episode_length = single_env.spec.max_episode_length
            self._train_sampler = LocalSampler.from_worker_factory(
                WorkerFactory(seed=get_seed(),
                              max_episode_length=self._max_episode_length,
                              n_workers=1,
                              worker_class=self._worker_class,
                              worker_args=self._worker_args),
                agents=algo.get_exploration_policy(),
                envs=single_env)
        
        # Gather exploration eps in the same way we would at test time
        for env_up in train_envs:
            policy = algo.get_exploration_policy()
            if self._n_exploration_eps == 0:
                eps = None
            else:
                eps = EpisodeBatch.concatenate(*[
                    self._train_sampler.obtain_samples(self._eval_itr, 1, policy,
                                                    env_up)
                    for _ in range(self._n_exploration_eps)
                ])
            # UPDATED FOR PEARL AND PEARL-LIKE ALGORITHMS - obtain 1 episode at a time 
            for _ in test_episodes_per_task:
                adapted_policy = algo.adapt_policy(policy, eps)
                adapted_eps = self._train_sampler.obtain_samples( 
                    self._eval_itr,
                    self._max_episode_length,
                    adapted_policy)
                adapted_episodes.append(adapted_eps)
                eps = EpisodeBatch.concatenate(eps, adapted_eps)
        logger.log('Finished train task eval')

        if self._test_task_names is not None:
            name_map = dict(enumerate(self._test_task_names))
        else:
            name_map = None

        with tabular.prefix(prefix + '/' if prefix else ''):
            log_multitask_performance(
                self._eval_itr,
                EpisodeBatch.concatenate(*adapted_episodes),
                getattr(algo, 'discount', 1.0),
                name_map=name_map)
    
    def evaluate(self, algo, test_episodes_per_task=None, only_eval_last_ep=True):
        """Evaluate the Meta-RL algorithm on the test tasks.
        Args:
            algo (MetaRLAlgorithm): The algorithm to evaluate.
            test_episodes_per_task (int or None): Number of episodes per task.
        """
        if test_episodes_per_task is None:
            test_episodes_per_task = self._n_test_episodes
        adapted_episodes = []
        logger.log('Sampling for adapation and meta-testing...')
        if self._is_sampler:
            env_updates = self._test_task_sampler.sample(self._n_test_tasks)
        else:
            env_updates = self._test_task_sampler
        if self._test_sampler is None:
            env = env_updates[0]()
            if self._deterministic_eval and env.unwrapped.__class__.__name__ in ['MetaWorldSetTaskEnv']:
                env.unwrapped._current_env.seed(get_seed() + self._eval_itr)
                env.unwrapped._current_env.action_space.seed(get_seed() + self._eval_itr)
            elif self._deterministic_eval and env.unwrapped.__class__.__name__ not in ['PointEnv', 'SparsePointEnv']:
                env.seed(get_seed() + self._eval_itr)
                env.action_space.seed(get_seed() + self._eval_itr)
            self._max_episode_length = env.spec.max_episode_length
            self._test_sampler = LocalSampler.from_worker_factory(
                WorkerFactory(seed=get_seed(),
                              max_episode_length=self._max_episode_length,
                              n_workers=1,
                              worker_class=self._worker_class,
                              worker_args=self._worker_args),
                agents=algo.get_exploration_policy(),
                envs=env)
        for env_up in env_updates:
            policy = algo.get_exploration_policy()
            if self._n_exploration_eps == 0:
                eps = None
            else:
                eps = EpisodeBatch.concatenate(*[
                    self._test_sampler.obtain_samples(self._eval_itr, 1, policy,
                                                    env_up)
                    for _ in range(self._n_exploration_eps)
                ])
            # UPDATED FOR PEARL AND PEARL-LIKE ALGORITHMS - obtain 1 episode at a time 
            for i in range(test_episodes_per_task):
                adapted_policy = algo.adapt_policy(policy, eps)
                adapted_eps = self._test_sampler.obtain_samples( 
                    self._eval_itr,
                    self._max_episode_length,
                    adapted_policy)
                # Only save the LAST adapted ep unless we specifically want to save all
                if not only_eval_last_ep: 
                    adapted_episodes.append(adapted_eps)
                elif only_eval_last_ep and i == (test_episodes_per_task-1):
                    adapted_episodes.append(adapted_eps)
                eps = EpisodeBatch.concatenate(eps, adapted_eps)
        logger.log('Finished meta-testing...')

        if self._test_task_names is not None:
            name_map = dict(enumerate(self._test_task_names))
        else:
            name_map = None

        with tabular.prefix(self._prefix + '/' if self._prefix else ''):
            log_multitask_performance(
                self._eval_itr,
                EpisodeBatch.concatenate(*adapted_episodes),
                getattr(algo, 'discount', 1.0),
                name_map=name_map)
        self._eval_itr += 1