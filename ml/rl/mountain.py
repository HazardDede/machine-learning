"""Train a reinforcement learning agent to drive a mountain car up the hill and be king of the hill."""

import gym
import matplotlib.pyplot as plt
import numpy as np

from ml import Context
from ml.repository.artifact import Repository
from ml.rl import QLearning
from ml.utils import LogMixin


class _Discretizer:
    """Discretizes the observation space. Puts the more or less infinite observation space (position and velocity)
    into discrete buckets."""
    def __init__(self, env, buckets):
        self.env = env
        self.buckets = buckets
        self.discrete_os_size = [buckets] * len(env.observation_space.high)
        self.window_size = (
            (env.observation_space.high - env.observation_space.low)
            / self.discrete_os_size
        )

    def discretize(self, state):
        discrete_state = (state - self.env.observation_space.low) / self.window_size
        return tuple(discrete_state.astype(np.int))


class Learner(LogMixin):
    """Solving the mountain car game using reinforcement learning."""

    def __init__(self, artifact_repo):
        """
        Args:
            artifact_repo (Artifacts): The artifact repository to use.
            dataset_repo (Datasets): The dataset repository to use.
        """
        self._artifact_repo: Repository = artifact_repo

    @classmethod
    def _from_context(cls):
        """Create a new instance by injecting the artifact from the application context."""
        return cls(
            artifact_repo=Context.artifacts
        )

    def simulate(self, agent_path: str):
        """
        Instructs to simulate the mountain car game by using a previously trained agent.

        Args:
            agent_path (str): Path to previously trained agent.
        """

        env = gym.make("MountainCar-v0")

        qlearn = QLearning.from_file(agent_path)
        discretizer = _Discretizer(env, qlearn.q_table.shape[0])
        discrete_state = discretizer.discretize(env.reset())  # current state

        done = False
        while not done:
            action = qlearn.action(discrete_state)  # Pick next action from q-table
            new_state, reward, done, _ = env.step(action)  # Tell our decision to the environment

            discrete_state = discretizer.discretize(new_state)
            env.render()

    def train(self, buckets=20, episodes=10000, learning_rate=0.1, gamma=0.9):
        """
        Train a reinforcement learning agent to drive up a hill.

        Args:
            buckets (int): Number of buckets for discretizing the observation space. Default is 20.
            episodes (int): Number of episodes to simulate. Default is 10,000.
            learning_rate (float): How much we trust the new q value (0.0 - 1.0). Default is 0.1.
            gamma (float): Measure on how import we find future actions (0.0 - 1.0).
             Default is 0.90.
        """
        env = gym.make("MountainCar-v0")

        discretizer = _Discretizer(env, buckets)

        summary_every = (episodes // 10)
        if summary_every <= 0:
            summary_every = 1

        qlearn = QLearning(
            observation_space=(buckets, ) * len(env.observation_space.high),
            no_of_actions=env.action_space.n,
            learning_rate=learning_rate,
            gamma=gamma,
            low_q=-1,
            high_q=1
        )

        for episode in range(1, episodes + 1):
            summary = episode % summary_every == 0 and episode

            discrete_state = discretizer.discretize(env.reset())  # current state
            done = False
            goal = False
            qlearn.episode_start()
            while not done:
                action = qlearn.action(discrete_state)

                # Tell our decision to the environment
                new_state, reward, done, _ = env.step(action)
                new_discrete_state = discretizer.discretize(new_state)

                qlearn.update(discrete_state, action, new_discrete_state, reward)
                goal = new_state[0] >= env.goal_position
                discrete_state = new_discrete_state

            qlearn.episode_finished(goal)
            if summary:
                self._logger.info("Episode %s: %s", episode, qlearn.summary(summary_every))
                path = self._artifact_repo.artifact_path(f'{episode}_qtable.pkl')
                qlearn.save(path)

        env.close()

        self._plot(qlearn, summary_every)

    def _plot(self, qlearn, summary_every):
        qlearn.plot_metrics(summary_every)
        path = self._artifact_repo.artifact_path('metrics.png')
        plt.savefig(path)

        plt.clf()
        qlearn.plot_goals(summary_every)
        path = self._artifact_repo.artifact_path('goals.png')
        plt.savefig(path)
