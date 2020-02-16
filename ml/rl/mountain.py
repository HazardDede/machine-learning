import random
from dataclasses import dataclass

import gym
import numpy as np
import matplotlib.pyplot as plt

from ml import Context
from ml.repository.artifact import Repository
from ml.utils import LogMixin


class _Discretizer:
    """Discretizes the observation space."""
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

    def random_q_table(self, q_low=-1, q_high=0):
        return np.random.uniform(
            low=q_low,
            high=q_high,
            size=(self.discrete_os_size + [self.env.action_space.n])
        )


@dataclass
class _Aggregate:
    count: int
    avg: float
    min: float
    max: float
    goals: int


class _Summary:
    """Result set for episodes"""
    def __init__(self):
        self.rewards = []
        self.goals = []

    def add_reward(self, reward: float, goal_achieved: bool) -> None:
        self.rewards.append(reward)
        self.goals.append(1 if goal_achieved else 0)

    def _aggregate_helper(self, rewards, goals):
        goal_cnt = sum(goals)
        episode_cnt = len(rewards)
        average_reward = sum(rewards) / len(rewards)
        min_reward = min(rewards)
        max_reward = max(rewards)

        return _Aggregate(
            count=episode_cnt, avg=average_reward, min=min_reward, max=max_reward, goals=goal_cnt
        )

    def aggregate(self, last_n=None):
        last_n = last_n or len(self.rewards)  # Aggregate all rewards

        return self._aggregate_helper(self.rewards[-last_n:], self.goals[-last_n:])

    def bucketize(self, n: int):
        def chunks(lst):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        reward_buckets, goal_buckets = list(chunks(self.rewards)), list(chunks(self.goals))
        episode = 0
        for i in range(len(reward_buckets)):
            aggr = self._aggregate_helper(reward_buckets[i], goal_buckets[i])
            episode += aggr.count
            yield episode, aggr

    def plot(self, last_n):
        for i in range(len(self.rewards)):
            min_idx = i - last_n + 1
            min_idx = 0 if min_idx < 0 else min_idx
            max_idx = i + 1
            aggr = self._aggregate_helper(
                self.rewards[min_idx:max_idx],
                self.goals[min_idx:max_idx]
            )
            yield i, aggr


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

    def solve(self, qtable_path: str):
        """
        Instructs to solve the mountain car game by using a previously trained q-table.

        Args:
            qtable_path (str): Path to the q-table to use.
        """

        env = gym.make("MountainCar-v0")

        q_table = np.load(qtable_path)
        discretizer = _Discretizer(env, q_table.shape[0])
        discrete_state = discretizer.discretize(env.reset())  # current state

        done = False
        while not done:
            action = np.argmax(q_table[discrete_state])  # Pick next action from q-table
            new_state, reward, done, _ = env.step(action)  # Tell our decision to the environment

            discrete_state = discretizer.discretize(new_state)
            env.render()

    def train(
            self, buckets=10, learning_rate=0.1, episodes=1000, gamma=0.9, summary_every=None,
            epsilon=0.5, exploration_episodes=None
    ):
        """
        Based on the game mountain car we will simulate reinforcement learning using a
        simple q-table.

        Args:
            buckets (int): Number of buckets for discretizing the observation space. Default is 10.
            learning_rate (float): How much we trust the new q value (0.0 - 1.0). Default is 0.1.
            episodes (int): Number of episodes to simulate. Default is 1000.
            gamma (float): Measure on how import we find future actions (0.0 - 1.0).
             Default is 0.90.
            epsilon (float): The chance to explore (0.0 - 1.0). 0.0 means no exploration; 1.0
             means exploration only. See exploration_episodes for further configuration.
             Default is 0.5.
            exploration_episodes (int): The number of episodes exploration is allowed. After the
             allowed value is reached only exploitation without exploration will take place.
            summary_every (int): How often we want to summarize.
        """
        env = gym.make("MountainCar-v0")

        discretizer = _Discretizer(env, buckets)

        summary_every = int(summary_every or (episodes // 10))
        exploration_episodes = int(exploration_episodes or (episodes // 2))

        # buckets x buckets x 3 (actions)
        q_table = discretizer.random_q_table()

        metrics = _Summary()

        for episode in range(1, episodes + 1):
            summary = episode % summary_every == 0 and episode

            discrete_state = discretizer.discretize(env.reset())  # current state
            done = False
            cum_reward = 0
            goal = False
            while not done:
                if episode <= exploration_episodes and random.uniform(0, 1) < epsilon:
                    # Exploration -> Random decision
                    action = random.randint(0, env.action_space.n - 1)  # Random action
                else:
                    # Exploitation -> Use q table
                    action = np.argmax(q_table[discrete_state])  # Pick next action from q-table

                # Tell our decision to the environment
                new_state, reward, done, _ = env.step(action)
                cum_reward += reward
                new_discrete_state = discretizer.discretize(new_state)

                if not done:
                    max_future_q = np.max(q_table[new_discrete_state])
                    current_q = q_table[discrete_state + (action, )]
                    new_q = (
                        (1 - learning_rate) * current_q + learning_rate
                        * (reward + gamma * max_future_q)
                    )
                    q_table[discrete_state + (action,)] = new_q
                elif new_state[0] >= env.goal_position:
                    goal = True
                    q_table[discrete_state + (action,)] = 0

                discrete_state = new_discrete_state

            metrics.add_reward(cum_reward, goal)

            if summary:
                aggr = metrics.aggregate(summary_every)
                self._logger.info(
                    f"Episode: {episode}, count: {aggr.count}, goals: {aggr.goals}, "
                    f"avg: {aggr.avg}, min: {aggr.min}, max: {aggr.max}"
                )
                path = self._artifact_repo.artifact_path(f'{episode}_qtable')
                np.save(path, q_table)

        env.close()

        self._plot(metrics, summary_every)

    def _plot(self, metrics, summary_every):
        buckets = list(metrics.plot(summary_every))
        ep = [ep for ep, _ in buckets]
        avg_reward = [aggr.avg for _, aggr in buckets]
        min_reward = [aggr.min for _, aggr in buckets]
        max_reward = [aggr.max for _, aggr in buckets]
        ep_cnt = [aggr.count for _, aggr in buckets]
        goals = [aggr.goals for _, aggr in buckets]

        # First plot: avg, max, min
        plt.plot(ep, avg_reward, label="avg")
        plt.plot(ep, min_reward, label="min")
        plt.plot(ep, max_reward, label="max")
        # plt.plot(ep, goals, label="goals")
        plt.legend(loc=4)
        path = self._artifact_repo.artifact_path('metrics.png')
        plt.savefig(path)

        # Second plot: Goal achievement
        plt.clf()
        plt.plot(ep, ep_cnt, label="episodes")
        plt.plot(ep, goals, label="goals")
        plt.legend(loc=4)
        path = self._artifact_repo.artifact_path('goals.png')
        plt.savefig(path)
