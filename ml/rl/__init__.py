"""Reinforcement learning helper classes."""

import pickle
import random
from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodesAggregate:
    """Represents an aggregation of multiple episodes."""
    count: int
    avg: float
    min: float
    max: float
    goals: int


class EpisodesSummary:
    """Represents a summary class of all the episodes and their associated rewards."""
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

        return EpisodesAggregate(
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

    def plot(self, bucket_size):
        for i in range(len(self.rewards)):
            min_idx = i - bucket_size + 1
            min_idx = 0 if min_idx < 0 else min_idx
            max_idx = i + 1
            aggr = self._aggregate_helper(
                self.rewards[min_idx:max_idx],
                self.goals[min_idx:max_idx]
            )
            yield i, aggr


class LearningBase:
    """Base class for all reinforcement agent."""
    def __init__(self):
        self.ep_summary = EpisodesSummary()
        self.ep_cum_reward = 0

    @classmethod
    def from_file(cls, file_path):
        """Load the agent from disk."""
        with open(file_path, "rb") as fp:
            return pickle.load(fp)

    def action(self, state):
        """Return the next best action for the given state."""
        raise NotImplementedError()

    def update(self, state, action, new_state, reward, goal):
        """Update the agent given the current state and action, the reward and the new state."""
        self.ep_cum_reward += reward

    def episode_start(self):
        """Starts a new episode. Always call this before processing a new episode."""
        self.ep_cum_reward = 0

    def episode_finished(self, goal_achived):
        """Ends an episode. You pass if the agent achived the goal or not."""
        self.ep_summary.add_reward(self.ep_cum_reward, goal_achived)

    def summary(self, last_n=50):
        """Return the summary / aggregation of key metrics for the last n episodes."""
        return self.ep_summary.aggregate(last_n)

    def plot_metrics(self, bucket_size):
        """Plot some metrics using matplotlib. X axis is the episode and Y axis are the metrics like
        average, max and min reward."""
        import matplotlib.pyplot as plt

        buckets = list(self.ep_summary.plot(bucket_size))
        ep = [ep for ep, _ in buckets]
        avg_reward = [aggr.avg for _, aggr in buckets]
        min_reward = [aggr.min for _, aggr in buckets]
        max_reward = [aggr.max for _, aggr in buckets]

        # First plot: avg, max, min
        plt.plot(ep, avg_reward, label="avg")
        plt.plot(ep, min_reward, label="min")
        plt.plot(ep, max_reward, label="max")
        # plt.plot(ep, goals, label="goals")
        plt.legend(loc=4)

    def plot_goals(self, bucket_size):
        """Plot the number of achived goals vs. the number of episodes in the bucket. X axis is the episode and Y axis
        is the number of achived goals."""
        import matplotlib.pyplot as plt

        buckets = list(self.ep_summary.plot(bucket_size))
        ep = [ep for ep, _ in buckets]
        ep_cnt = [aggr.count for _, aggr in buckets]
        goals = [aggr.goals for _, aggr in buckets]

        # Second plot: Goal achievement
        plt.plot(ep, ep_cnt, label="episodes")
        plt.plot(ep, goals, label="goals")
        plt.legend(loc=4)

    def save(self, file_path):
        with open(file_path, "wb") as fp:
            pickle.dump(self, fp)


class QLearning(LearningBase):
    def __init__(self, observation_space, no_of_actions, learning_rate=0.1, gamma=0.9, low_q=0, high_q=0):
        """
        Reinforcement learning agent that uses a simple q-table to learn the best policy.

        Args:
            observation_space (tuple): Tuple with the dimensions of the decision space.
             Example: Assume a 3x3 grid represent your state then you would pass (3, 3).
            no_of_actions (int): The number of actions the agent can choose from. The agent will return just
             a number. You have to convert it to an action your environment can understand.
            learning_rate (float): How much we trust new experiments. 0.0 - 1.0. Default is 0.1.
            gamma (float): How import the future reward is. 0 -> of no importance -> just the immediate reward vs.
             1.0 -> very important. Default is 0.9.
            low_q (int): Lower bound for initialized q-values.
            high_q (int): Upper bound for initialized q-values.
        """
        super().__init__()
        self.q_table = np.random.uniform(
            low=low_q,
            high=high_q,
            size=observation_space + (no_of_actions, )
        )
        self.learning_rate = learning_rate
        self.gamma = gamma

    def action(self, state):
        """Return the next best action for the given state."""
        return np.argmax(self.q_table[state])

    def update(self, state, action, new_state, reward, goal):
        """Update the q-table based on the current state and action, the reward and the new state."""
        self.ep_cum_reward += reward
        current_q = self.q_table[state + (action,)]
        max_future_q = np.max(self.q_table[new_state])

        new_q = current_q + self.learning_rate * (
            reward + self.gamma * max_future_q - current_q
        )
        self.q_table[state + (action,)] = new_q


class FixedAction(LearningBase):
    """Agent that will always return a fixed action. Good for baseline."""
    def __init__(self, fixed_action):
        """
        Args:
            fixed_action (int): The action that is fixed.
        """
        super().__init__()
        self.fixed_action = fixed_action

    def action(self, state):
        return self.fixed_action


class RandomAction(LearningBase):
    """Agent that will always return a random action. Good for baseline."""
    def __init__(self, no_of_actions):
        """
        Args:
            no_of_actions (int): The number of actions to randomly choose from.
        """
        super().__init__()
        self.no_of_actions = no_of_actions

    def action(self, state):
        return random.randint(0, self.no_of_actions - 1)
