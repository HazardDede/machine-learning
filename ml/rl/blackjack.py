"""Train a reinforcement learning agent to play blackjack."""
import gym

from ml import Context
from ml.repository.artifact import Repository
from ml.rl import QLearning, FixedAction, RandomAction
from ml.utils import LogMixin


class Learner(LogMixin):
    """Playing the blackjack game using reinforcement learning."""

    def __init__(self, artifact_repo):
        """
        Args:
            artifact_repo (Artifacts): The artifact repository to use.
        """
        self._artifact_repo: Repository = artifact_repo

    @classmethod
    def _from_context(cls):
        """Create a new instance by injecting the artifact from the application context."""
        return cls(
            artifact_repo=Context.artifacts
        )

    def _convert_state(self, raw_state):
        """Converts the state from the env into something the learner could understand."""
        sum_of_hand, sum_revealed, usable_ace = raw_state
        return sum_of_hand, sum_revealed, 1 if usable_ace else 0

    def _generic_train(self, episodes, qlearn):
        """Trains a model using the given q leaner.

        Args:
            episodes (int): The number of episodes to run.
            qlearn (LearnerBase): The learner to use.
        """
        env = gym.make('Blackjack-v0')

        summary_every = episodes // 10
        if summary_every <= 0:
            summary_every = 1

        for episode in range(1, episodes + 1):

            done = False
            state = self._convert_state(env.reset())
            qlearn.episode_start()
            goal = False

            while not done:
                action = qlearn.action(state)
                new_state, reward, done, _ = env.step(action)
                new_state = self._convert_state(new_state)
                goal = reward > 0
                qlearn.update(state, action, new_state, reward, goal)
                state = new_state

            qlearn.episode_finished(goal)
            if episode % summary_every == 0:
                self._logger.info("Episode %s: %s", episode, qlearn.summary(summary_every))
                path = self._artifact_repo.artifact_path(f'{episode}_qtable.pkl')
                qlearn.save(path)

    def train(self, episodes=50000, learning_rate=0.1, gamma=0.95):
        """
        Train a q-learning model to play blackjack.

        Args:
            episodes (int): The number of episodes to run. Default is 50,000.
            learning_rate (float): How much we trust the new q value. 0.0 - 1.0. Default is 0.1.
            gamma (float): Indicates how much future rewards should be taken into account. 0.0 - 1.0. Default is 0.95.
        """
        qlearn = QLearning(
            observation_space=(32, 11, 2),
            no_of_actions=2,
            learning_rate=learning_rate,
            gamma=gamma,
            low_q=-1,
            high_q=1
        )
        self._generic_train(episodes, qlearn)

    def baseline(self, episodes=50000, mode='fixed'):
        """Train a baseline model to play blackjack. Use it for comparison with your trained model.

        Args:
            episodes (int): The number of episodes to run. Default is 50,000.
            mode (str): Either fixed or random. If 'fixed' is selected the agent will never draw cards and always stick.
             Mode 'random' will either randomly stick or draw a card.
        """
        if mode not in ['fixed', 'random']:
            raise RuntimeError("Argument mode needs to be one of ['fixed', 'random']")

        qlearn = FixedAction(fixed_action=0) if mode == 'fixed' else RandomAction(2)
        self._generic_train(episodes, qlearn)
