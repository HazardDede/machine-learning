"""Main entry point."""

import logging

from ml import Context
from ml.text.generic import Classifier as GenericTextClassifier
from ml.rl.mountain import Learner as Mountain
from ml.rl.blackjack import Learner as Blackjack
from ml.classification.animals import Classifier as AnimalsClassifier
from ml.classification.iris import Classifier as IrisClassifier
from ml.classification.digits import Classifier as DigitsClassifier

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


class ClassificationModels:
    """Provided classification models."""
    animals = AnimalsClassifier._from_context
    iris = IrisClassifier._from_context
    digits = DigitsClassifier._from_context


class RLModels:
    """Provided reinforcement learning models."""
    mountain = Mountain._from_context
    blackjack = Blackjack._from_context


class TextModels:
    """Provided text models."""
    generic = GenericTextClassifier._from_context


class Models:
    """Model categories."""
    text = TextModels
    rl = RLModels
    classification = ClassificationModels


class Runner:
    """Machine Learning Sandbox."""

    _ARTIFACT_FOLDER = 'artifacts'
    _DATASET_FOLDER = 'datasets'

    def __init__(self, project=None, base_path=None):
        """
        Args:
            project (str): The name of the project to separate artifacts.
            base_path (str): The base path where to store/find artifacts and datasets.
        """
        if project:
            Context.project = str(project)
        if base_path:
            Context.base_path = str(base_path)

        self.artifacts = Context.artifacts
        self.datasets = Context.datasets
        self.models = Models()


if __name__ == '__main__':
    import fire
    fire.Fire(Runner)
