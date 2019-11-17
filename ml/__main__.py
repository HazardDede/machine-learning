"""Main entry point."""

import logging

from ml import Context
from ml.text.generic import Classifier as GenericTextClassifier
from ml.classification.iris import Classifier as IrisClassifier

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


class ClassificationModels:
    """Provided classification models."""
    iris = IrisClassifier.from_context


class TextModels:
    """Provided text models."""
    generic = GenericTextClassifier.from_context


class Models:
    """Model categories."""
    text = TextModels
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
        self.models = Models


if __name__ == '__main__':
    import fire
    fire.Fire(Runner)
