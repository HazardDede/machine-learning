import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from ml import Context
from ml.repository.artifact import Repository
from ml.utils import LogMixin


class Classifier(LogMixin):
    """A generic text classifier. Uses a grid search to find the best model
    for the iris dataset."""

    _ARTIFACT_MODEL = 'model.tf'

    def __init__(self, artifact_repo, dataset_repo):
        """
        Args:
            artifact_repo (Artifacts): The artifact repository to use.
            dataset_repo (Datasets): The dataset repository to use.
        """
        self._artifact_repo: Repository = artifact_repo
        self._dataset_repo = dataset_repo

    @classmethod
    def _from_context(cls):
        """Create a new instance by injecting the artifact and dataset repository
        from the application context."""
        return cls(
            artifact_repo=Context.artifacts,
            dataset_repo=Context.datasets
        )

    def _load_data(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return x_train, y_train, x_test, y_test

    def show_digit(self):
        """Print show an example digit."""
        x_train, _, _, _ = self._load_data()
        plt.imshow(x_train[0], cmap=plt.cm.binary)
        plt.show()

    def train(self):
        """Train the digits classifier."""
        x_train, y_train, x_test, y_test = self._load_data()
        x_train = tf.keras.utils.normalize(x_train, axis=1)  # Scale between 0-1
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        model = tf.keras.models.Sequential()
        # 28 x 28 (digits dimensions) -> flat 784
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        # neurons -> number of classification
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(x_train, y_train, epochs=3)

        val_loss, val_acc = model.evaluate(x_test, y_test)

        self._logger.info("Evaluation on test dataset: Loss: %s, Accuracy: %s", val_loss, val_acc)

        path = self._artifact_repo.artifact_path(self._ARTIFACT_MODEL)
        model.save(path)

    def predict(self):
        path = self._artifact_repo.artifact_path(self._ARTIFACT_MODEL)
        model = tf.keras.models.load_model(path)

        _, _, x_test, y_test = self._load_data()
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        preds = model.predict(x_test)

        self._logger.info("Predicted: %s, Actual: %s", np.argmax(preds[0]), y_test[0])
