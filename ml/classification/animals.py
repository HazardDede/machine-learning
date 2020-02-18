import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ml import Context
from ml.repository import Sets, ClassificationDataset
from ml.repository.artifact import Repository
from ml.utils import LogMixin


class Classifier(LogMixin):
    """Convoluted neural network to classify cats and dogs."""

    _ARTIFACT_MODEL = 'model.tf'
    _TENSORBOARD_LOGS = 'tb_logs'

    _IMAGE_SIZE_WIDTH = 100
    _IMAGE_SIZE_HEIGHT = 100

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

    def _prepare_X_y(self, dataset):
        X = np.array(
            list(dataset.data[dataset.feature_columns[0]])
        ).reshape(-1, self._IMAGE_SIZE_WIDTH, self._IMAGE_SIZE_HEIGHT, 1) / 255.0
        y = np.array(
            list(dataset.data[dataset.target_column])
        )
        return X, y

    def train(self, epochs=5, examples=None):
        """Train the network to classify cats and dogs.

        Args:
            epochs (int): Number of epochs to process.
            examples (Optional[int]): The number of examples to draw instead of processing all images.
        """
        dataset = self._dataset_repo.fetch(Sets.CATS_VS_DOGS)
        assert isinstance(dataset, ClassificationDataset)

        dtnow = datetime.now().strftime("%Y-%m-%dT%H:%M")
        tb_logs = self._artifact_repo.artifact_path(self._TENSORBOARD_LOGS)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='{}/{}'.format(tb_logs, dtnow))

        X, y = self._prepare_X_y(dataset)
        if examples:
            examples = int(examples)
            X = X[:examples]
            y = y[:examples]

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=X.shape[1:]))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64))
        model.add(tf.keras.layers.Activation("relu"))

        model.add(tf.keras.layers.Dense(1))
        model.add(tf.keras.layers.Activation("sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(X, y, batch_size=32, epochs=epochs, validation_split=0.2, callbacks=[tensorboard])

        model_path = self._artifact_repo.artifact_path(self._ARTIFACT_MODEL)
        model.save(model_path)

    def predict(self):
        """Use the network to predict cats and dogs on new photos."""
        dataset = self._dataset_repo.fetch(Sets.CATS_VS_DOGS)
        assert isinstance(dataset, ClassificationDataset)

        path = self._artifact_repo.artifact_path(self._ARTIFACT_MODEL)
        model = tf.keras.models.load_model(path)

        X, y = self._prepare_X_y(dataset)

        preds = model.predict(X)
        preds = [1 if pred[0] >= 0.5 else 0 for pred in preds]

        self._show_cf_matrix(preds, y)

    def _show_cf_matrix(self, preds, y):
        classes = ["cat", "dog"]

        cf_matrix = tf.math.confusion_matrix(y, preds).numpy()
        cf_matrix_norm = np.around(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
        cf_matrix_df = pd.DataFrame(cf_matrix_norm, index=classes, columns=classes)

        plt.figure(figsize=(2, 2))
        sns.heatmap(cf_matrix_df, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('Fact')
        plt.xlabel('Prediction')
        plt.show()
