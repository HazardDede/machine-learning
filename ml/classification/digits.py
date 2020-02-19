"""Mnist digits classification tasks."""
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from ml import Context
from ml.repository.artifact import Repository
from ml.utils import LogMixin


class Classifier(LogMixin):
    """Trains a neural network on the mnist dataset."""

    _ARTIFACT_MODEL = 'model.tf'
    _TENSORBOARD_LOGS = 'tb_logs'

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
        """Shows an example digit from the dataset."""
        x_train, _, _, _ = self._load_data()
        plt.imshow(x_train[0], cmap=plt.cm.binary)
        plt.show()

    def train(self, epochs=5):
        """Train a neural network on the mnist dataset.

        Args:
            epochs (int): The number of epochs. Default is 5.
        """
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

        dtnow = datetime.now().strftime("%Y-%m-%dT%H:%M")
        tb_logs = self._artifact_repo.artifact_path(self._TENSORBOARD_LOGS)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='{}/{}'.format(tb_logs, dtnow))
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(x_train, y_train, epochs=int(epochs), validation_data=(x_test, y_test), callbacks=[tensorboard])

        # val_loss, val_acc = model.evaluate(x_test, y_test)

        # self._logger.info("Evaluation on test dataset: Loss: %s, Accuracy: %s", val_loss, val_acc)

        path = self._artifact_repo.artifact_path(self._ARTIFACT_MODEL)
        model.save(path)

    def predict(self):
        """Uses the previously trained model to classify the test dataset."""
        path = self._artifact_repo.artifact_path(self._ARTIFACT_MODEL)
        model = tf.keras.models.load_model(path)

        _, _, x_test, y_test = self._load_data()
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        preds = model.predict(x_test)
        self._show_cf_matrix(np.array([np.argmax(probas) for probas in preds]), y_test)

    def _show_cf_matrix(self, preds, y):
        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        cf_matrix = tf.math.confusion_matrix(y, preds).numpy()
        cf_matrix_norm = np.around(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
        cf_matrix_df = pd.DataFrame(cf_matrix_norm, index=classes, columns=classes)

        plt.figure(figsize=(8, 8))
        sns.heatmap(cf_matrix_df, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('Fact')
        plt.xlabel('Prediction')
        plt.show()
