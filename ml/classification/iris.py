import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ml import Context
from ml.repository import Sets, ClassificationDataset
from ml.utils import LogMixin


class Classifier(LogMixin):
    """A generic text classifier. Uses a grid search to find the best model
    for the dataset."""

    _ARTIFACT_MODEL = 'model.pkl'
    _ARTIFACT_CV_RESUTS = 'cv_results_all.csv'
    _ARTIFACT_CLASSIFICATION_REPORT = 'report.txt'
    _ARTIFACT_BEST_CLASSIFIER = 'best_classifier.txt'

    def __init__(self, artifact_repo, dataset_repo):
        """
        Args:
            artifact_repo (Artifacts): The artifact repository to use.
            dataset_repo (Datasets): The dataset repository to use.
        """
        self._artifact_repo = artifact_repo
        self._dataset_repo = dataset_repo

    @classmethod
    def from_context(cls):
        """Create a new instance by injecting the artifact and dataset repository
        from the application context."""
        return cls(
            artifact_repo=Context.artifacts,
            dataset_repo=Context.datasets
        )

    def train(self):
        """Train the iris classifier."""
        dataset = self._dataset_repo.fetch(Sets.IRIS)
        assert isinstance(dataset, ClassificationDataset)

        X, y = dataset.data[dataset.feature_columns], dataset.data[dataset.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(
            y_test,
            y_pred
        )
        self._logger.info("\n%s", report)

        joblib.dump(clf, self._artifact_repo.artifact_path(self._ARTIFACT_MODEL))

    def predict(self):
        dataset = self._dataset_repo.fetch(Sets.IRIS)
        assert isinstance(dataset, ClassificationDataset)
        X, y = dataset.data[dataset.feature_columns], dataset.data[dataset.target_column]

        # Retrieve the model
        model_path = self._artifact_repo.artifact_path(self._ARTIFACT_MODEL)
        model = joblib.load(model_path)

        # Use the model
        y_pred = model.predict(X)
        report = classification_report(
            y,
            y_pred
        )
        self._logger.info("\n%s", report)
