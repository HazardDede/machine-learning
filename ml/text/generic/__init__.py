import itertools

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from ml import Context
from ml.repository import Dataset, Sets
from ml.text.generic.features import feature_space, feature_steps
from ml.text.generic.models import model_space, model_steps
from ml.utils import LogMixin


class Classifier(LogMixin):
    """A generic text classifier. Uses a grid search to find the best model
    for the dataset."""

    _ARTIFACT_MODEL = 'model.pkl'
    _ARTIFACT_CV_RESUTS = 'cv_results_all.csv'
    _ARTIFACT_CLASSIFICATION_REPORT = 'report.txt'
    _ARTIFACT_BEST_CLASSIFIER = 'best_classifier.txt'
    _ARTIFACT_PREDICTIONS = 'predictions.csv'
    _ARTIFACT_CONF_MATRIX = 'confusion_matrix.pkl'

    def __init__(self, artifact_repo, dataset_repo):
        """
        Args:
            artifact_repo (Artifacts): The artifact repository to use.
            dataset_repo (Datasets): The dataset repository to use.
        """
        self._artifact_repo = artifact_repo
        self._dataset_repo = dataset_repo

    @classmethod
    def _from_context(cls):
        """Create a new instance by injecting the artifact and dataset repository
        from the application context."""
        return cls(
            artifact_repo=Context.artifacts,
            dataset_repo=Context.datasets
        )

    def _store_artifacts(self, clf, report):
        best_clf_report = "Best Score: {}\n{}".format(clf.best_score_, clf.best_params_)
        self._logger.info("\n%s", best_clf_report)
        self._artifact_repo.dump_to_file(self._ARTIFACT_BEST_CLASSIFIER, best_clf_report)

        self._logger.info("Classification matrix on test data:\n%s", report)
        self._artifact_repo.dump_to_file(self._ARTIFACT_CLASSIFICATION_REPORT, report)

        df_cv = pd.DataFrame(clf.cv_results_)
        df_cv.to_csv(self._artifact_repo.artifact_path(self._ARTIFACT_CV_RESUTS), index=False)

        self._artifact_repo.save_model(clf)

    def train(
        self, language=None, dataset=None, cvector=False, tfidf=False, sgd=False,
        mnb=False, rforest=False, svc_linear=False, svc_nonlinear=False, gboost=False,
        lregression=False, oversample=False
    ):
        """
        Trains the model against the specified dataset.

        If the text language is any other than 'english' you have to specify it
        explicitly.

        You can enable/disable feature engineering (tf-idf transformation and
        tf-idf vectoring) and specific models. By default both feature engineering
        will be tried, but only the sgd model is enabled.

        Args:
            language (str): Language of the dataset.
            dataset (str): The name of the dataset.
            cvector (bool): Enable/disable the count vectorizer(default is False).
            tfidf (bool): Enable/disable the tf-idf vectoring (default is False).
            sgd (bool): Enable/disable the sgd model (default is False).
            mnb (bool): Enable/disable the mnb model (default is False).
            rforest (bool): Enable/disable the random forest model (default is False).
            svc_linear (bool): Enable/disable the support vector with linear kernel
             (default is False).
            svc_nonlinear (bool): Enable/disable the support vector with non-linear kernel
             (default is False).
            gboost (bool): Enable/disable the gboost model (default is False).
            lregression (bool): Enable/disable the logistic regression model (default is False).
            oversample (bool): Enable/disable oversampling for imbalanced classes.
        """
        language = str(language or 'english')
        dataset_name = str(dataset or Sets.BBC_NEWS)

        # Load data
        dataset = self._dataset_repo.fetch(dataset_name)
        Dataset.enforce_text_dataset(dataset)
        df, TEXT_LABEL, CLASS_LABEL = dataset.data, dataset.text_column, dataset.target_column
        X, y = df[TEXT_LABEL], df[CLASS_LABEL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42, test_size=0.2
        )

        if oversample:
            # Do oversampling for imbalanced classes
            X_train, y_train = RandomOverSampler(random_state=42).fit_sample(
                np.array(X_train).reshape(-1, 1), y_train
            )
            X_train = [str(x) for x in X_train]

        # Train the model using Pipeline and GridSearch
        ppl = Pipeline(
            feature_steps() + model_steps()
        )

        features = feature_space(
            language, enable_count_vector=cvector, enable_tfidf=tfidf
        )
        models = model_space(
            enable_sgd=sgd, enable_mnb=mnb, enable_rforest=rforest,
            enable_svc_linear=svc_linear, enable_svc_nonlinear=svc_nonlinear,
            enable_gboost=gboost, enable_lregression=lregression
        )

        parameter_space = (
            [{**pre, **clf}
             for pre, clf in itertools.product(features, models)]
        )
        clf = GridSearchCV(ppl, parameter_space, cv=5, iid=False, n_jobs=-1, verbose=1)
        clf.fit(X_train, y_train)

        # Predict on hold-back test data
        pred = clf.predict(X_test)
        report = metrics.classification_report(
            y_test,
            pred
        )

        # Store all artifacts
        self._store_artifacts(clf, report)

    def predict(self, dataset=None):
        """Uses a previously trained model to make predictions against a
        new dataset.

        Args:
            language (str): Language of the dataset.
            dataset (str): The name of the dataset.
        """
        # Retrieve the dataset
        dataset_name = str(dataset or Sets.BBC_NEWS)
        dataset = self._dataset_repo.fetch(dataset_name)
        Dataset.enforce_text_dataset(dataset)
        df, TEXT_LABEL, CLASS_LABEL = dataset.data, dataset.text_column, dataset.target_column
        X, y = df[TEXT_LABEL], df[CLASS_LABEL]

        # Retrieve the model
        model = self._artifact_repo.load_model()

        # Use the model
        y_pred = model.predict(X)
        report = metrics.classification_report(
            y,
            y_pred
        )
        self._logger.info("\n%s", report)

        conf_mat = metrics.confusion_matrix(y, y_pred)
        joblib.dump(conf_mat, self._artifact_repo.artifact_path(self._ARTIFACT_CONF_MATRIX))

        df_res = pd.concat([df, pd.DataFrame(y_pred, columns=("prediction",))], axis=1)

        try:
            y_proba = model.predict_proba(X)
            df_probas = pd.DataFrame(
                y_proba,
                columns=[clazz + '__proba' for clazz in model.classes_]
            )
            df_res = pd.concat([df_res, df_probas], axis=1)
        except Exception:  # pylint: disable=broad-except
            import traceback
            self._logger.warning(
                "Couldn't calculate prediction probabilities:\n%s", traceback.format_exc()
            )

        df_res.to_csv(self._artifact_repo.artifact_path(self._ARTIFACT_PREDICTIONS), index=False)
