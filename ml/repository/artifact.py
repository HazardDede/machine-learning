"""Provides classes for artifact retrieval and storing.."""

import os
from typing import (
    Any,
    Optional
)

import joblib
from sklearn.base import BaseEstimator

from ml.utils import LogMixin


class ArtifactError(Exception):
    """Base class of all artifact errors."""


class LoadModelError(ArtifactError):
    """Is raised when a model couldn't be loaded for some reason."""


class UnsupportedModelError(ArtifactError):
    """Is raised when the operation is unsupported for the specified model."""


class Repository(LogMixin):
    """Provides methods to store artifacts like models, cv results, ...."""
    _DEFAULT_ARTIFACT_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../artifacts')
    )

    def __init__(self, project: str, base_path: str = None):
        self._project = str(project)
        self._base_path = str(base_path or self._DEFAULT_ARTIFACT_PATH)

    @property
    def path(self) -> str:
        """Return the path to the projects artifact root."""
        res = os.path.join(self._base_path, self._project)
        os.makedirs(res, exist_ok=True)
        return res

    @property
    def root(self) -> str:
        os.makedirs(self._base_path, exist_ok=True)
        return self._base_path

    def artifact_path(self, artifact_name: str) -> str:
        """Return the path to the artifact."""
        return os.path.join(self.path, artifact_name)

    def dump_to_file(self, artifact: str, content: Any) -> None:
        """Dump the passed content to the artifact file."""
        with open(os.path.join(self.path, artifact), 'w') as fhandle:
            fhandle.write(content)

    def save_model(self, model: Any, mid: Optional[str] = None) -> None:
        mid = str(mid or self._project)
        if isinstance(model, BaseEstimator):
            mid += '.skl'
            path = os.path.join(self.path, mid)
            self._logger.info("Saving sklearn model to %s", path)
            joblib.dump(model, path)
            return

        raise UnsupportedModelError(
            "Operation save_model is unsupported for the passed model of type '{}'".format(
                type(model)
            )
        )

    def load_model(self, mid: Optional[str] = None) -> Any:
        mid = str(mid or self._project)
        path = os.path.join(self.path, mid)
        skl_path = path + '.skl'
        if os.path.isfile(skl_path):
            # Scikit-learn joblib dump
            self._logger.info("Retrieving sklearn model from %s", skl_path)
            return joblib.load(skl_path)

        raise LoadModelError("Cannot load persisted model. Tried: {}".format(
            [skl_path]
        ))
