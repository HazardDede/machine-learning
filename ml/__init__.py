import os
import uuid
from typing import (
    Optional
)

from ml.repository.artifact import Repository as ArtifactRepository
from ml.repository.datasets import Repository as DatasetRepository


class ContextMeta(type):
    """Meta class of Context to fake class properties."""

    _ARTIFACT_FOLDER = 'artifacts'
    _DATASET_FOLDER = 'datasets'

    _artifacts: Optional[ArtifactRepository] = None
    _base_path: Optional[str] = None
    _datasets: Optional[DatasetRepository] = None
    _project: Optional[str] = None

    @property
    def artifacts(cls) -> ArtifactRepository:
        """Return the configured artifact repository."""
        if cls._artifacts is None:
            return ArtifactRepository(
                project=cls.project,
                base_path=os.path.join(cls.base_path, cls._ARTIFACT_FOLDER)
            )
        return cls._artifacts

    @property
    def base_path(cls) -> str:
        """Return the base path of the project."""
        if cls._base_path is None:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        return cls._base_path

    @base_path.setter
    def base_path(cls, value : str) -> None:
        """Set the base path of the project."""
        cls._base_path = value

    @property
    def datasets(cls) -> DatasetRepository:
        """Return the dataset repository."""
        if cls._datasets is None:
            return DatasetRepository(
                base_path=os.path.join(cls.base_path, cls._DATASET_FOLDER)
            )
        return cls._datasets

    @property
    def project(cls) -> str:
        """Return the name of the current project."""
        if cls._project is None:
            return str(uuid.uuid4())
        return cls._project

    @project.setter
    def project(cls, value: str) -> None:
        """Set the name of the current project."""
        cls._project = value


class Context(metaclass=ContextMeta):
    """Application context. Stores some (meant to be) immutable properties."""
