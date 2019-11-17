"""Provides classes for artifact retrieval and storing.."""

import os
from typing import (
    Any
)


class Repository:
    """Provides methods to store artifacts like models, cv results, ...."""
    _DEFAULT_ARTIFACT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../artifacts'))

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

    def dump_to_file(self, artifact: str, content: Any):
        """Dump the passed content to the artifact file."""
        with open(os.path.join(self.path, artifact), 'w') as fhandle:
            fhandle.write(content)

    # TODO: Provide method to save model
    # TODO: Automatically infer type and use appropiate method
