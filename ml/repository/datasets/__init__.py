"""Provides convenience stuff to fetch datasets for training / experimenting."""

import os
from typing import Any, List, Optional

from ml.repository import Sets
from ml.repository.datasets.catalog import Catalog
from ml.repository.datasets.loader import BBCNews, Iris
from ml.utils import LogMixin


class UnknownDatasetError(Exception):
    """Is raised when an unknown dataset is requested."""
    DEFAULT_MESSAGE = "The dataset '{dataset_name}' is unknown."

    def __init__(self, dataset_name: str):
        super().__init__(self.DEFAULT_MESSAGE.format(dataset_name=str(dataset_name)))


class Repository(LogMixin):
    """Provides methods to retrieve datasets."""

    # Default base path to persisted datasets
    _DEFAULT_BASE_PATH = os.path.join(os.path.dirname(__file__), '../datasets')
    _CATALOG_NAME = 'catalog.yaml'

    _DEFAULT_TEXT_LABEL = 'text'
    _DEFAULT_TEXT_TARGET_LABEL = 'target'

    def __init__(self, base_path: Optional[str] = None):
        self._base_path = base_path or self._DEFAULT_BASE_PATH

        self._load_map = {
            Sets.BBC_NEWS: BBCNews,
            Sets.IRIS: Iris
        }

        catalog_path = os.path.join(self._base_path, self._CATALOG_NAME)
        if os.path.isfile(catalog_path):
            self._catalog = Catalog.from_yaml(catalog_path)
        else:
            self._catalog = Catalog.empty(self._base_path)

    def list(self) -> List[str]:
        """Return a list of available datasets."""
        return list(self._load_map.keys()) + self._catalog.list()

    def fetch(self, dataset_name: str) -> Any:
        """Fetch the specified dataset.
        
        Args:
            dataset_name: The name of the dataset to fetch.

        Returns:
            The dataset (probably a pandas dataframe) and
            some metadata information. The exact return depends
            on the dataset to fetch.
        """
        loader_clazz = self._load_map.get(dataset_name)
        
        if loader_clazz:
            real_dataset_name = dataset_name.split('/')[-1]
            dataset_path = os.path.abspath(
                os.path.join(self._base_path, real_dataset_name)
            )

            return loader_clazz(dataset_path).retrieve()

        if dataset_name in self._catalog.list():
            return self._catalog.load(dataset_name)

        raise UnknownDatasetError(dataset_name)
