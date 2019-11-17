"""Provides repositories for various tasks such a retrieving and storing datasets."""
from typing import Any

import attr
import pandas as pd


@attr.s
class Sets:
    # Dataset constants
    BBC_NEWS = 'text/bbc-news'
    IRIS = 'iris'


@attr.s
class Dataset:
    data = attr.ib(type=pd.DataFrame, default=None)
    description = attr.ib(type=str, default=None)

    def is_text(self) -> bool:
        """Return True if this dataset is a text dataset."""
        return False

    def is_classification(self) -> bool:
        """Return True if this dataset is a classification dataset."""
        return False

    @classmethod
    def enforce_text_dataset(cls, candidate: Any) -> None:
        """Enforces the given candidate to be a text dataset. If not a TypeError
        is raised."""
        if not hasattr(candidate, 'is_text'):
            raise TypeError("Dataset does not provide 'is_text' method")
        if not candidate.is_text:
            raise TypeError("Dataset is not a text dataset.")


@attr.s
class ClassificationDataset(Dataset):
    TARGET_COLUMN_DEFAULT = 'target'

    feature_columns = attr.ib(type=list, default=None)
    target_column = attr.ib(type=str, default=TARGET_COLUMN_DEFAULT)

    def is_classification(self) -> bool:
        return True


@attr.s
class TextDataset(Dataset):
    TEXT_COLUMN_DEFAULT = 'text'
    TARGET_COLUMN_DEFAULT = 'target'

    text_column = attr.ib(type=str, default=TEXT_COLUMN_DEFAULT)
    target_column = attr.ib(type=str, default=TARGET_COLUMN_DEFAULT)

    def is_text(self) -> bool:
        return True
