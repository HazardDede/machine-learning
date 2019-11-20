import os
from typing import Any, Dict

import pandas as pd
import schema as sc
from ruamel import yaml  # type: ignore

from ml.repository import TextDataset, Dataset


class UnknownCatalogItemError(Exception):
    def __init__(self, item_type: str):
        super().__init__(f"Unknown catalog item type '{item_type}'")


class UnknownLocationError(Exception):
    def __init__(self, location_type: str):
        super().__init__(f"Unknown location type '{location_type}'")


class CatalogItemConfig:
    CONF_NAME = 'name'
    CONF_TYPE = 'type'
    CONF_DESCRIPTION = 'desc'
    CONF_LOCATION = 'location'

    SCHEMA = {
        CONF_NAME: sc.Use(str),
        CONF_TYPE: sc.Use(str),
        sc.Optional(CONF_DESCRIPTION, default=''): sc.Use(str),
        CONF_LOCATION: {
            str: object
        },
        sc.Optional(str): object
    }

    @classmethod
    def item_type_map(cls):
        return {
            'text': TextCatalogItemConfig
        }

    @classmethod
    def from_dict(cls, dct: Dict[str, Any], base_path: str) -> Dataset:
        validated = sc.Schema(cls.SCHEMA, ignore_extra_keys=True).validate(dct)
        item_name = validated[cls.CONF_NAME]
        item_type = validated[cls.CONF_TYPE]
        parser = cls.item_type_map().get(item_type)
        if not parser:
            raise UnknownCatalogItemError(item_type)

        return parser.load_from_dict(dct, base_path=os.path.join(base_path, item_name))


class TextCatalogItemConfig:
    CONF_TEXT_TARGET = 'target_column'
    CONF_TEXT_TEXT = 'text_column'

    SCHEMA = {
        **CatalogItemConfig.SCHEMA,
        sc.Optional(CONF_TEXT_TARGET, default='target'): sc.Use(str),
        sc.Optional(CONF_TEXT_TEXT, default='text'): sc.Use(str)
    }

    @classmethod
    def load_from_dict(cls, dct: Dict[str, Any], base_path: str) -> TextDataset:
        validated = sc.Schema(cls.SCHEMA).validate(dct)
        data = LocationConfig.from_dict(
            validated[CatalogItemConfig.CONF_LOCATION],
            base_path=base_path
        )
        target = validated[cls.CONF_TEXT_TARGET]
        feature = validated[cls.CONF_TEXT_TEXT]
        data[target] = data[target].astype(str)
        data[feature] = data[feature].astype(str)
        return TextDataset(
            data=data,
            description=validated[CatalogItemConfig.CONF_DESCRIPTION],
            text_column=feature,
            target_column=target
        )


class LocationConfig:
    CONF_LOCATION_TYPE = 'type'

    SCHEMA = {
        CONF_LOCATION_TYPE: sc.Use(str)
    }

    @classmethod
    def location_type_map(cls):
        return {
            'csv': CsvLocationConfig
        }

    @classmethod
    def from_dict(cls, dct: Dict[str, Any], base_path: str) -> Any:
        validated = sc.Schema(cls.SCHEMA, ignore_extra_keys=True).validate(dct)
        location_type = validated[cls.CONF_LOCATION_TYPE]
        parser = cls.location_type_map().get(location_type)
        if not parser:
            raise UnknownLocationError(location_type)

        return parser.load_from_dict(dct, base_path)


class CsvLocationConfig:
    CONF_URI = 'uri'
    CONF_SEP = 'separator'

    SCHEMA = {
        **LocationConfig.SCHEMA,
        CONF_URI: sc.Use(str),
        sc.Optional(CONF_SEP, default=','): sc.Use(str)
    }

    @classmethod
    def load_from_dict(cls, dct: Dict[str, Any], base_path: str) -> pd.DataFrame:
        validated = sc.Schema(cls.SCHEMA).validate(dct)
        file_path = validated[cls.CONF_URI]
        separator = validated[cls.CONF_SEP]
        return pd.read_csv(os.path.join(base_path, file_path), sep=separator)


class Catalog:
    CONF_CATALOG = 'catalog'

    SCHEMA = {
        CONF_CATALOG: {
            str: sc.Schema(CatalogItemConfig.SCHEMA, ignore_extra_keys=True)
        }
    }

    def __init__(self, base_path: str, config: Dict[str, Any]):
        self._base_path = str(base_path)
        self._config = config

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as fp:
            cfg = yaml.safe_load(fp)

        validated = sc.Schema(cls.SCHEMA).validate(cfg)
        return cls(
            base_path=os.path.dirname(yaml_path),
            config=validated
        )

    @classmethod
    def empty(cls, base_path: str):
        return cls(base_path=base_path, config={cls.CONF_CATALOG: {}})

    def list(self):
        items = self._config[self.CONF_CATALOG]
        return list(items.keys())

    def load(self, item_name):
        items = self._config[self.CONF_CATALOG]
        item_cfg = items.get(item_name)
        if not item_cfg:
            raise UnknownCatalogItemError(item_name)
        return CatalogItemConfig.from_dict(item_cfg, base_path=self._base_path)


if __name__ == '__main__':
    import fire
    fire.Fire(Catalog)
