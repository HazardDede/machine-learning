import cv2
import os
import pandas as pd
import pickle
import random
import zipfile

from ml.repository import TextDataset, ClassificationDataset
from ml.utils import LogMixin
from ml.utils.io import download_url


class BBCNews(LogMixin):
    """Internal class to handle the download, unpack and merging of the bbc
    news dataset."""

    ZIP_NAME = 'bbc.zip'
    EXTRACTED_FOLDER_NAME = 'bbc'
    CSV_FILE_NAME = 'bbc-news.csv'

    FEATURE_TEXT_LABEL = 'text'
    TARGET_LABEL = 'target'

    CATS = ['business', 'entertainment', 'politics', 'sport', 'tech']

    def __init__(self, base_path: str):
        self.base_path = str(base_path)

    @staticmethod
    def _process_files(path, category):
        texts = []
        folder = os.path.join(path, category)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r', encoding="utf-8", errors="replace") as fhandle:
                text = fhandle.read()
            texts.append((text, category))

        return texts

    def retrieve(self) -> TextDataset:
        """Retrieve the bbc news dataset.
        If necessary it will download, unpack and merge the dataset.
        Returns a tuple of DataFrame containing the text and the target,
        the name of the text feature and the name of the target (y)."""

        # Download file
        path = self.base_path
        url = 'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip'
        zip_file = os.path.join(path, self.ZIP_NAME)
        if not os.path.exists(zip_file):
            os.makedirs(path, exist_ok=True)
            self._logger.info("Downloading '%s'", url)
            download_url(url, zip_file)
        else:
            self._logger.info("'%s' already exists. Skipping download", zip_file)

        # Unzip file
        extract_target = os.path.join(path, self.EXTRACTED_FOLDER_NAME)
        if not os.path.exists(extract_target):
            self._logger.info("Unzipping '%s' to '%s'", zip_file, path)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(path)
        else:
            self._logger.info("'%s' already exists. Skipping unzip", extract_target)

        data = None
        # Process files -> Merge them into one file and provide a dataframe
        process_target = os.path.join(path, self.CSV_FILE_NAME)
        if os.path.exists(process_target):
            self._logger.info("'%s' already exists. Skipping file merge", process_target)
            data = pd.read_csv(process_target)

        if data is None:
            res = []
            for category in self.CATS:
                self._logger.info("Merging category '%s'", category)
                res += self._process_files(extract_target, category)

            data = pd.DataFrame(res, columns=[self.FEATURE_TEXT_LABEL, self.TARGET_LABEL])
            data.to_csv(process_target, index=False)

        return TextDataset(
            data=data,
            text_column=self.FEATURE_TEXT_LABEL,
            target_column=self.TARGET_LABEL
        )


class Iris(LogMixin):
    def __init__(self, base_path: str):
        self.base_path = str(base_path)

    def retrieve(self) -> ClassificationDataset:
        """Retrieve the iris datasets from ics."""
        data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        target = 'class'

        data_path = os.path.join(self.base_path, 'iris.csv')
        if not os.path.isfile(data_path):
            os.makedirs(self.base_path, exist_ok=True)
            download_url(data_url, data_path)
        else:
            self._logger.info("Data file '%s' already exists. Skipping download...", data_path)

        data = pd.read_csv(data_path, header=None, names=features + [target])

        return ClassificationDataset(
            data=data,
            feature_columns=features,
            target_column=target
        )


class CatsVsDogs(LogMixin):

    _EXTRACTED_FOLDER_NAME = 'PetImages'
    _VECTOR_FILE = 'vector.pkl'
    _IMAGE_SIZE_WIDTH = 100
    _IMAGE_SIZE_HEIGHT = 100

    _FEATURE_LABEL = 'image'
    _TARGET_LABEL = 'target'

    def __init__(self, base_path: str):
        self.base_path = str(base_path)

    def retrieve(self) -> ClassificationDataset:
        """Retrieve the cats and dogs dataset from microsoft."""
        data_url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/' \
                   'kagglecatsanddogs_3367a.zip'

        zip_file = os.path.join(self.base_path, 'images.zip')
        if not os.path.isfile(zip_file):
            os.makedirs(self.base_path, exist_ok=True)
            download_url(data_url, zip_file)
        else:
            self._logger.info("Data file '%s' already exists. Skipping download...", zip_file)

        # Unzip file
        extract_target = os.path.join(self.base_path, self._EXTRACTED_FOLDER_NAME)
        if not os.path.exists(extract_target):
            self._logger.info("Unzipping '%s' to '%s'", zip_file, self.base_path)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.base_path)
        else:
            self._logger.info("'%s' already exists. Skipping unzip...", extract_target)

        vector_file = os.path.join(self.base_path, self._VECTOR_FILE)
        if not os.path.exists(vector_file):
            self._logger.info("Creating vector from '%s'", extract_target)
            dataset_root = extract_target
            categories = ['Dog', 'Cat']
            data = []

            for animal_type in categories:
                path = os.path.join(dataset_root, animal_type)
                class_num = categories.index(animal_type)
                for img in os.listdir(path):
                    try:
                        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                        resized = cv2.resize(img_array, (self._IMAGE_SIZE_WIDTH, self._IMAGE_SIZE_HEIGHT))
                        data.append((resized, class_num))
                    except Exception:  # pylint: disable=broad-except
                        pass

            self._logger.info("Storing result in '%s'", vector_file)
            random.shuffle(data)
            with open(vector_file, 'wb') as fp:
                pickle.dump(data, fp)
        else:
            self._logger.info("Loading vector from '%s'", vector_file)
            with open(vector_file, 'rb') as fp:
                data = pickle.load(fp)

        df = pd.DataFrame(data, columns=[self._FEATURE_LABEL, self._TARGET_LABEL])

        return ClassificationDataset(
            data=df,
            feature_columns=[self._FEATURE_LABEL],
            target_column=self._TARGET_LABEL
        )
