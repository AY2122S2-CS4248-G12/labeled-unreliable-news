import _csv
import csv
from typing import Callable, List, Optional, Union, Tuple

from torch.utils.data import Dataset

from datasets.utils import transform_label


class FakeNewsDetectionDataset(Dataset):
    """Represents the fake news detection dataset."""
    def __init__(self, raw_data_file: str, transform: Optional[Callable[[str], List[str]]] = None,
                 target_transform: Optional[Callable[[Union[int, str]], int]] = transform_label) -> None:
        """Initialises the dataset from the specified raw data file & transformation functions.

        For simplicity, the entire dataset will be loaded into memory from the file.

        Args:
            raw_data_file:
                The CSV file containing the dataset.
            transform:
                The transformation function to be run on the sentences.
            target_transform:
                The transformation function to be run on the labels.
        """
        self.transform = transform
        self.target_transform = target_transform

        self.sentences: List[str] = []
        self.labels: List[str] = []
        with open(raw_data_file) as dataset:
            reader: _csv.reader = csv.reader(dataset)
            row: List[str]
            for row in reader:
                label: str = row[0]
                sentence: str = row[1]
                self.sentences.append(sentence)
                self.labels.append(label)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            The length of the dataset.
        """
        assert len(self.sentences) == len(self.labels)
        return len(self.sentences)

    def __getitem__(self, index: int) -> Tuple[Union[str, List[str]], Union[str, int]]:
        """Returns the (sentence, label) pair at the specified index.

        The transformation functions `transform` and `target_transform` are applied on the
        sentence and label respectively. As a result, the output types might differ depending
        on the output types of the transformation functions.

        Args:
            index:
                The index of the item to be returned.

        Returns:
            A 2-tuple of (sentence, label). Note that sentence and/or label might have been transformed.
        """
        sentence: Union[str, List[str]] = self.sentences[index]
        label: Union[str, int] = self.labels[index]

        if self.transform:
            sentence = self.transform(sentence)
        if self.target_transform:
            label = self.target_transform(label)

        return sentence, label
