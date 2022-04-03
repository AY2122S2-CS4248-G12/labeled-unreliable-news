import _csv
import csv
from typing import Callable, List, Optional, Union, Tuple

from torch.utils.data import Dataset

from datasets.utils import transform_label


csv.field_size_limit(2147483647)


class FakeNewsDetectionDataset(Dataset):
    """Represents the fake news detection dataset."""
    def __init__(self, raw_data_file: str, transform: Optional[Callable[[str], List[str]]] = None,
                 target_transform: Optional[Callable[[Union[int, str]], int]] = transform_label) -> None:
        """Initialises the dataset from the specified raw data file & transformation functions.

        For simplicity, the entire dataset will be loaded into memory from the file.
        Then, the transformation functions will be applied before storing the sentences and labels.

        Args:
            raw_data_file:
                The CSV file containing the dataset.
            transform:
                The transformation function to be run on the sentences.
            target_transform:
                The transformation function to be run on the labels.
        """
        self.sentences: List[Union[str, List[str]]] = []
        self.labels: List[Union[str, int]] = []

        with open(raw_data_file) as dataset:
            reader: _csv.reader = csv.reader(dataset)
            row: List[str]
            for row in reader:
                label: Union[str, int] = row[0]
                sentence: Union[str, List[str]] = row[1]

                # Apply transformations.
                if transform:
                    sentence = transform(sentence)
                if target_transform:
                    label = target_transform(label)

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
        return sentence, label
