import string
from typing import List


def remove_punctuation(words: List[str]) -> List[str]:
    return [word for word in words if word not in string.punctuation]
