from typing import List

from nltk import PorterStemmer

stemmer = PorterStemmer()


def stem(words: List[str]) -> List[str]:
    return [stemmer.stem(word) for word in words]
