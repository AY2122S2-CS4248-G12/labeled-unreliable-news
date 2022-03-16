from typing import List

import nltk
from nltk.corpus import stopwords


def download_stopwords_corpora_if_not_exists() -> None:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


download_stopwords_corpora_if_not_exists()


def remove_stop_words(words: List[str]) -> List[str]:
    stop_words = stopwords.words('english')
    return [word for word in words if word.lower() not in stop_words]
