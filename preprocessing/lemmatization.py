from typing import List, Tuple, Dict

import nltk
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet, WordNetCorpusReader


def download_averaged_perceptron_tagger_if_not_exists() -> None:
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')


def download_wordnet_corpora_if_not_exists() -> None:
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')


def download_omw_corpora_if_not_exists() -> None:
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')


download_averaged_perceptron_tagger_if_not_exists()
download_wordnet_corpora_if_not_exists()
download_omw_corpora_if_not_exists()

pos_map: Dict[str, WordNetCorpusReader] = {
    'NN': wordnet.NOUN,
    'JJ': wordnet.ADJ,
    'VB': wordnet.VERB,
    'RB': wordnet.ADV
}
lemmatizer = WordNetLemmatizer()


def lemmatize(words: List[str]) -> List[str]:
    tagged_words: List[Tuple[str, str]] = pos_tag(words)
    lemmatized_words: List[str] = []
    for tagged_word in tagged_words:
        word: str = tagged_word[0]
        pos: wordnet = pos_map.get(tagged_word[1], wordnet.NOUN)
        lemmatized_words.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_words
