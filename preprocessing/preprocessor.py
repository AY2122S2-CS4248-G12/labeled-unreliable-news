from dataclasses import dataclass
from typing import List

from nltk import sent_tokenize, word_tokenize

from preprocessing.case_folding import convert_to_lowercase
from preprocessing.lemmatization import lemmatize
from preprocessing.punctuation_removal import remove_punctuation
from preprocessing.stemming import stem
from preprocessing.stop_word_removal import remove_stop_words


@dataclass
class Preprocessor:
    perform_case_folding: bool = False
    remove_stop_words: bool = False
    remove_punctuation: bool = False
    perform_lemmatization: bool = False
    perform_stemming: bool = False

    def process(self, document: str) -> List[str]:
        sentences: List[str] = sent_tokenize(document)
        all_words: List[str] = []
        for i, sentence in enumerate(sentences):
            words: List[str] = word_tokenize(sentence)

            # Perform POS tagging before any other processing for greater accuracy.
            if self.perform_lemmatization:
                words = lemmatize(words)

            if self.perform_stemming:
                words = stem(words)

            if self.perform_case_folding:
                words = map(convert_to_lowercase, words)

            if self.remove_stop_words:
                words = remove_stop_words(words)

            if self.remove_punctuation:
                words = remove_punctuation(words)

            all_words += words

        return all_words
