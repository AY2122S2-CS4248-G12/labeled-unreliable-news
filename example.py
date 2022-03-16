import _csv
import csv
import sys
from typing import List

from preprocessing.preprocessor import Preprocessor

# Control which linguistic preprocessing steps should run.
preprocessor = Preprocessor(perform_case_folding=True,
                            remove_stop_words=True,
                            remove_punctuation=False,
                            perform_lemmatization=False,
                            perform_stemming=True)

# Increase the field size limit.
csv.field_size_limit(sys.maxsize)
with open('../raw_data/fulltrain.csv') as training_dataset:
    reader: _csv.reader = csv.reader(training_dataset)
    row: List[str]
    for row in reader:
        document: str = row[1]
        print(preprocessor.process(document))
