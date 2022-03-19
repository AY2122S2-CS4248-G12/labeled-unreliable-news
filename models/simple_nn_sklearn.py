import _csv
import csv
import sys
from typing import List

# Change system path to base directory.
sys.path.append("..")
from preprocessing.preprocessor import Preprocessor

from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# Control which linguistic preprocessing steps should run.
preprocessor = Preprocessor(perform_case_folding=True,
                            remove_stop_words=True,
                            remove_punctuation=False,
                            perform_lemmatization=False,
                            perform_stemming=False)

csv.field_size_limit(2147483647)

def parse_dataset(path: str, limit: int = 0):
    documents: List[str] = []
    labels: List[int] = []

    with open(path) as dataset:
        reader: _csv.reader = csv.reader(dataset)
        row: List[str]
        i = 0
        for row in reader:
            if limit > 0 and i == limit: break
            label: int = row[0]
            labels.append(int(label) - 1)
            document: str = row[1]
            documents.append(document)
            i += 1
    return (documents, labels)

def main():
    documents, labels = parse_dataset('../raw_data/fulltrain.csv')
    test_documents, test_labels = parse_dataset('../raw_data/balancedtest.csv')

    model = Pipeline([
        ('prep', CountVectorizer(stop_words='english')),
        ('clf', MLPClassifier(early_stopping=True)),
    ])

    model.fit(documents, labels)

    preds = model.predict(documents)
    score = f1_score(labels, preds, average='macro')
    print('score on validation on training set = {}'.format(score))

    test_preds = model.predict(test_documents)
    score = f1_score(test_labels, test_preds, average='macro')
    print('score on validation on test set = {}'.format(score))

if __name__ == "__main__":
    main()
