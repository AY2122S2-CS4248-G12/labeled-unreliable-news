import sys
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

sys.path.append("..")
from preprocessing.preprocessor import Preprocessor

preprocessor = Preprocessor(perform_case_folding=True,
                            remove_stop_words=False,
                            remove_punctuation=False,
                            perform_lemmatization=False,
                            perform_stemming=False)

def main():

    # load data
    train = pd.read_csv("../raw_data/fulltrain.csv", header=None)
    test = pd.read_csv("../raw_data/balancedtest.csv", header=None)
    X_train = train[1]
    y_train = train[0]
    X_test = test[1]
    y_test = test[0]
    
    # Build the model
    model = make_pipeline(CountVectorizer(tokenizer=preprocessor.process), MultinomialNB())
    model.fit(X_train, y_train)
    predicted_categories = model.predict(X_test)


    # accuracy
    print("The accuracy = {}".format(accuracy_score(y_test, predicted_categories)))
    # f1 score
    print('f1 score = {}'.format(f1_score(y_test, predicted_categories, average='macro')))

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()

