from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def main():

    # load data
    train = pd.read_csv('fulltrain.csv', header=None)
    test = pd.read_csv('balancedtest.csv', header=None)
    X_train = train[1]
    y_train = train[0]
    X_test = test[1]
    y_test = test[0]
    
    # Build the model
    model = make_pipeline(CountVectorizer(analyzer='word',stop_words= 'english',lowercase=True, tokenizer = LemmaTokenizer()), MultinomialNB())
    # model = make_pipeline(TfidfVectorizer(analyzer='word',stop_words= 'english',lowercase=True, tokenizer = LemmaTokenizer()), MultinomialNB())
    model.fit(X_train, y_train)
    predicted_categories = model.predict(X_test)


    # accuracy
    print("The accuracy = {}".format(accuracy_score(y_test, predicted_categories)))
    # f1 score
    print('f1 score = {}'.format(f1_score(y_test, predicted_categories, average='macro')))

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()

