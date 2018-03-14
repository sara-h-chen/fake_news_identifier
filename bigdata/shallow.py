import argparse
import pandas
import spacy
import numpy as np
import sklearn.metrics as met

from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


########################################################
#                   LEMMA TOKENIZER                    #
########################################################
# Takes the base form of every word, and keeps it only #
# if it's a valid word or phrase:                      #
# https://stackoverflow.com/questions/45196312/spacy   #
# -and-scikit-learn-vectorizer                         #
########################################################

class LemmaTokenizer(object):
    def __init__(self):
        self.spacynlp = spacy.load('en_core_web_lg')

    def __call__(self, doc):
        nlpdoc = self.spacynlp(doc)
        nlpdoc = [token.lemma_ for token in nlpdoc if (len(token.lemma_) > 1) or (token.lemma_.isalnum())]
        return nlpdoc


########################################################
#                  INPUT SANITIZER                     #
########################################################
# Sanitize: remove HTML tags, digits, punctuation
def read_input(path_to_csv):
    dataframe = pandas.read_csv(path_to_csv)
    dataframe['CLEAN'] = dataframe['TEXT'].str.replace('<[^<]+?>|\d|[^\w\s]|^https?:\/\/.*[\r\n]*|\n', '')
    print("Data processed")
    return dataframe


########################################################
#                    DATA SPLITTER                     #
########################################################
# Use sklearn to randomly partition the data set:      #
# - X_df and X_test_df gives back the text itself      #
# - y_df and true_y_df gives back the labels           #
########################################################

def split_data(data, labels, ratio):
    trainx, testx, trainy, testy = train_test_split(data, labels, test_size=ratio, random_state=1337)
    return trainx, testx, trainy, testy


########################################################
#                 TF-IDF VECTORIZER                    #
########################################################
# Use sklearn's inbuilt tf-idf vectorizer.             #
# Attaches the lemma tokenizer. Produces 3- to 5-gram. #
# Converts everything to lowercase when preprocessing. #
########################################################
# Apply transformations
def calculate_tfidf(use_tfidf, trainx, testx):
    tf = TfidfVectorizer(input='content', lowercase=True, analyzer='word',
                         ngram_range=(3, 5), min_df=0, stop_words=STOP_WORDS,
                         tokenizer=None, use_idf=use_tfidf)
    X_train_dtm = tf.fit_transform(trainx)
    X_test_dtm = tf.transform(testx)
    return X_train_dtm, X_test_dtm


########################################################
#              MULTINOMIAL NAIVE BAYES                 #
########################################################
# Positive outcome means real(1)
# Fit
def fit_mnb(train_datamatrix, test_datamatrix, train_labels):
    clf = MultinomialNB()
    clf.fit(train_datamatrix, train_labels)
    y_pred = clf.predict(test_datamatrix)
    return y_pred


########################################################
#                    PRINT METRICS                     #
########################################################
# tn, fn, fp, tp = cf_matrix.flatten()
# accuracy = (tp + tn) / (tn + fp + fn + tp)
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# F1 = 2 * tp / (2 * tp + fp + fn)
def calc_metrics(true_y, predicted_y):
    cf_matrix = met.confusion_matrix(true_y, predicted_y)
    accuracy = met.accuracy_score(true_y, predicted_y)
    precision = met.precision_score(true_y, predicted_y)
    recall = met.recall_score(true_y, predicted_y)
    F1 = met.f1_score(true_y, predicted_y)
    null_accuracy = true_y.value_counts().head(1) / len(true_y)
    print(
        "Accuracy: {0}, Precision: {1},\nRecall: {2}, F1: {3}, \nNull Accuracy: {4}".format(accuracy, precision, recall,
                                                                                           F1, null_accuracy))
    return


########################################################
#                     MAIN METHOD                      #
########################################################

if __name__ == '__main__':
    # Read from command line
    use_idf = True

    # SETUP
    # Set seed for reproducability
    np.random.seed(1337)
    dir_to_data = 'data/news_ds.csv'

    print("Reading data ...")
    df = read_input(dir_to_data)

    # Work with a small subset
    num_sample = 50
    raw_data = df['CLEAN'][:]
    raw_labels = df['LABEL'][:]

    # Split ratios
    # 75% training, 25% test
    train_ratio, val_ratio = .75, .0
    test_ratio = 1 - train_ratio - val_ratio
    print("Splitting")
    X_train, X_test, y_train, true_y = split_data(raw_data, raw_labels, test_ratio)

    print("Counting")
    X_train_datamatrix, X_test_datamatrix = calculate_tfidf(use_idf, X_train, X_test)

    print("Fitting")
    print("Predicting")
    predicted_label = fit_mnb(X_train_datamatrix, X_test_datamatrix, y_train)
    calc_metrics(true_y, predicted_label)
