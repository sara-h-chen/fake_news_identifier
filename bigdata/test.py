import argparse
import pandas
import csv
import numpy as np
import spacy
import time
import re
import json

from spacy.lang.en.stop_words import STOP_WORDS

import sklearn.metrics as met
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras import optimizers
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences

# Load spaCy once
nlp = spacy.load('en_core_web_lg')

vocabulary = set()
# index_dict = {}
# word_dictionary = {}
# Keep track of transformed articles to pickle
# transformed_articles = {}

########################################################
#                METRICS CALLBACK CLASS                #
########################################################
# Wrap scikit metrics into a class because Keras can   #
# only provide these metrics as a callback function.   #
########################################################

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.confusion.append(met.confusion_matrix(targ, predict))
        self.precision.append(met.precision_score(targ, predict))
        self.recall.append(met.recall_score(targ, predict))
        self.f1s.append(met.f1_score(targ, predict))
        self.accuracy.append(met.accuracy_score(targ, predict.round()))
        print("\nPrecision: {0}, Recall: {1}, F1 Score: {2},\nAccuracy: {3}".format(self.precision[-1],
                                                                                  self.recall[-1],
                                                                                  self.f1s[-1],
                                                                                  self.accuracy[-1]))
        return


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
        self.spacynlp = nlp

    def __call__(self, doc):
        nlpdoc = self.spacynlp(doc)
        nlpdoc = [token.lemma_ for token in nlpdoc if (len(token.lemma_) > 1) or (token.lemma_.isalnum())]
        return nlpdoc


########################################################
#                   WORD VECTORIZER                    #
########################################################
# Uses spaCy's inbuilt pre-trained word2vec model      #
########################################################

# def create_embedding_matrix():
#     embedding_matrix = np.zeros((len(vocabulary), 300))
#     for word in vocabulary: 
#         embedding_matrix[index_tracker] = vector
#     return embedding_matrix


def string_to_wordvec(string, label_encoder, embedding_matrix):
    transformed_string = label_encoder.transform(string)
    for token in transformed_string:
        # if token is unseen
        embedding_matrix[token] = nlp(label_encoder.inverse_transform([token])).vector
    return transformed_string


def vectorize_words(pairs, label_pairs, index):
    vectorized_words = []
    e_matrix = np.zeros((len(vocabulary), 300))
    # List containing ((id, string), (id, label))
    le = LabelEncoder()
    le.fit(list(vocabulary))
    for string in pairs:
        vectorized_words.append(string_to_wordvec(string, le, e_matrix))
        # DEBUG: Take this out when not pickling
        print("Article %d transformed" % article_count)
        article_count += 1
    return vectorized_words, [x for x in label_pairs], e_matrix


# Transform words into tokens
def transform_words(x_dataframe, y_dataframe, counter):
    data, labels, embedding_mtrx = vectorize_words(x_dataframe, y_dataframe, counter)
    # Truncate longer to 1000 and pad shorter to 1000
    data = pad_sequences(data, maxlen=1000, padding='post')
    return data, labels, embedding_mtrx


########################################################
#                  INPUT SANITIZER                     #
########################################################
# Sanitize: remove HTML tags, digits, punctuation
# Remove stop words
def read_input(path_to_csv):
    dataframe = pandas.read_csv(path_to_csv)
    dataframe['CLEAN'] = dataframe['TEXT'].str.lower().str.replace('<[^<]+?>|^\d+\s|\s\d+\s|\s\d+$|[^\w\s]|^https?:\/\/.*[\r\n]*', '')
    dataframe['SPLIT'] = dataframe['CLEAN'].str.split()
    dataframe['SPLIT'] = dataframe['SPLIT'].apply(lambda x: [item for item in x if item not in STOP_WORDS])
    dataframe['SPLIT'] = dataframe['SPLIT'].apply(lambda x: [vocabulary.add(item) for item in x if item not in vocabulary])
    print(dataframe['SPLIT'])
    return dataframe


########################################################
#              LSTM MODEL IMPLEMENTATION               #
########################################################
# Create model
def create_model(timesteps, dimensions, train_data, train_labels, val_data, val_labels, mtrcs, embedding_matrix):
    model = Sequential()
    model.add(Embedding(timesteps, 300, weights=[embedding_matrix], input_length=1000, trainable=False))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.02))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    # Loss is t * log(y) + (1 - t) * log (1 - y)
    # sgd = optimizers.SGD(lr=0.01, clipvalue=0.3)
    model.summary()
    # adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=True)
    rmsprop = optimizers.RMSprop()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # print(index_dict)
    # print(val_data, val_labels)
    print("Fitting")
    model.fit(train_data, train_labels, epochs=10, batch_size=64, verbose=1,
              validation_data=(val_data, val_labels), callbacks=[mtrcs])
    scores = model.evaluate(val_data, val_labels, verbose=0)
    print('Accuracy on validation: %.5f' % (scores[1] * 100))
    return model


########################################################
#                  HELPER FUNCTIONS                    #
########################################################

# def read_json():
#     with open("pickled_data.json", 'r') as f:
#         reconstructed_summaries = np.zeroes((


########################################################
#                     MAIN METHOD                      #
########################################################

if __name__ == '__main__':
    # SETUP
    # Set seed for reproducability
    np.random.seed(1337)
    metrics = Metrics()
    parser = argparse.ArgumentParser()
    # TODO: Add argument parser params
    args = parser.parse_args()

    t0 = time.time()

    print("Reading data")
    dir_to_data = 'data/news_ds.csv'

    # if args.read:
    #     df = pandas.read_csv('dumped_df.csv')
    # else:
    df = read_input(dir_to_data)

    # Work with a small subset
    num_sample = 500
    raw_data = df['SPLIT'][:num_sample]
    raw_labels = df['LABEL'][:num_sample]

    # if args.dump:
    # 	df.to_csv('dumped_df.csv', sep=',', encoding='utf-8', index=False)

    counter = 0
    data, labels, e_matrix = transform_words(raw_data, raw_labels, counter)

    # with open('pickled_data.json', 'w') as outfile:
    #     json.dump(data, outfile)
    # with open('picked_labels.json', 'w') as outfile:
    #     json.dump(labels, outfile)
    # with open('pickled_indices.json', 'w') as outfile:
    #     serializable_dict = {k: v.tolist() for k, v in word_dictionary.items()}
    #     json.dump(serializable_dict, outfile)
    # with open('pickled_words.json', 'w') as outfile:
    #     json.dump(index_dict, outfile)

    print("Reading: ", time.time() - t0, "seconds wall time")

    t0 = time.time()
    # Split ratios
    train_ratio, val_ratio = .7, .2
    test_ratio = 1 - train_ratio - val_ratio
    X, X_test, y, y_test = train_test_split(data, labels, test_size=test_ratio, random_state=1337)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio / (1 - test_ratio), random_state=1337)
    print("Splitting into %d train, %d validation and %d test" % (len(X_train), len(X_val), len(X_test)))

    # e_matrix = create_embedding_matrix(word_dictionary)

    print("Splitting and creating embedding matrix: ", time.time() - t0, "seconds wall time")

    num_timesteps, num_dimensions = e_matrix.shape

    lstm_model = create_model(num_timesteps, num_dimensions, np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), metrics, np.array(e_matrix))

    # print("Predicting")
    # TODO: Implement testing
