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

from keras import optimizers
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

# Load spaCy once
nlp = spacy.load('en_core_web_lg')

index_dict = {}
word_dictionary = {}
# Keep track of transformed articles to pickle
transformed_articles = {}

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
        self.accuracy.append(met.accuracy_score(targ, predict))
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

def create_embedding_matrix(bag_of_words):
    embedding_matrix = np.zeros((len(bag_of_words.keys()), 300))
    for word, vector in bag_of_words.items():
        index_tracker = index_dict[word]
        embedding_matrix[index_tracker] = vector
    return embedding_matrix


def string_to_wordvec(string, ctr):
    returned_sentence = []
    for token in string:
        # if token is unseen
        if token not in index_dict:
            index_dict[token] = ctr
            ctr += 1
            if token in nlp.vocab:
                word_dictionary[token] = nlp(token).vector
        returned_sentence.append(index_dict[token])
    return returned_sentence


def vectorize_words(pairs, label_pairs, index):
    vectorized_words = []
    # List containing ((id, string), (id, label))
    article_count = 0
    for string in pairs:
        transformed_sentence = string_to_wordvec(string, index)
        vectorized_words.append(transformed_sentence)
        # DEBUG: Take this out when not pickling
        transformed_articles[article_count] = transformed_sentence
        print("Article %d transformed" % article_count)
        article_count += 1
    return vectorized_words, [x for x in label_pairs]


# Transform words into tokens
def transform_words(x_dataframe, y_dataframe, counter):
    data, labels = vectorize_words(x_dataframe, y_dataframe, counter)
    # Truncate longer to 1000 and pad shorter to 1000
    data = pad_sequences(data, maxlen=1000, padding='post')
    return data, labels


########################################################
#                  INPUT SANITIZER                     #
########################################################
# Sanitize: remove HTML tags, digits, punctuation
# Remove stop words
def read_input(path_to_csv, lemmatizer):
    dataframe = pandas.read_csv(path_to_csv)
    dataframe['CLEAN'] = dataframe['TEXT'].str.lower().str.replace('<[^<]+?>|^\d+\s|\s\d+\s|\s\d+$|[^\w\s]|^https?:\/\/.*[\r\n]*', '')
    dataframe['SPLIT'] = dataframe['CLEAN'].apply(lambda x: lemmatizer(x))
    dataframe['SPLIT'] = dataframe['SPLIT'].apply(lambda x: [item for item in x if item not in STOP_WORDS])
    # print(dataframe['SPLIT'])
    return dataframe


########################################################
#              LSTM MODEL IMPLEMENTATION               #
########################################################
# Create model
def create_model(timesteps, dimensions, train_data, train_labels, val_data, val_labels, mtrcs, embedding_matrix):
    model = Sequential()
    model.add(Embedding(timesteps, 300, weights=[embedding_matrix], input_length=1000, trainable=False))
    model.add(LSTM(20, input_shape=train_data.shape[1:], return_sequences=True, activation='relu'))
    model.add(Dropout(0.02))
    model.add(LSTM(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Loss is t * log(y) + (1 - t) * log (1 - y)
    # sgd = optimizers.SGD(lr=0.01, clipvalue=0.3)
    model.summary()
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    # print(index_dict)
    # print(val_data, val_labels)
    print("Fitting")
    model.fit(train_data, train_labels, epochs=3, batch_size=32, verbose=1,
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

    t0 = time.time()

    print("Reading data")
    dir_to_data = 'data/news_ds.csv'

    tokenizer = LemmaTokenizer()
    df = read_input(dir_to_data, tokenizer)

    # Work with a small subset
    num_sample = 500
    raw_data = df['SPLIT'][:num_sample]
    raw_labels = df['LABEL'][:num_sample]

    counter = 0
    data, labels = transform_words(raw_data, raw_labels, counter)

    with open('pickled_data.json', 'w') as outfile:
        json.dump(data, outfile)
    with open('picked_labels.json', 'w') as outfile:
        json.dump(labels, outfile)
    with open('pickled_indices.json', 'w') as outfile:
        serializable_dict = {k: v.tolist() for k, v in word_dictionary.items()}
        json.dump(serializable_dict, outfile)
    with open('pickled_words.json', 'w') as outfile:
        json.dump(index_dict, outfile)

    print("Reading: ", time.time() - t0, "seconds wall time")

    t0 = time.time()
    # Split ratios
    train_ratio, val_ratio = .7, .2
    test_ratio = 1 - train_ratio - val_ratio
    X, X_test, y, y_test = train_test_split(data, labels, test_size=test_ratio, random_state=1337)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio / (1 - test_ratio), random_state=1337)
    print("Splitting into %d train, %d validation and %d test" % (len(X_train), len(X_val), len(X_test)))

    e_matrix = create_embedding_matrix(word_dictionary)

    print("Splitting and creating embedding matrix: ", time.time() - t0, "seconds wall time")

    num_timesteps, num_dimensions = e_matrix.shape

    lstm_model = create_model(num_timesteps, num_dimensions, np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), metrics, np.array(e_matrix))

    # print("Predicting")
    # TODO: Implement testing
