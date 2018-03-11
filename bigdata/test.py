import pandas
import csv
import numpy as np
import spacy
import time
import re
import json
import operator

from collections import Counter

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
nlp = spacy.load('en_core_web_md')

# vocabulary = set()
common_words = set()
counter = Counter()

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

def create_embedding_matrix(label_encoder, max_valid, sorted_list):
    embedding_matrix = np.zeros((len(common_words), 300))
    for word, appearances in sorted_list:
        index = label_encoder.transform([word])
        embedding_matrix[index] = nlp(word).vector
    return embedding_matrix


def string_to_wordvec(string, label_encoder):
    string = [x for x in string if x in common_words]
    transformed_sentence = label_encoder.transform(string)
    return transformed_sentence


def build_common_words_set(max_size):
    sorted_occurrences = list(sorted(counter.items(), key=operator.itemgetter(1), reverse=True))
    sorted_occurrences = sorted_occurrences[:max_size]
    common_words.update([x[0] for x in sorted_occurrences])
    return sorted_occurrences


def vectorize_words(pairs, label_pairs, max_valid_words):
    sorted_words = build_common_words_set(max_valid_words)

    le = LabelEncoder()
    transformed_word_bag = le.fit(list(common_words))

    embedding_matrix = create_embedding_matrix(le, max_valid_words, sorted_words)
    vectorized_words = []
    # List containing ((id, string), (id, label))
    article_count = 0
    for string_list in pairs:
        transformed = string_to_wordvec(string_list, le)
        vectorized_words.append(transformed)
        print("Article %d transformed" % article_count)
        article_count += 1
    return vectorized_words, label_pairs, embedding_matrix


# Transform words into tokens
def transform_words(x_dataframe, y_dataframe, max_words):
    data, labels, embed_matrix = vectorize_words(x_dataframe, y_dataframe, max_words)
    # Truncate longer to 1000 and pad shorter to 1000
    data = pad_sequences(data, maxlen=1000, padding='pre')
    return data, labels, embed_matrix


########################################################
#                   HELPER FUNCTIONS                   #
########################################################

def increment_count(list_of_words):
    for word in list_of_words:
        counter[word] += 1
    return


########################################################
#                  INPUT SANITIZER                     #
########################################################
# Sanitize: remove HTML tags, digits, punctuation
# Remove stop words
# Find vocabulary set
def read_input(path_to_csv):
    dataframe = pandas.read_csv(path_to_csv)
    dataframe['CLEAN'] = dataframe['TEXT'].str.lower().str.replace('<[^<]+?>|\d|[^\w\s]|^https?:\/\/.*[\r\n]*|\n', '')
    dataframe = dataframe.replace(r'^\s*$', np.nan, regex=True)
    dataframe = dataframe.dropna()
    dataframe['SPLIT'] = dataframe['CLEAN'].str.split()
    dataframe['SPLIT'] = dataframe['SPLIT'].apply(lambda x: [item for item in x if item not in STOP_WORDS])
    dataframe['SPLIT'].apply(lambda x: increment_count(x))
    # dataframe['SPLIT'].apply(lambda x: vocabulary.update(x))
    # print(dataframe['SPLIT'])
    return dataframe


########################################################
#              LSTM MODEL IMPLEMENTATION               #
########################################################
# Create model
def create_model(timesteps, dimensions, train_data, train_labels, val_data, val_labels, mtrcs, embedding_matrix):
    embedding_layer = Embedding(output_dim=300, input_dim=embedding_matrix.shape[0], trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])

    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(20, input_shape=train_data.shape[1:], return_sequences=True, activation='tanh'))
    model.add(Dropout(0.05))
    model.add(LSTM(10, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    # Loss is t * log(y) + (1 - t) * log (1 - y)
    # sgd = optimizers.SGD(lr=0.01, clipvalue=0.3)
    model.summary()
    adam = optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-3, decay=0.0, amsgrad=True)
    # rmsprop = optimizers.RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    # print(index_dict)
    # print(val_data, val_labels)
    print("Fitting")
    model.fit(train_data, train_labels, epochs=10, batch_size=16, verbose=1,
              validation_data=(val_data, val_labels), callbacks=[mtrcs])
    scores = model.evaluate(val_data, val_labels, verbose=0)
    print('Accuracy on validation: %.5f' % (scores[1] * 100))
    return model


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

    df = read_input(dir_to_data)

    # Work with a small subset
    num_sample = 500
    raw_data = df['SPLIT'][:num_sample]
    raw_labels = df['LABEL'][:num_sample]

    # e_matrix = create_embedding_matrix(word_dictionary)
    max_recognized_words = 5000
    data, labels, e_matrix = transform_words(raw_data, raw_labels, max_recognized_words)

    print("Reading: ", time.time() - t0, "seconds wall time")

    t0 = time.time()
    # Split ratios
    train_ratio, val_ratio = .7, .2
    test_ratio = 1 - train_ratio - val_ratio
    X, X_test, y, y_test = train_test_split(data, labels, test_size=test_ratio, random_state=1337)

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=val_ratio / (1 - test_ratio), random_state=1337)
    print("Splitting into %d train, %d validation and %d test" % (len(X_train), len(X_val), len(X_test)))

    print("Splitting and creating embedding matrix: ", time.time() - t0, "seconds wall time")

    num_timesteps, num_dimensions = e_matrix.shape

    lstm_model = create_model(num_timesteps, num_dimensions, np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), metrics, np.array(e_matrix))

    # print("Predicting")
    # TODO: Implement testing
