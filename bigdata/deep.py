import pandas
import re
import numpy as np
import spacy

import sklearn.metrics as met
from sklearn.model_selection import train_test_split

from keras import optimizers
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

# Set seed for reproducability
np.random.seed(1337)

# Load spaCy once
nlp = spacy.load('en_core_web_md')


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
        print("Precision: {0}, Recall: {1}, F1 Score: {2},\nAccuracy: {3}".format(self.precision[-1], self.recall[-1], self.f1s[-1], self.accuracy[-1]))
        return

metrics = Metrics()

class LemmaTokenizer(object):
	def __init__(self):
		self.spacynlp = nlp
		
	def __call__(self, doc):
		nlpdoc = self.spacynlp(doc)
		nlpdoc = [token.lemma_ for token in nlpdoc if (len(token.lemma_) > 1) or(token.lemma_.isalnum())]
		return nlpdoc

def string_to_wordvec(string, token):
	tokens = token(string)
	return np.array([nlp(token).vector for token in tokens])

def vectorize_words(pairs, label_pairs, tokenizer):
	word_vecs = []
	label_after_vec = []
	paired_up = zip(pairs, label_pairs)
	# List of ((id, string),(id, label))
	for text_tuple, label_tuple in paired_up:
		paragraphs = text_tuple[1].split("\n")
		for paragraph in paragraphs:
			word_vecs.append(string_to_wordvec(paragraph, tokenizer))
			label_after_vec.append(label_tuple[1])
		print(len(word_vecs))
	return word_vecs, label_after_vec


### INITIAL
df = pandas.read_csv('data/news_ds.csv')
# Remove HTML tags, digits, punctuation
df['CLEAN'] = df['TEXT'].str.replace('<[^<]+?>|^\d+\s|\s\d+\s|\s\d+$|[^\w\s]', '')

# Work with a small subset
num_sample = 30
raw_data = df['CLEAN'][:num_sample]
raw_labels = df['LABEL'][:num_sample]

lemma_token = LemmaTokenizer()
data, labels = vectorize_words(raw_data.items(), raw_labels.items(), lemma_token)

# Truncate longer to 1000 and pad shorter to 1000
data = pad_sequences(data, maxlen=1000, padding='post')

# Split ratios
train_ratio, val_ratio = .7, .2
test_ratio = 1 - train_ratio - val_ratio

print("Splitting")

X, X_test, y, y_test = train_test_split(data, labels, test_size=test_ratio)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio/(1 - test_ratio))

dataset_size, num_timesteps, num_dimensions = X_train.shape

# Create model
model = Sequential()
model.add(LSTM(20, input_shape=(num_timesteps, num_dimensions), activation='relu', return_sequences=True))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(10, input_shape=(num_timesteps, num_dimensions), activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
# Loss is t * log(y) + (1 - t) * log (1 - y)
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("Fitting")
model.fit(X_train, y_train, epochs=3, batch_size=1, verbose=2, validation_data=[X_val, y_val], callbacks=[metrics])

print("Predicting")

scores = model.evaluate(X_val, y_val, verbose=0)
print('Accuracy: %.5f' % (scores[1]*100))
# Check validation score
# TODO: Check if positives means fake(0) or real(1)
# y_pred = np.round(model.predict(X_val))
# cf_matrix = confusion_matrix(y_pred, y_val)
# tn, fn, fp, tp = cf_matrix.flatten()
# accuracy = (tp + tn) / (tn + fp + fn + tp)
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# F1 = 2 * tp / (2 * tp + fp + fn)
# print("Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}".format(accuracy, precision, recall, F1))