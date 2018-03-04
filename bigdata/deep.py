import pandas
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import spacy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences

# Set seed for reproducability
np.random.seed(1337)

def string_to_wordvec(string, nlp):
	tokens = nlp(string)
	return np.array([token.vector for token in tokens])

def process_data(pairs, nlp):
	word_vecs = []
	for id_num, string in pairs:
		word_vecs.append(string_to_wordvec(string, nlp))
		print(len(word_vecs))
	return word_vecs

### INITIAL

nlp = spacy.load('en', vectors='en_glove_cc_300_1m')

df = pandas.read_csv('data/news_ds.csv')

# Work with a small subset
num_sample = 40
raw_data = df['TEXT'][:num_sample]
raw_labels = df['LABEL'][:num_sample]

data = process_data(raw_data.items(), nlp)
labels = np.array([x[1] for x in raw_labels.items()])

# Truncate longer to 1000 and pad shorter to 1000
data = pad_sequences(data, maxlen=1000, padding='post')

# Split ratios
train_ratio, val_ratio = .7, .2
test_ratio = 1 - train_ratio - val_ratio

print("Splitting")

X, X_test, y, y_test = train_test_split(data, labels, test_size=test_ratio)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_ratio/(1 - test_ratio))

dataset_size, num_timesteps, num_dimensions = X_train.shape

# Create model
model = Sequential()
model.add(LSTM(10, input_shape=(num_timesteps, num_dimensions), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Loss is t * log(y) + (1 - t) * log (1 - y)
model.compile(loss='binary_crossentropy', optimizer='sgd')

print("Fitting")
model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=2, validation_data=[X_val, y_val])

print("Predicting")

# Check validation score
# TODO: Check if positives means fake(0) or real(1)
y_pred = np.round(model.predict(X_val))
cf_matrix = confusion_matrix(y_pred, y_val)
tn, fn, fp, tp = cf_matrix.flatten()
accuracy = (tp + tn) / (tn + fp + fn + tp)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
F1 = 2 * tp / (2 * tp + fp + fn)
print("Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}".format(accuracy, precision, recall, F1))