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
from keras.preprocessing.sequence import pad_sequences

# Load spaCy once
nlp = spacy.load('en_core_web_md')

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
        print("Precision: {0}, Recall: {1}, F1 Score: {2},\nAccuracy: {3}".format(self.precision[-1],
		                                                                          self.recall_score[-1],
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
		nlpdoc = [token.lemma_ for token in nlpdoc if (len(token.lemma_) > 1) or(token.lemma_.isalnum())]
		return nlpdoc


########################################################
#                   WORD VECTORIZER                    #
########################################################
# Uses spaCy's inbuilt pre-trained word2vec model      #
########################################################

def string_to_wordvec(string, token):
	tokens = token(string)
	return np.array([nlp(token).vector for token in tokens])

def vectorize_words(pairs, label_pairs, tokenizer):
	word_vecs = []
	label_after_vec = []
	paired_up = zip(pairs, label_pairs)
	# List containing ((id, string), (id, label))
	for text_tuple, label_tuple in paired_up:
		paragraphs = text_tuple[1].split("\n")
		for paragraph in paragraphs:
			word_vecs.append(string_to_wordvec(paragraph, tokenizer))
			label_after_vec.append(label_tuple[1])
		print(len(word_vecs))
	return word_vecs, label_after_vec

# Transform words into tokens
def transform_words(x_dataframe, y_dataframe):
	lemma_token = LemmaTokenizer()
	data, labels = vectorize_words(x_dataframe.items(), y_dataframe.items(), lemma_token)
	# Truncate longer to 1000 and pad shorter to 1000
	data = pad_sequences(data, maxlen=1000, padding='post')
	return data, labels


########################################################
#                  INPUT SANITIZER                     #
########################################################
# Sanitize: remove HTML tags, digits, punctuation
def read_input(path_to_csv):
	dataframe = pandas.read_csv(path_to_csv)
	dataframe['CLEAN'] = dataframe['TEXT'].str.replace('<[^<]+?>|^\d+\s|\s\d+\s|\s\d+$|[^\w\s]', '')
	return dataframe


########################################################
#              LSTM MODEL IMPLEMENTATION               #
########################################################
# Create model
def create_model(timesteps, dimensions, train_data, train_labels, val_data, val_labels, mtrcs):
	model = Sequential()
	model.add(LSTM(20, input_shape=(timesteps, dimensions), 
	          activation='relu', return_sequences=True))
	model.add(Dense(10, activation='relu'))
	model.add(Dropout(0.2))
	model.add(LSTM(10, input_shape=(timesteps, dimensions), activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Loss is t * log(y) + (1 - t) * log (1 - y)
	sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

	print("Fitting")
	model.fit(train_data, train_labels, epochs=3, batch_size=1, verbose=2, 
	          validation_data=[val_data, val_labels], callbacks=[mtrcs])
	scores = model.evaluate(X_val, y_val, verbose=0)
	print('Accuracy on validation: %.5f' % (scores[1]*100))
	return model


########################################################
#                     MAIN METHOD                      #
########################################################

if __name__ == '__main__':
	# SETUP
	# Set seed for reproducability
	np.random.seed(1337)
	metrics = Metrics()

	dir_to_data = 'data/news_ds.csv'

	df = read_input(dir_to_data)

	# Work with a small subset
	num_sample = 10
	raw_data = df['CLEAN'][:num_sample]
	raw_labels = df['LABEL'][:num_sample]

	data, labels = transform_words(raw_data, raw_labels)

	# Split ratios
	train_ratio, val_ratio = .7, .2
	test_ratio = 1 - train_ratio - val_ratio
	print("Splitting")
	X, X_test, y, y_test = train_test_split(data, labels, test_size=test_ratio)
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio/(1 - test_ratio))

	dataset_size, num_timesteps, num_dimensions = X_train.shape

	lstm_model = create_model(num_timesteps, num_dimensions, X_train, y_train, X_val, y_val, metrics)
	
	print("Predicting")
	# TODO: Implement testing

