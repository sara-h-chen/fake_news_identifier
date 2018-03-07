import pandas
import re
import spacy
import string
import numpy as np
import sklearn.metrics as met

from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from spacy.lang.en.stop_words import STOP_WORDS

from scipy.sparse.csr import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class LemmaTokenizer(object):
	def __init__(self):
		self.spacynlp = spacy.load('en')
		
	def __call__(self, doc):
		nlpdoc = self.spacynlp(doc)
		nlpdoc = [token.lemma_ for token in nlpdoc if (len(token.lemma_) > 1) or(token.lemma_.isalnum())]
		return nlpdoc

# Set seed for reproducability
np.random.seed(1337)

# TODO: Put everything into functions
### INITIAL
df = pandas.read_csv('data/news_ds.csv')
# Remove HTML tags, digits, punctuation
df['CLEAN'] = df['TEXT'].str.replace('<[^<]+?>|^\d+\s|\s\d+\s|\s\d+$|[^\w\s]', '')

# Work with a small subset
num_sample = 250
raw_data = df['CLEAN'][:]
raw_labels = df['LABEL'][:]
# Split ratios
# 75% training, 25% test
train_ratio, val_ratio = .75, .0
test_ratio = 1 - train_ratio - val_ratio

print("Splitting")
# sklearn randomly removes the training set
# X_df and X_test_df gives back the text itself
# y_df and y_test_df gives back the labels
X_train, X_test, y_train, y_test = train_test_split(raw_data, raw_labels, test_size=test_ratio, random_state=1337)

print("Counting")

# Apply transformations
tf = TfidfVectorizer(input='content', lowercase=True, analyzer='word', ngram_range=(3,5), min_df=0, stop_words=STOP_WORDS, tokenizer=LemmaTokenizer())
X_train_dtm = tf.fit_transform(X_train)
X_test_dtm = tf.transform(X_test)

print("Fitting")

# Fit
clf = MultinomialNB()
clf.fit(X_train_dtm, y_train)

print("Predicting")

# Positive means real(1)
y_pred = clf.predict(X_test_dtm)
# tn, fn, fp, tp = cf_matrix.flatten()
# accuracy = (tp + tn) / (tn + fp + fn + tp)
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# F1 = 2 * tp / (2 * tp + fp + fn)
cf_matrix = met.confusion_matrix(y_test, y_pred)
accuracy = met.accuracy_score(y_test, y_pred)
precision = met.precision_score(y_test, y_pred)
recall = met.recall_score(y_test, y_pred)
F1 = met.f1_score(y_test, y_pred)
null_accuracy = y_test.value_counts().head(1) / len(y_test)
print("Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}, \nNull Accuracy: {4}".format(accuracy, precision, recall, F1, null_accuracy))

