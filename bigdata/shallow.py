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

from collections import Counter
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

# TODO: Clean data
# def get_tokens(string):
# 	return [x.lower() for x in re.split("\W+", string) if x]

# def identify_tokens(pairs):
# 	tokens = set()
# 	for id_num, string in pairs:
# 		new_tokens = get_tokens(string)
# 		tokens = tokens.union(new_tokens)
# 	return sorted(tokens)

# def count_tokens(pairs, unique_tokens):
# 	tfs = []
# 	for id_num, string in pairs:
# 		token_counter = Counter()
# 		token_counter.update(get_tokens(string))
# 		tf_counts = list(map(lambda el: token_counter[el], unique_tokens))
# 		tfs.append(tf_counts)
# 		print(len(tfs))
# 	return np.array(tfs)

# def save_tokens(pairs, tokens, fname):
# 	f = open(fname, 'w')
# 	f.write("ID" + ",".join(tokens) + "\n")
# 	for id_num, string in pairs:
# 		print(id_num)
# 		token_counter = Counter()
# 		token_counter.update(get_tokens(string))
# 		tf_counts = list(map(lambda el: token_counter[el], tokens))
# 		f.write(",".join(map(str, [id_num] + tf_counts)) + "\n")
# 	f.close()

# def tf(data):
# 	return data

# def idf(data):
# 	N, D = data.shape
# 	idf = np.zeros(D)
# 	for term_idx in range(D):
# 		# Apply the 1 + trick to avoid dividing by 0
# 		idf[term_idx] = np.log(D / (1 + np.count_nonzero(data[:, term_idx])))
# 	return idf

# def tf_idf(data):
# 	return tf(data) * idf(data)

# def n_grams(pairs):
# 	pass

# def save_data(header, **kwargs):
# 	for name, data in kwargs.items():
# 		np.savetxt('data/{0}.csv'.format(name), data, delimiter=',', header=header)

# Set seed for reproducability
np.random.seed(1337)

# ### INITIAL
df = pandas.read_csv('data/news_ds.csv')
# Remove HTML tags, digits, punctuation
df['CLEAN'] = df['TEXT'].str.replace('<[^<]+?>|^\d+\s|\s\d+\s|\s\d+$|[^\w\s]', '')

# # Work with a small subset
num_sample = 250
raw_data = df['CLEAN'][:num_sample]
raw_labels = df['LABEL'][:num_sample]
# Split ratios
# 75% training, 25% test
train_ratio, val_ratio = .75, .0
test_ratio = 1 - train_ratio - val_ratio

print("Splitting")
# sklearn randomly removes the training set
# X_df and X_test_df gives back the text itself
# y_df and y_test_df gives back the labels
# PREV. VER.
# X_df, X_test_df, y_df, y_test_df = train_test_split(raw_data, raw_labels, test_size=test_ratio)
X_train, X_test, y_train, y_test = train_test_split(raw_data, raw_labels, test_size=test_ratio, random_state=1337)

# Use sklearn to randomly pick validation set
# PREV. VER.
# X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(X_df, y_df, test_size = val_ratio/(1-test_ratio))

print("Counting")

# Identify all unique words in training set
# unique_tokens = identify_tokens(X_train_df.items())
# X_train, X_val, X_test = map(lambda partition: count_tokens(partition.items(), unique_tokens), [X_train_df, X_val_df, X_test_df])
# y_train, y_val, y_test = map(lambda partition: np.array([x[1] for x in partition.items()]), [y_train_df, y_val_df, y_test_df])

# Apply transformations
# X_train_tf, X_val_tf, X_test_tf = map(lambda partition: tf_idf(partition), [X_train, X_val, X_test])

tf = TfidfVectorizer(input='content', lowercase=True, analyzer='word', ngram_range=(2,4), min_df=0, stop_words=STOP_WORDS, tokenizer=LemmaTokenizer())
X_train_dtm = tf.fit_transform(X_train)
X_test_dtm = tf.transform(X_test)

print("Fitting")
# Fit
clf = MultinomialNB()
clf.fit(X_train_dtm, y_train)

print("Predicting")

# Check validation score
# TODO: Check if positives means fake(0) or real(1)
y_pred = clf.predict(X_test_dtm)
# cf_matrix = confusion_matrix(y_pred, y_val)
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

# print("Saving")
# save_tokens(raw_data.items(), tokens, "data/X_tokens.csv")

# ### WITH TOKENS
# X_tokens = pandas.read_csv("data/X_tokens.csv")
