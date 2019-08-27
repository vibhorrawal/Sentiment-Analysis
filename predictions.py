# This file contains all the modules that vote for positive/negative sentiments
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
import sklearn
import keras
import pickle

# to disable tensorflow warnings and outputs:
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
debug = False

def predict(corpus,debug=debug,accuracy=False):
	vote = []
	tokenizer = Tokenizer(num_words=2500)
	with open('weights/tokenizer.pickle', 'rb') as handle:
   		tokenizer = pickle.load(handle)


	X = tokenizer.texts_to_sequences(corpus)
	X = pad_sequences(X,maxlen=140)

	# vote.append(model_LSTM(X))
	# vote.append(model_MultinomialNB(X))
	# vote.append(model_BernoulliNB(X))
	# vote.append(model_GaussianNB(X))
	# vote.append(model_LogisticRegression(X))
	# vote.append(model_SVC(X))
	# vote.append(model_LinearSVC(X))
	# vote.append(model_NuSVC(X))
	vote = model_LSTM(X,debug) + model_MultinomialNB(X,debug) + model_BernoulliNB(X,debug) + model_GaussianNB(X,debug) + model_LogisticRegression(X,debug) + model_SVC(X,debug) + model_LinearSVC(X,debug) + model_NuSVC(X,debug)
	vote = vote / 8
	if debug:
		print('Vote: ',vote)
	if not accuracy:
		return vote
	vote = model_LSTM(X,debug) + model_SVC(X,debug) + model_NuSVC(X,debug)
	vote = vote / 3
	if debug:
		print('Vote(acc): ', vote)
	return vote
#-------------------------------------------------------------------------------
def model_LSTM(X,debug=False):
	model = Sequential()
	model.add(Embedding(2500,140,input_length=X.shape[1]))
	model.add(LSTM(300, dropout=0.2))
	model.add(Dense(2,activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
	model.load_weights("weights/lstm.h5")
	if debug:
		print("LSTM")
		print(model.predict(X))
	return model.predict(X)
#-------------------------------------------------------------------------------
def model_MultinomialNB(X,debug=False):
	MN_classifier = MultinomialNB()
	with open('weights/MultiNomialNB.pickle', 'rb') as handle:
		MN_classifier = pickle.load(handle)
	if debug:
		print("MultiNomialNB")
		print(MN_classifier.predict_proba(X))
	return MN_classifier.predict_proba(X)
#-------------------------------------------------------------------------------
def model_BernoulliNB(X,debug=False):
	BN_classifier = BernoulliNB()
	with open('weights/BernoulliNB.pickle', 'rb') as handle:
		BN_classifier = pickle.load(handle)
	if debug:
		print("BernoulliNB")
		print(BN_classifier.predict_proba(X))
	return BN_classifier.predict_proba(X)
#-------------------------------------------------------------------------------
def model_GaussianNB(X,debug=False):
	classifier = GaussianNB()
	with open('weights/GaussianNB.pickle', 'rb') as handle:
		classifier = pickle.load(handle)
	if debug:
		print("GaussianNB")
		print(classifier.predict_proba(X))
	return classifier.predict_proba(X)
#-------------------------------------------------------------------------------
def model_LogisticRegression(X,debug=False):
	classifier = LogisticRegression(solver='liblinear', max_iter=2000)
	with open('weights/LogisticRegression.pickle', 'rb') as handle:
		classifier = pickle.load(handle)
	if debug:
		print("LogisticRegression")
		print(classifier.predict_proba(X))
	return classifier.predict_proba(X)
#-------------------------------------------------------------------------------
def model_SVC(X,debug=False):
	classifier = sklearn.svm.SVC(probability=True)
	with open('weights/SVC.pickle', 'rb') as handle:
		classifier = pickle.load(handle)
	prob = classifier.predict(X)
	res = []
	for it in prob:
		if it == 1:
			res.append([0,1])
		else:
			res.append([1,0])
	from numpy import array
	if debug:
		print("SVC")
		print(array(res))
	return array(res)
#-------------------------------------------------------------------------------
def model_LinearSVC(X,debug=False):
	classifier = sklearn.svm.LinearSVC()
	with open('weights/LinearSVC.pickle', 'rb') as handle:
		classifier = pickle.load(handle)
	prob = classifier.predict(X)
	res = []
	for it in prob:
		if it == 1:
			res.append([0,1])
		else:
			res.append([1,0])
	from numpy import array
	if debug:
		print("LinearSVC")
		print(array(res))
	return array(res)
#-------------------------------------------------------------------------------
def model_NuSVC(X,debug=False):
	classifier = sklearn.svm.NuSVC()
	with open('weights/NuSVC.pickle', 'rb') as handle:
		classifier = pickle.load(handle)
	prob = classifier.predict(X)
	# return prob
	res = []
	for it in prob:
		if it == 1:
			res.append([0,1])
		else:
			res.append([1,0])
	from numpy import array
	if debug:
		print("NuSVC")
		print(array(res))
	return array(res)
#-------------------------------------------------------------------------------

if __name__ == '__main__':
	print(predict(['I love you', 'What a waste of money and time!.','I hate you'],accuracy=True))
