# This file is used to train models and save weight(s) if any
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn

# to disable tensorflow warnings and outputs:
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
loadweights = True
#-------------------------------------------------------------------------------
def trainLSTM(x,y,loadweights=True):
	print("Training LSTM...")
	Y = []
	for val in y:
	    if(val == 0):
	        Y.append([1,0])
	    else:
	        Y.append([0,1])
	Y = np.array(Y)
	x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.8)
	model = Sequential()
	model.add(Embedding(2500,140,input_length=X.shape[1]))
	model.add(LSTM(300, dropout=0.2))
	model.add(Dense(2,activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
	if loadweights:
		model.load_weights("weights/lstm.h5")
		print("Loaded model from disk")
	model.fit(x_train,y_train,epochs=20,verbose=2,batch_size=32)
	print(model.evaluate(x_test,y_test))
	model.save_weights("weights/lstm.h5")
	# print("Saved model to disk")
#-------------------------------------------------------------------------------
def trainBernoulliNB(X,y,loadweights):
	print("Training BernoulliNB...")
	BN_classifier = BernoulliNB()
	if loadweights:
		with open('weights/BernoulliNB.pickle', 'rb') as handle:
			BN_classifier = pickle.load(handle)
	for _ in range(10):
		BN_classifier.partial_fit(X,y,classes=[0,1])
	with open('weights/BernoulliNB.pickle', 'wb') as handle:
		pickle.dump(BN_classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print (BN_classifier.score(X,y))
#-------------------------------------------------------------------------------
def trainMultinomialNB(X,y,loadweights):
	print("Training MultiNomialNB...")
	MN_classifier = MultinomialNB()
	if loadweights:
		with open('weights/MultiNomialNB.pickle', 'rb') as handle:
			MN_classifier = pickle.load(handle)
	for _ in range(10):
		MN_classifier.partial_fit(X,y,classes=[0,1])
	with open('weights/MultiNomialNB.pickle', 'wb') as handle:
		pickle.dump(MN_classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print (MN_classifier.score(X,y))
#-------------------------------------------------------------------------------
def trainGaussianNB(X,y,loadweights):
	print("Training GaussianNB...")
	classifier = GaussianNB()
	if loadweights:
		with open('weights/GaussianNB.pickle', 'rb') as handle:
			classifier = pickle.load(handle)
	for _ in range(10):
		classifier.partial_fit(X,y,classes=[0,1])
	with open('weights/GaussianNB.pickle', 'wb') as handle:
		pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print (classifier.score(X,y))
#-------------------------------------------------------------------------------
def trainLogisticRegression(X,y,loadweights):
	print("Training LogisticRegression...")
	classifier = LogisticRegression(solver='liblinear', max_iter=2000)
	if loadweights:
		with open('weights/LogisticRegression.pickle', 'rb') as handle:
			classifier = pickle.load(handle)
	classifier.fit(X,y)
	with open('weights/LogisticRegression.pickle', 'wb') as handle:
		pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print (classifier.score(X,y))
#-------------------------------------------------------------------------------
def trainSVC(X,y,loadweights):
	print("Training SVC...")
	classifier = sklearn.svm.SVC(probability=True)
	if loadweights:
		with open('weights/SVC.pickle', 'rb') as handle:
			classifier = pickle.load(handle)
	classifier.fit(X=X,y=y)
	with open('weights/SVC.pickle', 'wb') as handle:
		pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
	# print(classifier.predict_proba(X[:10]))
	print (classifier.score(X,y))
#-------------------------------------------------------------------------------
def trainLinearSVC(X,y,loadweights):
	print("Training LinearSVC...")
	classifier = sklearn.svm.LinearSVC(max_iter=8000,tol=3)
	if loadweights:
		with open('weights/LinearSVC.pickle', 'rb') as handle:
			classifier = pickle.load(handle)
	classifier.fit(X=X,y=y)
	with open('weights/LinearSVC.pickle', 'wb') as handle:
		pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print (classifier.score(X,y))
#-------------------------------------------------------------------------------
def trainNuSVC(X,y,loadweights):
	print("Training NuSVC...")
	classifier = sklearn.svm.NuSVC()
	if loadweights:
		with open('weights/NuSVC.pickle', 'rb') as handle:
			classifier = pickle.load(handle)
	classifier.fit(X=X,y=y)
	with open('weights/NuSVC.pickle', 'wb') as handle:
		pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print (classifier.score(X,y))
#-------------------------------------------------------------------------------

if __name__ == '__main__':
	with open("training-data/amazon_cells_labelled.txt") as f1:
		lines = f1.readlines()

	with open("training-data/imdb_labelled.txt") as f1:
	    temp = f1.readlines()
	    lines=lines+temp

	with open("training-data/yelp_labelled.txt") as f1:
	    temp = f1.readlines()
	    lines = lines + temp

	x = []
	y = []
	for value in lines:
	    temp = value.split('\t')
	    x.append(temp[0].lower())
	    temp[1].replace('\n','')
	    y.append(int(temp[1]))

	with open("training-data/positive.txt" ,encoding='latin-1') as f1:
	    temp = f1.readlines()
	    for line in temp:
	        x.append(line[:-4].lower())
	        y.append(1)

	with open("training-data/negative.txt" ,encoding='latin-1') as f1:
	    temp = f1.readlines()
	    for line in temp:
	        x.append(line[:-4].lower())
	        y.append(0)

	print('x',x[:10])
	print('y',y[:10])

	tokenizer = Tokenizer(num_words=2500)
	if loadweights:
		with open('weights/tokenizer.pickle', 'rb') as handle:
   			tokenizer = pickle.load(handle)
	tokenizer.fit_on_texts(x)
	X = tokenizer.texts_to_sequences(x)
	X = pad_sequences(X,maxlen=140)

	with open('weights/tokenizer.pickle', 'wb') as handle:
	    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("size of data:",len(y))
	# training models here
	trainLSTM(X, y,	loadweights)
	trainMultinomialNB(X, y, loadweights)
	trainBernoulliNB(X,y,loadweights)
	trainGaussianNB(X, y, loadweights)
	trainLogisticRegression(X, y, loadweights)
	trainSVC(X,y,loadweights)
	trainLinearSVC(X,y,loadweights)
	trainNuSVC(X,y,loadweights)
