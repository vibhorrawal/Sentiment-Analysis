# This file is used to train models and save weight(s) if any
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import keras 
import numpy as np 
from sklearn.model_selection import train_test_split
import pickle

def trainLSTM(x,y,loadweights=True):
	tokenizer = Tokenizer(num_words=2500)
	if loadweights:
		with open('weights/tokenizer.pickle', 'rb') as handle:
   			tokenizer = pickle.load(handle)
	tokenizer.fit_on_texts(x)
	X = tokenizer.texts_to_sequences(x)
	X = pad_sequences(X,maxlen=100)
	Y = []
	for val in y:
	    if(val == 0):
	        Y.append([1,0])
	    else:
	        Y.append([0,1])
	Y = np.array(Y)
	x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.8)
	model = Sequential()
	model.add(Embedding(2500,128,input_length=X.shape[1]))
	model.add(LSTM(300, dropout=0.2))
	model.add(Dense(2,activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
	if loadweights:
		model.load_weights("weights/lstm.h5")
		print("Loaded model from disk")
	model.fit(x_train,y_train,epochs=5,verbose=2,batch_size=32)
	print(model.evaluate(x_test,y_test))

	model.save_weights("weights/lstm.h5")
	with open('weights/tokenizer.pickle', 'wb') as handle:
	    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Saved model to disk")

if __name__ == '__main__':
	with open("test-folder/amazon_cells_labelled.txt") as f1:
		lines = f1.readlines()

	with open("test-folder/imdb_labelled.txt") as f1:
	    temp = f1.readlines()
	    lines=lines+temp

	with open("test-folder/yelp_labelled.txt") as f1:
	    temp = f1.readlines()
	    lines = lines + temp
	x = []
	y = []
	for value in lines:
	    temp = value.split('\t')
	    x.append(temp[0])
	    temp[1].replace('\n','')
	    y.append(int(temp[1]))

	trainLSTM(x, y,	loadweights=False)    