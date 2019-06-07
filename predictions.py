# This file contains all the modules that vote for positive/negative sentiments
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import keras 
import pickle

def predict(corpus,debug=True):
	vote = []
	vote.append(model_LSTM(corpus))
	print(vote)

def model_LSTM(corpus):
	tokenizer = Tokenizer(num_words=2500)
	with open('weights/tokenizer.pickle', 'rb') as handle:
   		tokenizer = pickle.load(handle)

	X = tokenizer.texts_to_sequences(corpus)
	X = pad_sequences(X,maxlen=128)
	model = Sequential()
	model.add(Embedding(2500,128,input_length=X.shape[1]))
	model.add(LSTM(300, dropout=0.2))
	model.add(Dense(2,activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
	model.load_weights("weights/lstm.h5")
	return model.predict(X)

if __name__ == '__main__':
	predict(['I love you', 'I hate you'])