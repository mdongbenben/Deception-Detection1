import re
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, Activation, LSTM

def load_data(maxword):
	Xtrain = []
	ytrain = []
	Xdevel = []
	ydevel = []
	with open('lab/ComParE2016_Deception.tsv') as f:
		for line in f.readlines()[1: ]:
			lists = line.split(',')
			filepath = 'wav/' + lists[0]
			sampling_freq, audio = wavfile.read(filepath)
			mfcc_features = mfcc(audio, sampling_freq)
			mfcc_features = mfcc_features.reshape(1, mfcc_features.size)
			if re.match('train', lists[0]):
				Xtrain.extend(mfcc_features)
				if re.match('D', lists[1]):
					ytrain.append(1)
				else:
					ytrain.append(0)
			else:
				Xdevel.extend(mfcc_features)
				if re.match('D', lists[1]):
					ydevel.append(1)
				else:
					ydevel.append(0)
	Xtrain = sequence.pad_sequences(Xtrain, maxlen = maxword * 13)
	Xtrain = Xtrain.reshape(Xtrain.shape[0], maxword, 13)
	Xdevel = sequence.pad_sequences(Xdevel, maxlen = maxword * 13)
	Xdevel = Xdevel.reshape(Xdevel.shape[0], maxword, 13)
	return np.array(Xtrain), np.array(ytrain), np.array(Xdevel), np.array(ydevel)
	
def train(Xtrain, ytrain, Xdevel, ydevel, maxword):
	model = Sequential()
	model.add(LSTM(128, input_shape = (maxword, 13, ), return_sequences = True))
	model.add(Dropout(0.2))
	model.add(LSTM(64, return_sequences = True))
	model.add(Dropout(0.2))
	model.add(LSTM(32))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation = 'sigmoid'))
	model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
	model.fit(Xtrain, ytrain, validation_data = (Xdevel, ydevel), epochs = 10, batch_size = 10, verbose = 1)
	return model
	
def test(model, Xtrain, ytrain, Xdevel, ydevel):
	print(model.summary())
	score = model.evaluate(Xtrain, ytrain)
	print("Model performance on train dataset")
	print(score)
	print("Model performance on development dataset")
	score = model.evaluate(Xdevel, ydevel)
	print(score)

Xtrain, ytrain, Xdevel, ydevel = load_data(760)
model = train(Xtrain, ytrain, Xdevel, ydevel, 760)
test(model, Xtrain, ytrain, Xdevel, ydevel)
