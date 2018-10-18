import re
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding

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
	Xtrain = sequence.pad_sequences(Xtrain, maxlen = maxword)
	Xdevel = sequence.pad_sequences(Xdevel, maxlen = maxword)
	return np.array(Xtrain), np.array(ytrain), np.array(Xdevel), np.array(ydevel)
	
def train(Xtrain, ytrain, Xdevel, ydevel, maxword):
	model = Sequential()
	model.add(Dense(4096, input_shape = (maxword, ), activation = 'relu'))
	model.add(Dense(512, activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(8, activation = 'relu'))
	model.add(Dense(1, activation = 'sigmoid'))
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	history = model.fit(Xtrain, ytrain, validation_data = (Xdevel, ydevel), epochs = 5, batch_size = 10, verbose = 1)
	model.save('model.h5')
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc = 'upper left')
	plt.show()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss') 
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc = 'upper right')
	plt.show()
	return model
	
def test(model, Xtrain, ytrain, Xdevel, ydevel):
	print(model.summary())
	score = model.evaluate(Xtrain, ytrain)
	print("Model performance on train dataset")
	print(score)
	print("Model performance on development dataset")
	score = model.evaluate(Xdevel, ydevel)
	print(score)

Xtrain, ytrain, Xdevel, ydevel = load_data(760*13)
model = train(Xtrain, ytrain, Xdevel, ydevel, 760*13)
test(model, Xtrain, ytrain, Xdevel, ydevel)
