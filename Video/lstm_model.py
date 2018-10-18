import math
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
import functools
import numpy as np


def lazy_property(function):
	attribute = '_cache_' + function.__name__

	@property
	@functools.wraps(function)
	def wrapper(self):
		if not hasattr(self, attribute):
			setattr(self, attribute, function(self))
		return getattr(self, attribute)
	return wrapper

class MultiLstm(object):
	def __init__(self,
				 seq_images,
				 labels,
				 dropout,
				 num_hiddens,
				 regularization_value,
				 initial_learning_rate,
				 tau):
		self.data = seq_images
		self.target = labels
		self.dropout = dropout
		self._num_hiddens = num_hiddens
		self.regularizer = layers.l2_regularizer(regularization_value)
		self._initial_learning_rate = initial_learning_rate
		self.tau = tau
		self.prediction
		self.error
		self.optimize
		self.accuracy
		self.test_prediction
	 
	
	@lazy_property
	def prediction(self):
		cells = []
		layeroutput=[]
		layererror=[]
		herror=[]
		houtput=[]
		for i in range(len(self._num_hiddens)):
			cell = rnn.LSTMCell(self._num_hiddens[i])
			cell = rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
			cells.append(cell)
			localnetwork=rnn.MultiRNNCell(cells) 
			output, _ = tf.nn.dynamic_rnn(localnetwork, self.data, dtype=tf.float32)
			output = tf.transpose(output, [1, 0, 2])           
			last = tf.gather(output, int(output.get_shape()[0]) - 1)
			layeroutput.append(last)
			logits = layers.fully_connected(last, int(self.target.get_shape()[1]),activation_fn=None,	weights_regularizer=self.regularizer)
			mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(logits, 1))
			layererror.append(tf.reduce_mean(tf.cast(mistakes, tf.float32)))  
			if i==0:
				houtput.append(last)
				herror.append(layererror[i])
			if ((i>0) and (herror[i-1]<=layererror[i])) is not None:
				 alpha=(tf.log((herror[i-1]/layererror[i])))/2
				 houtput.append(alpha*last+(1-alpha)*houtput[i-1])          
			if ((i>0) and (herror[i-1]>layererror[i])) is not None:
				 temp=layeroutput[0]*0
				 for ii in range(0,i):
					 if (ii>=self.tau) is not None:
						 temp=temp+layeroutput[ii]
				 houtput.append(temp/(i-self.tau))            
			hlast=houtput[i]
			hlogits = layers.fully_connected(hlast,	int(self.target.get_shape()[1]),	activation_fn=None,	weights_regularizer=self.regularizer)
			hmistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(hlogits, 1))
			herror.append(tf.reduce_mean(tf.cast(hmistakes, tf.float32)))  
		return hlogits

	@lazy_property
	def cost(self):
		loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=self.prediction, labels=self.target))
		reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_term = layers.apply_regularization(self.regularizer,
											   reg_variables)
		loss_op += reg_term
		return loss_op

	@lazy_property
	def optimize(self):
		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(self._initial_learning_rate,
												   global_step,
												   100000,
												   0.96,
												   staircase=True)
		learning_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost,
														global_step=global_step)
		return learning_step

	@lazy_property
	def accuracy(self):
		corrects = tf.equal(
			tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
		return tf.reduce_mean(tf.cast(corrects, tf.float32))

	@lazy_property
	def test_prediction(self):
		return tf.argmax(self.prediction, 1)	
   
   
	
	@lazy_property
	def error(self):
		mistakes = tf.not_equal(
			tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
		return tf.reduce_mean(tf.cast(mistakes, tf.float32))

	@lazy_property
	def _stacked_lstm_cell(self):
		cells = []
		for i in range(len(self._num_hiddens)):
			cell = rnn.LSTMCell(self._num_hiddens[i])
			cell = rnn.DropoutWrapper(cell,
									  output_keep_prob=self.dropout)
			cells.append(cell)
		return rnn.MultiRNNCell(cells)
