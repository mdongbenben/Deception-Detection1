from data import  DataSet
from lstm_model import MultiLstm
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold
import math
from itertools import chain



def train(seq_length):
	# Set variables.
	nb_epoch = 10000
	batch_size = 32
	regularization_value = 0.004
	learning_rate = 0.001
	nb_feature = 2048
	database = 'Deception'

	data = DataSet(database, seq_length)
	skf = StratifiedKFold(n_splits=5)
	nb_class = len(data.classes)

	# Set model
	num_hiddens = [90, 90, 90]
	seq_images = tf.placeholder(dtype=tf.float32,
								 shape=[None,
										seq_length,
										nb_feature])
	input_labels = tf.placeholder(dtype=tf.float32,
								  shape=[None, nb_class])
	drop_out = tf.placeholder(dtype=tf.float32)
	rnn_model = MultiLstm(seq_images,
						  input_labels,
						  drop_out,
						  num_hiddens,
						  regularization_value,
						  learning_rate,
						  5)

	# training
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		all_samples_prediction, all_samples_true = [], []
		for train, test in skf.split(data.data,data.label):
			genaretor = data.frame_generator_train(batch_size, train)
			for epoch in range(nb_epoch):
				batch_seq_images, batch_labels = next(genaretor)
				sess.run(rnn_model.optimize,
						 feed_dict={seq_images: batch_seq_images,
									input_labels: batch_labels,
									drop_out: 0.5})
				accuracy = sess.run(rnn_model.accuracy,
									feed_dict={seq_images: batch_seq_images,
											   input_labels: batch_labels,
											   drop_out: 1.})
				print("Epoch {:2d}, training accuracy {:4.2f}".format(epoch,
					accuracy))

			test_data, test_label = data.get_set_from_data(test)
			all_samples_true.append(test_label)
			for test_epoch in range(1, math.ceil(len(test) / batch_size) + 1):
				test_batch_images = data.frame_generator_test(
					test_data, batch_size, test_epoch)
				test_predict_labels = sess.run(rnn_model.test_prediction,
												feed_dict={seq_images: test_batch_images,
														   drop_out: 1.})
				all_samples_prediction.append(list(test_predict_labels))
	all_samples_prediction = np.array(list(chain.from_iterable(
	    all_samples_prediction)), dtype=float)
	all_samples_true = np.array(list(chain.from_iterable(
	    all_samples_true)), dtype=float)
	test_accuracy_cv = np.mean(np.equal(all_samples_prediction,
		 all_samples_true))
	print("CV test accuracy {:4.2f}".format(test_accuracy_cv))


def main():
	time_steps = 50
	train(time_steps)


if __name__ == '__main__':
	main()
