import csv
import numpy as np
import random
import glob
import os
import pandas as pd
import sys
import operator
from itertools import chain



class DataSet():
    def __init__(self, database_name, seq_length=40):
        self.seq_length = seq_length
        self.sequence_path = '.\\data\\sequences\\'
        self.datasetname = database_name
        self.data = self.get_data()
        self.classes = self.get_classes()
        self.label = self.get_label()


    def get_data(self):
        """Load our data from file."""
        with open(os.path.join('.\\data', self.datasetname + '.csv'),
                  mode='r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
        return data

    def get_label(self):
        label = []
        for item in self.data:
            label.append(item[2])
        return label



    def get_classes(self):
        classes = []
        for item in self.data:
            if item[0] not in classes:
                classes.append(item[0])
        classes = sorted(classes)
        return classes

    def get_class_one_hot(self, index):
        """
        Given a class as a string, return its number
        in the classes lit. This let us encode and one-hot
        it for training.
        """
        label_hot = np.zeros(len(self.classes))
        label_hot[index] = 1
        return label_hot

    def get_all_sequences_in_memory(self, train_test):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right DataSet
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memorry for %sing." %(len(data), train_test))

        X, y = [], []
        for row in data:
            sequence = self.get_extracted_sequence(data_type, row)
            if sequence is None:
                print("Can't find sequence. Did you generate them?")
                raise

        X.append(sequence)
        y.append(self.get_class_one_hot(row[1]))

    def get_extracted_sequence(self, sample):
        filename = sample[1]
        path = self.sequence_path + filename + '-' +str(self.seq_length) + \
               '-' + 'features' + '.txt'
        if os.path.isfile(path):
            features = pd.read_csv(path, sep=" ", header=None)
            return features.values
        else:
            return None

    def get_set_from_data(self, train_test):
        data = []
        label = []
        for item in train_test:
            sample = self.data[item]
            data.append(sample)
            label.append(sample[3])
        return data, label

    def frame_generator_test(self, test_data, batch_size, epoch):
        X = []
        for i in range(batch_size):
            if (epoch-1)*batch_size + i < len(test_data):
                sequence = None
                sample = test_data[(epoch-1)*batch_size + i]
                sequence = self.get_extracted_sequence(sample)
                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    sys.exit()
                X.append(sequence)
            else:
                break
        return np.array(X)

    def frame_generator_train(self, batch_size, train_set):
        """
        Return a generator that we can use to train on. There are a couple
        different things we can return:
        data_type: 'features'
        """
        # Get the right dataset for the generator.
        data, _ = self.get_set_from_data(train_set)

        while True:
            X, y = [], []

            # Generate batch_sie samples.
            for _ in range(batch_size):
                sequence = None

                # Get a random sample
                sample = random.choice(data)

                # Get the squence from disk
                sequence = self.get_extracted_sequence(sample)
                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    sys.exit()
                X.append(sequence)
                y.append(self.get_class_one_hot(int(sample[3])))
            yield np.array(X), np.array(y, dtype=float)


    @staticmethod
    def get_frames_for_sample(sample):
        """
        Given a sample row from the data file, get all the corrsponding frame
        filenames.
        """
        path = '.\\data\\' + sample[0] + '\\' + sample[1] + '\\'
        filename = sample[2]
        images = sorted(glob.glob(path + filename + '*jpg'))
        return images


    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split('\\')
        return parts[-1].replace('.jpg', '')


    @staticmethod
    def rescale_list(input_list, size):
        """
        Given a list and a size, return a rescaled/samples list. For example, if
        we want a list of size 5 and we have a list of size 25, return a new
        list of size which is every 5th element of the origina list.
        """

        if len(input_list) >= size:
            # Get the number to skip between iterations.
            skip = len(input_list) // size

            # Build our new output.
            output = [input_list[i] for i in range(0, len(input_list), skip)]
        else:
            temp = []
            copy = size // len(input_list)
            for a_input in input_list:
                temp.append([a_input] * copy)
            while len(temp) < size:
                temp.append([input_list[-1]])
            output = list(chain.from_iterable(temp))
        return output[:size]
