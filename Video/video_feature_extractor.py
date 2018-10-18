import csv
import glob
import os
import pathlib
from subprocess import call
from itertools import chain
from extractor import Extractor
from tqdm import tqdm
import numpy as np
import random


class VideoFeatureExtractor():
    def __init__(self, video_root_dir, database_name, seq_length, model_dir):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        pathlib.Path(os.path.join(self.dir_path, 'data', database_name)).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.dir_path, 'data', 'sequences')).mkdir(
            parents=True, exist_ok=True)
        self.root_dir = video_root_dir
        self.database = database_name
        self.seq_length = seq_length
        self.data_path = self.dir_path + '\\data'
        self.model_dir = model_dir
        self.dataset_infro = []
        self.extract_files
        self.extract_features

    @ property
    def extract_files(self):
        classes_videos_folders = glob.glob(self.root_dir + '\\' + '*')
        self.classes = self.get_classes(classes_videos_folders)
        for class_videos in classes_videos_folders:
            videos_per_class = glob.glob(class_videos + '\\*.avi')

            for video_path in videos_per_class:
                video_information = self.get_video_parts(video_path)
                databasename, classname, filename_no_ext, filename, \
                    _ = video_information
                pathlib.Path(os.path.join(self.data_path, databasename,
                    classname)).mkdir(parents=True, exist_ok=True)
                if not self.check_already_extracted(video_information):
                    src = self.root_dir + '\\' + classname + '\\' + \
        		        filename

                    dest = self.data_path + '\\' + databasename + '\\' + \
                        classname + '\\' + filename_no_ext + '-%04d.jpg'
                    call(["ffmpeg", "-i", src, dest])

                # Now get how many frames it is.
                nb_frames = self.get_nb_frames_for_video(video_information)
                self.dataset_infro.append([classname,
                                           filename_no_ext,
                                           nb_frames,
                                           self.get_class_index(classname)])
        random.shuffle(self.dataset_infro)
        with open(os.path.join(self.data_path,self.database +'.csv'),
                  'w',  newline='') as fout:
            writer = csv.writer(fout)
            writer.writerows(self.dataset_infro)

    def get_classes(self, classes_video_folders):
        classes = []
        for item in classes_video_folders:
            classes.append(item.split('\\')[-1])
        classes = sorted(classes)
        return classes

    def get_class_index(self, class_str):
        return self.classes.index(class_str)


    @ property
    def extract_features(self):
        # build model for feature extraction
        feature_extractor = Extractor(self.model_dir)
        # Loop through data.
        pbar = tqdm(total=len(self.dataset_infro))
        for each_video in self.dataset_infro:
            print(each_video)
            path = self.data_path + '\\sequences\\' + each_video[1] + \
                '-' + str(self.seq_length) + '-features.txt'

            if os.path.isfile(path):
                pbar.update(1)
                continue

            frames = self.get_frames_for_sample(each_video)
            frames = self.rescale_list(frames, self.seq_length)

            # Now loop through and extract features to build the sequence
            sequence = []
            for image in frames:
                features = feature_extractor.inception(image)
                sequence.append(features)
            npsquence = np.array(sequence)

            # save the sequence
            np.savetxt(path, sequence)

            pbar.update(1)
        pbar.close()

    def get_frames_for_sample(self, video_infro):
        """
        Given a sample row from the data file, get all the corrsponding frame
        filenames.
        """
        path = os.path.join(self.data_path, self.database,
                            video_infro[0], video_infro[1])
        images = sorted(glob.glob(path + '*jpg'))
        return images


    def get_video_parts(self, video_path):
    	"""Given a full path to a video, return its parts."""
    	parts = video_path.split('\\')
    	filename = parts[-1]
    	filename_no_ext = parts[-1].split('.')[0]
    	personname = filename_no_ext.split('_')[0]
    	classname = parts[-2]
    	databasename = self.database

    	return databasename, classname, filename_no_ext, filename, personname


    def check_already_extracted(self, video_parts):
    	"""Check to see if we created the -0001 frame of this file."""
    	databasename, classname, filename_no_ext, _, _ = video_parts
    	return bool(os.path.exists(self.data_path + databasename + '\\' +
            classname + '\\' + filename_no_ext + '-0001.jpg'))

    def get_nb_frames_for_video(self, video_parts):
    	"""Given video parts of an (assumed) already extracted video,
    	   return the number of frames that were extracted."""
    	databasename, classname, filename_no_ext, _, _ = video_parts
    	generated_files = glob.glob(self.data_path + '\\' + databasename + \
            '\\' + classname + '\\' + filename_no_ext + '*.jpg')
    	return len(generated_files)

    def rescale_list(self, input_list, size):
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

    def get_extracted_sequence(self, sample):
        filename = sample['video']
        path = self.data_path + '\\sequences\\' + filename + '-' + \
            str(self.seq_length) + '-' + 'features' + '.txt'
        if os.path.isfile(path):
            features = pd.read_csv(path, sep=" ", header=None)
            return features.values
        else:
            return None

def main():
    video_root_dir = 'C:\\Users\\Administrator\\Desktop\\cnn_lstm\\data\\HMDB'
    database_name = 'HMDB'
    seq_length = 50
    model_dir = 'C:\\Users\\Administrator\\Desktop\\inception-2015-12-05'
    VideoFeatureExtractor(video_root_dir, database_name,
                          seq_length, model_dir)

if __name__ == '__main__':
	main()
