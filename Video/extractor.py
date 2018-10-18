from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os


class Extractor(object):
    def __init__(self, model_dir):
        self._model_dir = model_dir
        inception_proto_file = os.path.join(self._model_dir,
                                            'classify_image_graph_def.pb')
        self._load_inception(inception_proto_file)

    def _load_inception(self, proto_file):
        graph_def = tf.GraphDef.FromString(open(proto_file, 'rb').read())
        self._inception_graph = tf.Graph()
        with self._inception_graph.as_default():
                _ = tf.import_graph_def(graph_def, name='')
                self.session = tf.Session()

    def inception(self, frame_rgb):
        frame_rgb_data = tf.gfile.FastGFile(frame_rgb, 'rb').read()
        with self._inception_graph.as_default():
            frame_features = self.session.run('pool_3/_reshape:0',
                                              feed_dict={'DecodeJpeg/contents:0':
                                                         frame_rgb_data})
            frame_features = frame_features[0]
        return frame_features
