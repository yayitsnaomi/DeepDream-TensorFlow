import numpy as np
import tensorflow as tf
import os

data_url = "http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
data_dir = "inception/5h/"
path_graph_def = "tensorflow_inception_graph.pb"


class Inception5h:

    tensor_name_input_image = "input:0"

    layer_names = ['conv2d0', 'conv2d1', 'conv2d2',
                   'mixed3a', 'mixed3b',
                   'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e',
                   'mixed5a', 'mixed5b']

    def __init__(self):

        self.graph = tf.Graph()

        with self.graph.as_default():
            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name='')

            self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)
            self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]

    def create_feed_dict(self, image=None):
        image = np.expand_dims(image, axis=0)
        feed_dict = {self.tensor_name_input_image: image}
        return feed_dict

    def get_gradient(self, tensor):

        with self.graph.as_default():
            tensor = tf.square(tensor)
            tensor_mean = tf.reduce_mean(tensor)
            gradient = tf.gradients(tensor_mean, self.input)[0]

        return gradient