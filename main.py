from __future__ import print_function
import os
import PIL.Image
import numpy as np
from io import BytesIO
from functools import partial
from IPython.display import clear_output, Image, display, HTML

import tensorflow as tf

# TensorFlow Models
model_fn = './models/tensorflow_inception_graph.pb'

# TensorFlow session and model loading
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Input tensor
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

# Convutional layers patterns
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

# Print layer infos
print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))

# TensorFlow graph visualization


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()

    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)

        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)

            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)

    return strip_def


def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()

    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)

        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0] != '^' else '^' + rename_func(s[1:])

    return res_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'graph_def'):
        graph_def = graph_def.as_graph_def()

    strip_def = strip_consts(graph_def, max_const_size=max_const_size)

    code = """"
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))

    display(HTML(iframe))


# Network graph visualization
tmp_def = rename_nodes(graph_def, lambda s: "/".join(s.split('_', 1)))
show_graph(tmp_def)


# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
img_noise = np.random.uniform(size=(244, 244, 3)) + 100.0


def showarray(a, fmt='jpeg'):
    a = np.unit8(np.clip(a, 0, 1) * 255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def visstd(a, s=0.1):
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5


def T(layer):
    return graph.get_tensor_by_name("import/%s:0"%layer)

def render_native(t_obj, img0=img_noise, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input:img})
        g /= g.std() + 1e-8
        img += g*step
        print(score, end=' ')
    clear_output()
    showarray(visstd(img))

render_native(T(layer)[:, :, :, channel])