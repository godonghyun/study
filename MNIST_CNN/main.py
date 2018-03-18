import numpy as np
import tensorflow as tf
from model import MNCNN


""" get mnist data set from tensorflow """
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

images = mnist.test.images.reshape(-1, 28, 28, 1)
labels = mnist.test.labels

cnn = MNCNN()

saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./model')

with tf.Session() as sess:

    saver.restore(sess, ckpt.model_checkpoint_path)

    result = sess.run(cnn.model,feed_dict={cnn.X: images[:1],
                                           cnn.keep_prob: 1})
    result = sess.run(tf.argmax(result, 1))

print(result)