import tensorflow as tf
import numpy as np


class MNIST_GAN:

    def __init__(self):
        self.batch_size = 100
        self.X = tf.placeholder(tf.float32, [None, 28 * 28])
        self.Z = tf.placeholder(tf.float32, [None, 128])

        self.G_W1 = tf.Variable(tf.random_normal([128, 256]))
        self.G_b1 = tf.Variable(tf.zeros([256]))
        self.G_W2 = tf.Variable(tf.random_normal([256, 28 * 28]))
        self.G_b2 = tf.Variable(tf.zeros([28 * 28]))

        self.D_W1 = tf.Variable(tf.random_normal([28 * 28, 256], stddev=0.01))
        self.D_b1 = tf.Variable(tf.zeros([256]))
        self.D_W2 = tf.Variable(tf.random_normal([256, 1], stddev=0.01))
        self.D_b2 = tf.Variable(tf.zeros([1]))

        self.build_graph()

    def make_noise(self, size):
        return np.random.normal(size=(size, 128))

    def generator(self, noise):
        L1 = tf.matmul(noise, self.G_W1) + self.G_b1
        L1 = tf.nn.relu(L1)

        L2 = tf.matmul(L1, self.G_W2) + self.G_b2
        result = tf.nn.sigmoid(L2)

        return result

    def discriminator(self, input):
        L1 = tf.matmul(input, self.D_W1) + self.D_b1
        L1 = tf.nn.relu(L1)

        L2 = tf.matmul(L1, self.D_W2) + self.D_b2
        result = tf.nn.sigmoid(L2)

        return result

    def build_graph(self):
        self.G = self.generator(self.Z)

        self.D_gen = self.discriminator(self.G)
        self.D_real = self.discriminator(self.X)
