import tensorflow as tf


class MNCNN:

    def __init__(self):
        print("class loading complete")
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_size = 100

        self.W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        self.W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        self.W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))



        self.build_graph()


    def build_graph(self):
        L1 = tf.nn.conv2d(self.X, self.W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L1 = tf.nn.dropout(L1, self.keep_prob)

        L2 = tf.nn.conv2d(L1, self.W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.dropout(L2, self.keep_prob)

        L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
        L3 = tf.matmul(L3, self.W3)
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.dropout(L3, self.keep_prob)

        W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
        self.model = tf.matmul(L3, W4)