import tensorflow as tf
from model import MNCNN

""" get mnist data set from tensorflow """
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


CNN = MNCNN()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=CNN.model, labels=CNN.Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


with tf.Session() as sess:

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('restoring complete')
    else:
        sess.run(tf.global_variables_initializer())

    total_batch = int(mnist.train.num_examples / CNN.batch_size)

    for epoch in range(1000):
        total_cost = 0

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(CNN.batch_size)
            batch_x = batch_x.reshape(-1, 28, 28, 1)

            _, c = sess.run([optimizer, cost],
                                feed_dict={CNN.X: batch_x,
                                           CNN.Y: batch_y,
                                           CNN.keep_prob: 0.7})
            total_cost += c

        print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.3f}'.format(total_cost))

        if (total_cost < 5):
            break

    print('training complete')

    is_correct = tf.equal(tf.argmax(CNN.model, 1), tf.argmax(CNN.Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도:', sess.run(accuracy,
                        feed_dict={CNN.X: mnist.test.images.reshape(-1, 28, 28, 1),
                                   CNN.Y: mnist.test.labels,
                                   CNN.keep_prob: 1}))

    saver.save(sess, './model/cnn.ckpt')
    print('saving complete')