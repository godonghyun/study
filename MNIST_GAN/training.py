import tensorflow as tf
import numpy as np
from model import MNIST_GAN
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


GAN = MNIST_GAN()

D_var_list = [GAN.D_W1, GAN.D_b1, GAN.D_W2, GAN.D_b2]
G_var_list = [GAN.G_W1, GAN.G_b1, GAN.G_W2, GAN.G_b2]

optimizer = tf.train.AdamOptimizer(0.0003)

loss_D = -(tf.reduce_mean(tf.log(GAN.D_real) + tf.log(1 - GAN.D_gen)))
loss_G = -(tf.reduce_mean(tf.log(GAN.D_gen)))

train_D = optimizer.minimize(loss_D, var_list=D_var_list)
train_G = optimizer.minimize(loss_G, var_list=G_var_list)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    total_batch = int(mnist.train.num_examples/batch_size)
    loss_val_D, loss_val_G = 0, 0

    for epoch in range(100):
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            noise = GAN.make_noise(batch_size)

            _, loss_val_D = sess.run([train_D, loss_D],
                                     feed_dict={GAN.X: batch_x, GAN.Z: noise})
            _, loss_val_G = sess.run([train_G, loss_G],
                                     feed_dict={GAN.Z: noise})

        print('Epoch:', '%04d' % epoch,
              'D loss: {:.4}'.format(loss_val_D),
              'G loss: {:.4}'.format(loss_val_G))

        if epoch == 0 or (epoch + 1) % 10 == 0:
            sample_size = 10
            noise = GAN.make_noise(sample_size)
            samples = sess.run(GAN.G, feed_dict={GAN.Z: noise})

            fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

            for i in range(sample_size):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(samples[i], (28, 28)))

            plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

print('complete')