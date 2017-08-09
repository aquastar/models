import numpy as np
import tensorflow as tf

ph = tf.placeholder(shape=[None, 3], dtype=tf.int32)

# look the -1 in the first position
x = tf.slice(ph, [1, 1], [2, 2])

input_ = np.array([[1, 2, 3],
                   [3, 4, 5],
                   [5, 6, 7]])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(x, feed_dict={ph: input_}))
