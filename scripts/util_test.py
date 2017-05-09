import tensorflow as tf
import numpy as np

graound_des = np.load("../data/destinations.npy")
destination_embedding = tf.Variable(
    tf.convert_to_tensor(graound_des, dtype=tf.float64), trainable=False,
    name="des_embedding")

input_x = tf.placeholder(tf.int32, [None], "input")

embed = tf.nn.embedding_lookup(destination_embedding, input_x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    embed_value = sess.run(embed, feed_dict={input_x: [1]})
    print embed_value - graound_des[1]
    print np.shape(graound_des)
