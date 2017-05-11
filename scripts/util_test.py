import tensorflow as tf
import numpy as np

graound_des = np.load("../data/destinations.npy")
destination_embedding = tf.Variable(
    tf.convert_to_tensor(graound_des, dtype=tf.float64), trainable=False,
    name="des_embedding")
embeddings = tf.Variable(
    tf.random_uniform([2, 8], -1.0, 1.0))
input_x = tf.placeholder(tf.int32, [None], "input")
hashed_input = tf.mod(input_x, 2)
embed = tf.nn.embedding_lookup(embeddings, hashed_input)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    embed_value = sess.run(embed, feed_dict={input_x: [1, 2, 3]})
    print embed_value[0]
    print embed_value[2]
    print np.shape(embed_value)
