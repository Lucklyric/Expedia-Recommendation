import tensorflow as tf
import numpy as np

############## Embedding Test ##################
graound_des = np.load("../data/destinations.npy")
destination_embedding = tf.Variable(
    tf.convert_to_tensor(graound_des, dtype=tf.float64), trainable=False,
    name="des_embedding")
embeddings = tf.Variable(
    tf.random_uniform([2, 8], -1.0, 1.0))
input_x = tf.placeholder(tf.int32, [None, 2], "input")
hashed_input = tf.mod(input_x[:, 1], 2)
embed = tf.nn.embedding_lookup(embeddings, hashed_input)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    embed_value = sess.run(embed, feed_dict={input_x: [[0, 1], [0, 2], [0, 3]]})
    print embed_value[0]
    print embed_value[2]
    print np.shape(embed_value)


############## TF Records Test ###################
# def read_and_decode(filename):
#     # Create queue
#     filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)  # Filename
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.float32),
#                                            'feature': tf.FixedLenFeature([25], tf.float32),
#                                        })
#
#     target_label = tf.cast(features['label'], tf.int64)
#     input_feature = tf.cast(features['feature'], tf.float64)  # Input shape batch_size x 25
#
#     # Drop user id content's
#     # input_feature = tf.concat([input_feature[:11], input_feature[12:]], axis=0)
#
#     return input_feature, target_label
#
#
# feature, label = read_and_decode("../data/train-14-part.tfrecords")
# feature, label_batch = tf.train.batch([feature, label], batch_size=512, num_threads=1,
#                                       capacity=32,
#                                       allow_smaller_final_batch=False)
