import tensorflow as tf


def read_and_decode(filename):
    # Create queue
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # Filename
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.float32),
                                           'feature': tf.FixedLenFeature([25], tf.float32),
                                       })

    label = tf.cast(features['label'], tf.int32)
    return features['feature'], label


feature, label = read_and_decode("../data/train-13.tfrecords")
img_batch, label_batch = tf.train.shuffle_batch([feature, label], batch_size=125, num_threads=2, capacity=2000,
                                                min_after_dequeue=1000)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print "session start"
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print "start queue runners"
    for i in range(1):
        val, l = sess.run([img_batch, label_batch])
        print (val, l)
    coord.request_stop()
    coord.join(threads)
    sess.close()

