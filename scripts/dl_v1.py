import tensorflow as tf
import tensorflow.contrib as tc

IS_TRAINING = True
NUM_EPOCHS = 1
LEARNING_RATE = 0.00001


def read_and_decode(filename):
    # Create queue
    filename_queue = tf.train.string_input_producer([filename], num_epochs=NUM_EPOCHS)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # Filename
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.float32),
                                           'feature': tf.FixedLenFeature([25], tf.float32),
                                       })

    target_label = tf.cast(features['label'], tf.int32)
    input_feature = features['feature']  # Input shape batch_size x 25

    # Drop user id content's
    input_feature = tf.concat([input_feature[:11], input_feature[12:]], axis=0)

    return input_feature, target_label


class RDWModel(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        with tf.name_scope("Input"):
            if IS_TRAINING is True:
                feature, label = read_and_decode("../data/train-14.tfrecords")
                self.feature, self.label_batch = tf.train.shuffle_batch([feature, label], batch_size=1, num_threads=3,
                                                                        capacity=2000,
                                                                        min_after_dequeue=1000,
                                                                        allow_smaller_final_batch=True)
            else:
                self.input = tf.placeholder(tf.float32, shape=[None, 24], name="user_input")
                self.target_label = tf.placeholder(tf.float32, shape=[None, 1])

        with tf.name_scope("FC"):
            self.net = self.feature
            self.net = tf.nn.dropout(tc.layers.fully_connected(self.net, 500), 0.5)
            self.net = tf.nn.dropout(tc.layers.fully_connected(self.net, 500), 0.5)
            self.net = tf.nn.dropout(tc.layers.fully_connected(self.net, 500), 0.5)
            self.net = tf.nn.dropout(tc.layers.fully_connected(self.net, 500), 0.5)

        with tf.name_scope("Output"):
            self.output = tc.layers.fully_connected(self.net, 100, activation_fn=None)

        with tf.name_scope("Loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_batch, logits=self.output))

        with tf.name_scope("Train"):
            self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    def run_train(self, sess):
        try:
            step = 0
            while True:
                _, loss_value = sess.run([self.train_op, self.loss])

                if step % 100 == 0:
                    print ("Step %d: loss= %.4f" % (step, loss_value))
                step += 1
        except tf.errors.OutOfRangeError:
            print ("Done training for %d epochs, %d steps." % (NUM_EPOCHS, step))

    def run_evl(self, out_file=True):
        pass


if __name__ == "__main__":
    # RDWModel.run_test()
    model = RDWModel()
    with tf.Session() as session:
        saver = tf.train.Saver()

        if tf.gfile.Exists("log/v1") is False:
            tf.gfile.MkDir("log/v1")
        if tf.gfile.Exists("model/v1") is False:
            tf.gfile.MkDir("model/v1")

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/v1", session.graph)
        ckpt = tf.train.get_checkpoint_state("model/v1")

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            print ("Restore ckpt")
        else:
            print ("No ckpt found")

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        if IS_TRAINING:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            model.run_train(session)
            coord.request_stop()
            coord.join(threads)
        else:
            model.run_evl()

        session.close()
