import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np

IS_TRAINING = True
NUM_EPOCHS = 10000
LEARNING_RATE = 0.001


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

    target_label = tf.cast(features['label'], tf.int64)
    input_feature = tf.cast(features['feature'], tf.float64)  # Input shape batch_size x 25

    # Drop user id content's
    input_feature = tf.concat([input_feature[:11], input_feature[12:]], axis=0)

    return input_feature, target_label


class RDWModel(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        if IS_TRAINING:
            self.dropout_prob = 0.5
            self.pos_fix = "train"
        else:
            self.dropout_prob = 1
            self.pos_fix = "test"
        with tf.name_scope("Config"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
            # Read dest embedding data
            self.destination_embedding = tf.Variable(
                tf.convert_to_tensor(np.load("../data/destinations.npy"), dtype=tf.float64), trainable=False,
                name="des_embedding")

        with tf.name_scope("Input" + self.pos_fix):
            if IS_TRAINING is True:
                feature, label = read_and_decode("../data/train-13.tfrecords")
                self.feature, self.label_batch = tf.train.shuffle_batch([feature, label], batch_size=128, num_threads=3,
                                                                        capacity=2000,
                                                                        min_after_dequeue=1000,
                                                                        allow_smaller_final_batch=True)
            else:
                feature, label = read_and_decode("../data/train-13.tfrecords")
                self.feature, self.label_batch = tf.train.batch([feature, label], batch_size=1, num_threads=3,
                                                                capacity=2000,
                                                                allow_smaller_final_batch=True)
        # Load test Data-set

        #
        # self.input = tf.placeholder(tf.float32, shape=[None, 24], name="user_input")
        # self.target_label = tf.placeholder(tf.float32, shape=[None, 1])
        with tf.name_scope("Des_Embedding"):
            des_embedding_feature = tf.nn.embedding_lookup(self.destination_embedding,
                                                           tf.cast(self.feature[:, 17], tf.int64))

            site_name = self.add_bucket_embedding(tf.cast(self.feature[:, 5], tf.int64), 1000, 8, "site_name")

            self.feature = tf.concat(
                [self.feature[:, :5], self.feature[:, 6:17], self.feature[:, 18:],
                 site_name,
                 des_embedding_feature],
                axis=1)

        with tf.name_scope("FC"):
            self.net = self.add_norm(self.feature, 24 - 1 + 149 - 1 + 8)
            self.net = self._add_fc_layer(self.net, 500, dropout=IS_TRAINING)
            self.net = self._add_fc_layer(self.net, 500, dropout=IS_TRAINING)
            self.net = self._add_fc_layer(self.net, 500, dropout=IS_TRAINING)
            self.net = self._add_fc_layer(self.net, 500, dropout=IS_TRAINING)
            self.net = self._add_fc_layer(self.net, 500, dropout=IS_TRAINING)

        with tf.name_scope("Output"):
            self.output = tc.layers.fully_connected(self.net, 100, activation_fn=None)

        with tf.name_scope("Batch_eval"):
            self.num_correct_prediction = tf.reduce_sum(
                tf.cast(tf.equal(self.label_batch, tf.argmax(self.output, 1)), tf.float32))

        if IS_TRAINING is False:
            return

        with tf.name_scope("Loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_batch, logits=self.output))
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("Train"):
            self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
            self.increase_step = self.global_step.assign_add(1)

    def _add_fc_layer(self, layer_input, size, activation_fn=tf.nn.relu, dropout=True, norm=True):
        output = tc.layers.fully_connected(layer_input, size, activation_fn=activation_fn)

        if norm:
            output = self.add_norm(output, size=size)
        if dropout is True:
            output = tf.nn.dropout(output, self.dropout_prob)
        return output

    @staticmethod
    def add_bucket_embedding(inputs, bucket_size, dim, name):
        with tf.variable_scope(name):
            embeddings = tf.Variable(
                tf.random_uniform([bucket_size, dim], -1.0, 1.0, dtype=tf.float64), dtype=tf.float64)
            mod_input = tf.mod(inputs, bucket_size)
            return tf.nn.embedding_lookup(embeddings, mod_input)

    @staticmethod
    def add_norm(layer_input, size):
        fc_mean, fc_var = tf.nn.moments(layer_input, axes=[0])
        scale = tf.Variable(tf.ones([size], dtype=tf.float64))
        shift = tf.Variable(tf.zeros([size], dtype=tf.float64))
        epsilon = 0.001

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = mean_var_with_update()

        layer_output = tf.nn.batch_normalization(layer_input, mean, var, shift, scale, epsilon)
        return layer_output

    def run_train(self, sess):
        step = 0
        try:
            while True:
                _, _, merged_summary, step_value, loss_value, net_output = sess.run(
                    [self.train_op, self.increase_step, merged, self.global_step, self.loss, self.output])
                writer.add_summary(merged_summary, global_step=step_value)

                if step % 100 == 0:
                    saver.save(sess, "model/v1/model.ckpt")
                    print ("Step %d: loss= %.4f" % (step, loss_value))
                step += len(net_output)
        except tf.errors.OutOfRangeError:
            print ("Done training for %d epochs, %d steps." % (NUM_EPOCHS, step))

    def run_evl(self, sess):
        step = 0
        correnct_entry = 0.0
        try:
            while True:
                net_output, target_output, num_correct = sess.run(
                    [self.output, self.label_batch, self.num_correct_prediction])
                test_out = np.argmax(net_output)
                correnct_entry += num_correct

                step += len(net_output)
        except tf.errors.OutOfRangeError:
            print ("Done training for %d epochs, %d steps, %2f accuracy ." % (NUM_EPOCHS, step, correnct_entry / step))


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
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            model.run_evl(session)
            coord.request_stop()
            coord.join(threads)

        session.close()
