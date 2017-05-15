# VERSION 2
# Add user id as feature
#
#
#
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import tensorflow.contrib.keras as keras

# import tensorflow.contrib as slim

VERSION = "v2"
IS_TRAINING = True
NUM_EPOCHS = 1000000
# IS_TRAINING = False
# NUM_EPOCHS = 1
LEARNING_RATE = 0.01


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
    # input_feature = tf.concat([input_feature[:11], input_feature[12:]], axis=0)

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
                self.feature, self.label_batch = tf.train.shuffle_batch([feature, label], batch_size=256, num_threads=3,
                                                                        capacity=2000,
                                                                        min_after_dequeue=1000,
                                                                        allow_smaller_final_batch=False)
            else:
                feature, label = read_and_decode("../data/train-14.tfrecords")
                self.feature, self.label_batch = tf.train.batch([feature, label], batch_size=512, num_threads=3,
                                                                capacity=2000,
                                                                allow_smaller_final_batch=True)
        # Load test Data-set

        #
        # self.input = tf.placeholder(tf.float32, shape=[None, 24], name="user_input")
        # self.target_label = tf.placeholder(tf.float32, shape=[None, 1])
        with tf.name_scope("Des_Embedding"):

            # Time duriation
            src_ci_month = self.add_bucket_embedding(tf.cast(self.feature[:, 0], tf.int64), 12, 8, "src_ci_month")
            src_ci_day = self.add_bucket_embedding(tf.cast(self.feature[:, 1], tf.int64), 31, 8, "src_ci_day")
            src_co_month = self.add_bucket_embedding(tf.cast(self.feature[:, 2], tf.int64), 12, 8, "src_co_month")
            src_co_day = self.add_bucket_embedding(tf.cast(self.feature[:, 3], tf.int64), 31, 8, "src_co_day")
            self.time_feature = tf.concat([src_ci_month, src_ci_day, src_co_day, src_co_month], axis=1)
            self.time_feature = self.add_norm(self.time_feature, 4 * 8)
            self.time_feature = self.add_fc_stack_layers(self.time_feature, [64, 128, 256, 128])

            # Source
            is_mobile = self.add_bucket_embedding(tf.cast(self.feature[:, 12], tf.int64), 2, 8, "is_mobile")
            is_package = self.add_bucket_embedding(tf.cast(self.feature[:, 13], tf.int64), 2, 8, "is_package")
            channel = self.add_bucket_embedding(tf.cast(self.feature[:, 14], tf.int64), 10000, 8, "channel")
            site_name = self.add_bucket_embedding(tf.cast(self.feature[:, 5], tf.int64), 1000, 8, "site_name")
            posa_continent = self.add_bucket_embedding(tf.cast(self.feature[:, 6], tf.int64), 100, 8, "posa_continent")
            self.source_feature = tf.concat([is_mobile, is_package, channel, site_name, posa_continent], axis=1)
            self.source_feature = self.add_norm(self.source_feature, 5 * 8)
            self.source_feature = self.add_fc_stack_layers(self.source_feature, [128, 256, 256, 128])

            # Destination
            des_embedding_feature = tf.nn.embedding_lookup(self.destination_embedding,
                                                           tf.cast(self.feature[:, 18], tf.int64))
            des_type_id = self.add_bucket_embedding(tf.cast(self.feature[:, 19], tf.int64), 100000, 8, "des_type_id")

            # Hotel info
            h_continent = self.add_bucket_embedding(tf.cast(self.feature[:, 22], tf.int64), 100, 8, "h_continent")
            h_contry = self.add_bucket_embedding(tf.cast(self.feature[:, 23], tf.int64), 1000, 8, "h_contry")
            h_market = self.add_bucket_embedding(tf.cast(self.feature[:, 24], tf.int64), 100000, 8, "h_market")
            self.des_feature = tf.concat([des_embedding_feature, des_type_id, h_market, h_contry, h_continent], axis=1)
            self.des_feature = self.add_norm(self.des_feature, 4 * 8 + 149)
            self.des_feature = self.add_fc_stack_layers(self.des_feature, [256, 512, 512, 256])

            # User info
            u_loc_contry = self.add_bucket_embedding(tf.cast(self.feature[:, 7], tf.int64), 1000, 8, "u_loc_contry")
            u_loc_region = self.add_bucket_embedding(tf.cast(self.feature[:, 8], tf.int64), 100000, 8, "u_loc_region")
            u_loc_city = self.add_bucket_embedding(tf.cast(self.feature[:, 9], tf.int64), 100000, 8, "u_loc_city")
            self.user_feature = tf.concat([u_loc_city, u_loc_region, u_loc_contry, self.feature[:, 10:11]], axis=1)
            self.user_feature = self.add_norm(self.user_feature, 3 * 8 + 1)
            self.user_feature = self.add_fc_stack_layers(self.user_feature, [64, 128, 128])

            # Query Requirements
            self.query_feature = tf.concat([self.feature[:, 15:18]], axis=1)
            self.query_feature = self.add_norm(self.query_feature, 3)
            self.query_feature = self.add_fc_stack_layers(self.query_feature, [64, 128, 256, 128])

            # other feature
            tran_month = self.add_bucket_embedding(tf.cast(self.feature[:, 4], tf.int64), 12, 8, "trans_month")
            booking = self.add_bucket_embedding(tf.cast(self.feature[:, 20], tf.int64), 2, 8, "is_booking")
            self.other_feature = tf.concat([tran_month, booking], axis=1)
            self.other_feature = self.add_norm(self.other_feature, 16)
            self.other_feature = self.add_fc_stack_layers(self.other_feature, [64, 128, 128])

            # user id
            user_id = self.add_bucket_embedding(tf.cast(self.feature[:, 11], tf.int64), 100000, 8, "user_id")
            self.user_id_feature = self.add_norm(user_id, 8)

            self.stack_features = tf.concat([self.time_feature,
                                             self.source_feature,
                                             self.des_feature,
                                             self.user_feature,
                                             self.query_feature,
                                             self.other_feature,
                                             self.user_id_feature], axis=1)

        with tf.name_scope("FC"):
            self.net = self.add_fc_stack_layers(self.stack_features, [1024])
            self.net = self.add_fc_stack_layers(self.stack_features, [1024]) + self.net
            self.net = self.add_fc_stack_layers(self.stack_features, [512])
            self.net = self.add_fc_stack_layers(self.stack_features, [512]) + self.net
            self.net = self.add_fc_stack_layers(self.stack_features, [256])
        with tf.name_scope("Output"):
            self.output = tc.layers.fully_connected(self.net, 100, activation_fn=None)

        with tf.name_scope("Batch_eval"):
            self.num_correct_prediction = tf.reduce_sum(
                tf.cast(tf.equal(self.label_batch, tf.argmax(self.output, 1)), tf.float32))
            self.mAP, self.mAP_update = tc.metrics.streaming_sparse_average_precision_at_k(self.output,
                                                                                           self.label_batch, 5)

        if IS_TRAINING is False:
            return

        with tf.name_scope("Loss"):
            self.label_vector = tf.one_hot(self.label_batch,100)
            # self.s_output = tf.nn.softmax(self.output)
            self.loss = tf.reduce_mean(
                keras.backend.sparse_categorical_crossentropy(output=self.output, target=self.label_batch, from_logits=True))
            # self.loss = tf.reduce_mean(
            #     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_batch, logits=self.output))
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("Train"):
            self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
            self.increase_step = self.global_step.assign_add(1)

    def _add_fc_layer(self, layer_input, size, activation_fn=tf.nn.relu, dropout=True, norm=True):
        output = tc.layers.fully_connected(layer_input, size, activation_fn=activation_fn)

        if norm:
            output = self.add_norm(output, size=size)
        # if dropout is True:
        #     output = tf.nn.dropout(output, self.dropout_prob)
        return output

    def add_fc_stack_layers(self, inputs, layer_configure, norm=True):
        out = inputs
        for size in layer_configure:
            out = self._add_fc_layer(out, size, dropout=IS_TRAINING, norm=norm)
        return out

    @staticmethod
    def add_bucket_embedding(inputs, bucket_size, dim, name):
        with tf.variable_scope(name):
            embeddings = tf.Variable(
                tf.random_uniform([bucket_size, dim], -1.0, 1.0, dtype=tf.float64), dtype=tf.float64)
            mod_input = tf.mod(inputs, bucket_size)
            return tf.nn.embedding_lookup(embeddings, mod_input)

    @staticmethod
    def add_norm(layer_input, size):
        scale = tf.Variable(tf.ones([size], dtype=tf.float64))
        shift = tf.Variable(tf.zeros([size], dtype=tf.float64))
        pop_mean = tf.Variable(tf.zeros([layer_input.get_shape()[-1]], dtype=tf.float64), trainable=False)
        pop_var = tf.Variable(tf.ones([layer_input.get_shape()[-1]], dtype=tf.float64), trainable=False)
        epsilon = 0.001
        if IS_TRAINING:
            # batch_mean, batch_var = tf.nn.moments(layer_input, axes=[0])
            fc_mean, fc_var = tf.nn.moments(layer_input, axes=[0])

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op, tf.assign(pop_var, fc_var), tf.assign(pop_mean, fc_mean)]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
            layer_output = tf.nn.batch_normalization(layer_input, mean, var, shift, scale, epsilon)
            # decay = 0.5
            # train_mean = tf.assign(pop_mean,
            #                        pop_mean * decay + batch_mean * (1 - decay))
            # train_var = tf.assign(pop_var,
            #                       pop_var * decay + batch_var * (1 - decay))
            # with tf.control_dependencies([train_mean, train_var]):
            #     return tf.nn.batch_normalization(layer_input,
            #                                      pop_mean, pop_var, shift, scale, epsilon)
        else:
            layer_output = tf.nn.batch_normalization(layer_input, pop_mean, pop_var, shift, scale, epsilon)
        return layer_output

    def run_train(self, sess):
        step = 0
        try:
            while True:
                _, _, merged_summary, step_value, loss_value, net_output = sess.run(
                    [self.train_op, self.increase_step, merged, self.global_step, self.loss, self.output])
                writer.add_summary(merged_summary, global_step=step_value)

                if step % 100 == 0:
                    saver.save(sess, "model/" + VERSION + "/model.ckpt")
                    print ("Step %d: loss= %.4f" % (step, loss_value))
                step += len(net_output)
        except tf.errors.OutOfRangeError:
            print ("Done training for %d epochs, %d steps." % (NUM_EPOCHS, step))

    def run_evl(self, sess):
        step = 0
        correnct_entry = 0.0
        try:
            while True:
                # sess.run(self.mAP_update)
                mAP, _, net_output, feature_value, target_label, num_correct = sess.run(
                    [self.mAP, self.mAP_update, self.output, self.feature, self.label_batch,
                     self.num_correct_prediction])
                test_out = np.argmax(net_output, axis=1)
                correnct_entry += num_correct
                # print test_out
                # print feature_value
                # print target_label
                # print correnct_entry
                # net_output, mAP, _, _, _ = sess.run(
                #     [self.output, self.mAP, self.mAP_update, self.output, self.num_correct_prediction])
                # print mAP
                step += len(net_output)
                print step
                print mAP
        except tf.errors.OutOfRangeError:
            print ("Done training for %d epochs, %d steps, %f mAP@5 %f accuracy ." % (
                NUM_EPOCHS, step, mAP, correnct_entry / step))
            # print ("Done training for %d epochs, %d steps %f mAP" % (
            #     NUM_EPOCHS, step, mAP))


if __name__ == "__main__":
    # RDWModel.run_test()
    model = RDWModel()
    with tf.Session() as session:
        keras.backend.set_session(session)
        saver = tf.train.Saver()

        if tf.gfile.Exists("log/" + VERSION) is False:
            tf.gfile.MkDir("log/" + VERSION)
        if tf.gfile.Exists("model/" + VERSION) is False:
            tf.gfile.MkDir("model/" + VERSION)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/" + VERSION, session.graph)
        ckpt = tf.train.get_checkpoint_state("model/" + VERSION)

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            print ("Restore ckpt")
        else:
            print ("No ckpt found")

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
