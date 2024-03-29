# VERSION 2
# Add user id as feature
#
#
#
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import tensorflow.contrib.keras as keras

TRAINING = 0
TESTING = 1
INFERENCE = 2
VERSION = "v8"
MODE = 0
NUM_EPOCHS = 1000000
# MODE = TESTING
# NUM_EPOCHS = 1
LEARNING_RATE = 0.000001


def read_and_decode(filename, num_epochs=1):
    # Create queue
    filename_queue = tf.train.string_input_producer(filename, num_epochs=num_epochs)
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
        if MODE == TRAINING:
            self.dropout_prob = 0.5
            self.pos_fix = "train"
        else:
            self.dropout_prob = 1
            self.pos_fix = "test"
        with tf.name_scope("Config"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)

        with tf.name_scope("Input" + self.pos_fix):
            self.learning_rate = tf.placeholder(tf.float64, name="LR")
            if MODE == TRAINING:
                feature, label = read_and_decode(["../data/train-13-all-book-type.tfrecords"], num_epochs=NUM_EPOCHS)
                self.feature, self.label_batch = tf.train.shuffle_batch([feature, label], batch_size=128, num_threads=3,
                                                                        capacity=2000,
                                                                        min_after_dequeue=1000,
                                                                        allow_smaller_final_batch=False)
            elif MODE == TESTING:
                feature, label = read_and_decode(["../data/train-14.tfrecords"])
                self.feature, self.label_batch = tf.train.batch([feature, label], batch_size=512, num_threads=3,
                                                                capacity=2000,
                                                                allow_smaller_final_batch=True)
            elif MODE == INFERENCE:
                feature, label = read_and_decode(["../data/evl.tfrecords"])
                self.feature, _ = tf.train.batch([feature, label], batch_size=1024, num_threads=1,
                                                 capacity=2000,
                                                 allow_smaller_final_batch=True)

        with tf.name_scope("FC"):
            self.net = self.add_norm(self.feature)
            self.net = self._add_fc_layer(self.net, 500, dropout=(MODE == TRAINING))
            self.net = self._add_fc_layer(self.net, 500, dropout=(MODE == TRAINING))
            self.net = self._add_fc_layer(self.net, 500, dropout=(MODE == TRAINING))
            self.net = self._add_fc_layer(self.net, 500, dropout=(MODE == TRAINING))
            self.net = self._add_fc_layer(self.net, 500, dropout=(MODE == TRAINING))

        with tf.name_scope("Output"):
            self.output = tc.layers.fully_connected(self.net, 100, activation_fn=None)

        if MODE == INFERENCE:
            return
        with tf.name_scope("Batch_eval"):
            self.num_correct_prediction = tf.reduce_sum(
                tf.cast(tf.equal(self.label_batch, tf.argmax(self.output, 1)), tf.float32))
            self.mAP, self.mAP_update = tc.metrics.streaming_sparse_average_precision_at_k(self.output,
                                                                                           self.label_batch, 5)
        if MODE == TESTING:
            return

        with tf.name_scope("Loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_batch, logits=self.output))

            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("Train"):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self.increase_step = self.global_step.assign_add(1)

    def _add_fc_layer(self, layer_input, size, activation_fn=tf.nn.relu, dropout=True, norm=True):
        output = tc.layers.fully_connected(layer_input, size, activation_fn=activation_fn)

        if norm:
            output = self.add_norm(output)
        if dropout is True:
            output = tf.nn.dropout(output, self.dropout_prob)
        return output

    def add_fc_stack_layers(self, inputs, layer_configure, norm=True):
        out = inputs
        for size in layer_configure:
            out = self._add_fc_layer(out, size, dropout=(MODE == TRAINING), norm=norm)
        return out

    @staticmethod
    def add_bucket_embedding(inputs, bucket_size, dim, name):
        with tf.variable_scope(name):
            embeddings = tf.Variable(
                tf.random_uniform([bucket_size, dim], -1.0, 1.0, dtype=tf.float64), dtype=tf.float64)
            mod_input = tf.mod(inputs, bucket_size)
            return tf.nn.embedding_lookup(embeddings, mod_input)

    @staticmethod
    def add_norm(layer_input):
        size = layer_input.get_shape()[-1]
        scale = tf.Variable(tf.ones([size], dtype=tf.float64))
        shift = tf.Variable(tf.zeros([size], dtype=tf.float64))
        pop_mean = tf.Variable(tf.zeros([layer_input.get_shape()[-1]], dtype=tf.float64), trainable=False)
        pop_var = tf.Variable(tf.ones([layer_input.get_shape()[-1]], dtype=tf.float64), trainable=False)
        epsilon = 0.001
        if MODE == TRAINING:
            fc_mean, fc_var = tf.nn.moments(layer_input, axes=[0])

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op, tf.assign(pop_var, fc_var), tf.assign(pop_mean, fc_mean)]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
            layer_output = tf.nn.batch_normalization(layer_input, mean, var, shift, scale, epsilon)
        else:
            layer_output = tf.nn.batch_normalization(layer_input, pop_mean, pop_var, shift, scale, epsilon)
        return layer_output

    def run_train(self, sess):
        step = 0
        try:
            while not coord.should_stop():
                _, _, merged_summary, step_value, loss_value, net_output = sess.run(
                    [self.train_op, self.increase_step, merged, self.global_step, self.loss, self.output], feed_dict={
                        self.learning_rate: LEARNING_RATE
                    })
                writer.add_summary(merged_summary, global_step=step_value)
                step = step_value
                if step_value % 100 == 0:
                    saver.save(sess, "model/" + VERSION + "/model.ckpt")
                    print ("Step %d: loss= %.4f" % (step_value, loss_value))

        except tf.errors.OutOfRangeError:
            print ("Done training for %d epochs, %d steps." % (NUM_EPOCHS, step))
        finally:
            coord.request_stop()

    def run_evl(self, sess):
        step = 0
        correnct_entry = 0.0
        try:
            while not coord.should_stop():
                # sess.run(self.mAP_update)
                mAP, _, net_output, feature_value, target_label, num_correct = sess.run(
                    [self.mAP, self.mAP_update, self.output, self.feature, self.label_batch,
                     self.num_correct_prediction])
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
        finally:
            coord.request_stop()

    def run_inference(self, sess):
        step = 0
        o_file = open("output_" + VERSION + ".csv", "w")
        o_file.write("id,hotel_cluster\n")
        try:
            while not coord.should_stop():
                net_output = sess.run(self.output)
                for row in net_output:
                    top_pred = row.argsort()[-5:]
                    write_p = " ".join([str(l) for l in top_pred])
                    write_frame = "{0},{1}".format(step, write_p)
                    o_file.write(write_frame + "\n")

                    step += 1
                print step
        except tf.errors.OutOfRangeError:
            print ("Done for inferencefro %d epochs, %d steps." % (1, step))
        finally:
            o_file.close()
            coord.request_stop()


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

        if MODE == TRAINING:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            model.run_train(session)
            coord.join(threads)
        elif MODE == TESTING:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            model.run_evl(session)
            coord.join(threads)
        elif MODE == INFERENCE:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            model.run_inference(session)
            coord.join(threads)

        session.close()
