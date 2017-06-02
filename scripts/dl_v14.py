# VERSION 2
# Add user id as feature
#
#
#
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import tensorflow.contrib.keras as keras
from BoostFlow import BoostFlow

TRAINING = 0
TESTING = 1
INFERENCE = 2
VERSION = "v14"
MODE = 0
NUM_EPOCHS = 1000000
# MODE = TESTING
# NUM_EPOCHS = 1
LEARNING_RATE = 0.01


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
    input_feature = features['feature']  # Input shape batch_size x 25

    # Drop user id content's
    # input_feature = tf.concat([input_feature[:11], input_feature[12:]], axis=0)

    return input_feature, target_label


class RDWModel(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        if MODE == TRAINING:
            self.dropout_prob = 0.66
            self.pos_fix = "train"
        else:
            self.dropout_prob = 1
            self.pos_fix = "test"
        with tf.name_scope("Config"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)
            # Read dest embedding data
            self.destination_embedding = tf.Variable(
                tf.convert_to_tensor(np.load("../data/destinations.npy"), dtype=tf.float32), trainable=False,
                name="des_embedding")

            # self.p_cluster = tf.Variable(tf.convert_to_tensor(np.load("../data/p_cluster.npy"), dtype=tf.float64),
            #                              trainable=False, name="p_cluster")
            # self.p_cluster = tf.reshape(self.p_cluster, [100])

        with tf.name_scope("Input" + self.pos_fix):
            self.learning_rate = tf.placeholder(tf.float32, name="LR")
            self.lowest_loss_value = tf.placeholder(tf.float32, [], name="LL")
            if MODE == TRAINING:
                self.lowest_loss = tf.Variable(1e4, name="lowest_loss")
                self.learning_rate_recorder = tf.Variable(LEARNING_RATE, dtype=tf.float32, name="LR_Recorder")
                feature, label = read_and_decode(["../data/train-13.tfrecords"],
                                                 num_epochs=NUM_EPOCHS)
                # self.feature, self.label_batch = tf.train.batch([feature, label], batch_size=512, num_threads=3,
                #                                                 capacity=1000 + 3 * 512,
                #                                                 allow_smaller_final_batch=True)

                self.feature, self.label_batch = tf.train.shuffle_batch([feature, label], batch_size=512, num_threads=3,
                                                                        capacity=1000 + 3 * 512,
                                                                        min_after_dequeue=1000,
                                                                        allow_smaller_final_batch=True)
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

        with tf.name_scope("Des_Embedding"):

            # Date Feature
            src_ci_month = self.add_bucket_embedding(tf.cast(self.feature[:, 0], tf.int64), 12, 8, "src_ci_month")
            src_ci_day = self.add_bucket_embedding(tf.cast(self.feature[:, 1], tf.int64), 31, 8, "src_ci_day")
            src_co_month = self.add_bucket_embedding(tf.cast(self.feature[:, 2], tf.int64), 12, 8, "src_co_month")
            src_co_day = self.add_bucket_embedding(tf.cast(self.feature[:, 3], tf.int64), 31, 8, "src_co_day")

            # Source
            is_mobile = self.add_bucket_embedding(tf.cast(self.feature[:, 12], tf.int64), 2, 8, "is_mobile")
            is_package = self.add_bucket_embedding(tf.cast(self.feature[:, 13], tf.int64), 2, 8, "is_package")
            channel = self.add_bucket_embedding(tf.cast(self.feature[:, 14], tf.int64), 10000, 8, "channel")
            site_name = self.add_bucket_embedding(tf.cast(self.feature[:, 5], tf.int64), 1000, 8, "site_name")
            posa_continent = self.add_bucket_embedding(tf.cast(self.feature[:, 6], tf.int64), 100, 8, "posa_continent")

            # booking type
            booking_type = self.add_bucket_embedding(tf.cast(self.feature[:, 20], tf.int64), 2, 8, "booking_type")
            # booking_type = self.add_norm(booking_type)

            # user location city
            u_loc_city = self.add_bucket_embedding(tf.cast(self.feature[:, 9], tf.int64), 100000, 8, "u_loc_city")
            # u_loc_city = self.add_norm(u_loc_city)

            # orig destination distance
            orig_destination_distance = self.feature[:, 10:11]
            orig_destination_distance = self.add_fc_stack_layers(orig_destination_distance, [8, 8])
            # orig_destination_distance = self.add_norm(orig_destination_distance)

            # orig destination
            des_embedding_feature = tf.nn.embedding_lookup(self.destination_embedding,
                                                           tf.cast(self.feature[:, 18], tf.int64))

            des_embedding_feature = self.add_norm(des_embedding_feature)
            des_embedding_feature = self.add_fc_stack_layers(des_embedding_feature, [128, 64, 8])

            h_contry = self.add_bucket_embedding(tf.cast(self.feature[:, 23], tf.int64), 1000, 8, "h_contry")

            h_market = self.add_bucket_embedding(tf.cast(self.feature[:, 24], tf.int64), 100000, 8, "h_market")

            # user id
            user_id = self.add_bucket_embedding(tf.cast(self.feature[:, 11], tf.int64), 1200000, 8, "user_id")

            tran_month = self.add_bucket_embedding(tf.cast(self.feature[:, 4], tf.int64), 12, 8, "trans_month")

            self.stack_features = tf.concat(
                [src_ci_month, src_ci_day, src_co_month, src_co_day, is_mobile, is_package, channel, site_name,
                 posa_continent, booking_type, u_loc_city, des_embedding_feature,
                 des_embedding_feature, h_market, h_contry, user_id, tran_month, orig_destination_distance],
                axis=1)  # [batch_size, 17*8]

            self.BF = BoostFlow(self.stack_features, MODE, [128, 256], 4, self.learning_rate, [256, 256],
                                self.label_batch, self.dropout_prob)

        if MODE == TRAINING:
            self.update_lr = tf.assign(self.learning_rate_recorder, self.learning_rate)
            self.update_ll = tf.assign(self.lowest_loss, self.lowest_loss_value)

        if MODE == INFERENCE:
            return
        with tf.name_scope("Batch_eval"):
            self.num_correct_prediction = tf.reduce_sum(
                tf.cast(tf.equal(self.label_batch, tf.argmax(self.BF.final_output, 1)), tf.float32))
            self.mAP, self.mAP_update = tc.metrics.streaming_sparse_average_precision_at_k(self.BF.final_output,
                                                                                           self.label_batch, 5)
        if MODE == TESTING:
            return

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
                tf.random_uniform([bucket_size, dim], -1.0, 1.0, dtype=tf.float32), dtype=tf.float32)
            mod_input = tf.mod(inputs, bucket_size)
            return tf.nn.embedding_lookup(embeddings, mod_input)

    @staticmethod
    def add_norm(layer_input):
        size = layer_input.get_shape()[-1]
        scale = tf.Variable(tf.ones([size], dtype=tf.float32))
        shift = tf.Variable(tf.zeros([size], dtype=tf.float32))
        pop_mean = tf.Variable(tf.zeros([layer_input.get_shape()[-1]], dtype=tf.float32), trainable=False)
        pop_var = tf.Variable(tf.ones([layer_input.get_shape()[-1]], dtype=tf.float32), trainable=False)
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
        previous_losses = []
        learning_rate_value = self.learning_rate_recorder.eval(sess)
        # learning_rate_value = 0.001
        try:
            while not coord.should_stop():
                _, _, _, _, _, _, fusion_loss_value, merged_summary, step_value, ll_value = sess.run(
                    self.BF.m_train_ops + [self.BF.train_op, self.update_lr, self.BF.final_loss, merged,
                                           self.global_step, self.lowest_loss
                                           ], feed_dict={
                        self.learning_rate: learning_rate_value
                    })
                if ll_value > fusion_loss_value:
                    sess.run(self.update_ll, feed_dict={
                        self.lowest_loss_value: fusion_loss_value
                    })
                    saver.save(sess, "model/" + VERSION + "/best/model.ckpt")
                    print ("ll value model saved")
                if step % 1000 == 0:
                    if len(previous_losses) > 8.0 and fusion_loss_value > max(
                            previous_losses[-9:]) and learning_rate_value > 0.00001:
                        print ("decay learning rate!!!")
                        learning_rate_value *= 0.5
                previous_losses.append(fusion_loss_value)
                writer.add_summary(merged_summary, global_step=step_value)
                step = step_value
                if step % 300 == 0:
                    saver.save(sess, "model/" + VERSION + "/model.ckpt")
                if step % 50 == 0:
                    print("Step: %d, Loss:%.4f" % (step_value, fusion_loss_value))
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
                    [self.mAP, self.mAP_update, self.BF.final_output, self.feature, self.label_batch,
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
                print num_correct
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
                net_output = sess.run(self.BF.final_output)
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
            tf.gfile.MkDir("model/" + VERSION + "/best/")

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/" + VERSION, session.graph)
        # ckpt = tf.train.get_checkpoint_state("model/" + VERSION+"/best")
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
