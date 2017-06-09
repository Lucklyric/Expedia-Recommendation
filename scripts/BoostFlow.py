import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.keras as keras


class BoostFlow(object):
    def __init__(self, inputs, mode, general_layer_config, num_M, learning_rate, between_m_config, target_output,
                 keep_prob):
        self.inputs = inputs
        self.mode = mode
        self.keep_prob = keep_prob
        self.general_layer_config = general_layer_config
        self.num_M = num_M
        self.between_m_config = between_m_config
        self.target_output = target_output
        self.learning_rate = learning_rate

        self._build_graph()

    def _build_graph(self):
        """
        Build model graph
        :return:
        """
        # weights
        with tf.variable_scope("m_weights"):
            self.m_weights = tf.nn.softmax(
                [tf.Variable(tf.ones([]), name="weight_m_" + str(m), trainable=False) for m in xrange(self.num_M)])

        self.net = self.inputs
        # general layer config
        with tf.name_scope("BF_general_layer"):
            self.net = self.add_fc_stack_layers(self.net, self.general_layer_config, is_training=(self.mode == 0))

        self.one_hot_target = tf.one_hot(self.target_output, 100, dtype=tf.float32)
        self.m_targets = []
        self.m_targets.append(self.one_hot_target)

        self.m_losses = []

        self.m_train_ops = []

        self.m_outputs = []

        # for-loop M
        for m in xrange(self.num_M):
            with tf.name_scope("BF_m" + str(m) + "_part"):
                self.net = self.add_fc_stack_layers(self.net, self.between_m_config, is_training=(self.mode == 0))
                m_output = self._add_fc_layer(self.net, 100, activation_fn=None, batch_norm=True,
                                              is_training=(self.mode == 0))
                self.m_outputs.append(m_output)

                gap = self.m_targets[-1] - m_output
                m_loss = tf.reduce_mean(tf.reduce_sum(tf.square(gap), axis=-1), axis=-1)
                self.m_losses.append(m_loss)

                m_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.m_losses[-1])
                self.m_train_ops.append(m_train_op)

                self.m_targets.append(gap)

                tf.summary.scalar(str(m) + '_sub_loss', m_loss)

        self.final_output = self.m_outputs[0] * self.m_weights[0]
        tf.summary.scalar('0_weight', self.m_weights[0])
        for m in xrange(self.num_M - 1):
            self.final_output += self.m_outputs[m + 1] * self.m_weights[m + 1]
            tf.summary.scalar(str(m + 1) + '_weight', self.m_weights[m + 1])

        with tf.name_scope("BF_final"):
            self.final_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_output,
                                                                                            logits=self.final_output))
            tf.summary.scalar('final_loss', self.final_loss)

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.final_loss)

    def _add_fc_layer(self, layer_input, size, activation_fn=tf.nn.relu, batch_norm=True,
                      is_training=True):
        """
        Add a single fc layer
        :param layer_input:
        :param size:
        :param activation_fn:
        :param batch_norm:
        :param is_training:
        :return:
        """
        layer_output = tc.layers.fully_connected(layer_input, size, activation_fn=activation_fn)

        if batch_norm:
            layer_output = self.add_norm(layer_output, size, is_training)
        if is_training:
            layer_output = tf.nn.dropout(layer_output, self.keep_prob)

        return layer_output

    def m_outputs(self):
        return self.m_outputs

    def m_weights(self):
        return self.m_weights

    def m_train_ops(self):
        return self.m_train_ops

    def m_losses(self):
        return self.m_losses

    def final_output(self):
        return self.final_output

    def add_fc_stack_layers(self, layer_input, layer_configure, is_training, batch_norm=True):
        """
        Add a stack of fc layers with size configureation
        :param layer_input:
        :param layer_configure:
        :param is_training:
        :param batch_norm:
        :return:
        """
        out = layer_input
        for size in layer_configure:
            out = self._add_fc_layer(out, size, is_training=is_training, batch_norm=batch_norm)
        return out

    @staticmethod
    def add_norm(layer_input, size, is_training):
        """
        Add batch normalization
        :param layer_input:
        :param size:
        :param is_training:
        :return:
        """
        scale = tf.Variable(tf.ones([size], dtype=tf.float32))
        shift = tf.Variable(tf.zeros([size], dtype=tf.float32))
        pop_mean = tf.Variable(tf.zeros([layer_input.get_shape()[-1]], dtype=tf.float32), trainable=False)
        pop_var = tf.Variable(tf.ones([layer_input.get_shape()[-1]], dtype=tf.float32), trainable=False)
        epsilon = 0.001
        if is_training:
            fc_mean, fc_var = tf.nn.moments(layer_input, axes=[0])

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies(
                        [ema_apply_op, tf.assign(pop_var, fc_var), tf.assign(pop_mean, fc_mean)]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
            layer_output = tf.nn.batch_normalization(layer_input, mean, var, shift, scale, epsilon)
        else:
            layer_output = tf.nn.batch_normalization(layer_input, pop_mean, pop_var, shift, scale, epsilon)
        return layer_output
