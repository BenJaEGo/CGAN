import tensorflow as tf


class AffinePlusNonlinearLayer(object):
    def __init__(self, name, n_input, n_hidden, activation=None):
        self._name = name
        self._n_input = n_input
        self._n_hidden = n_hidden
        self._activation = activation

        self._weights = tf.get_variable(name=name + "_weights",
                                        shape=[n_input, n_hidden],
                                        initializer=tf.truncated_normal_initializer(
                                            mean=0.0,
                                            stddev=0.01
                                        ))
        self._biases = tf.get_variable(name=name + "_bias",
                                       shape=[n_hidden],
                                       initializer=tf.constant_initializer(value=0.0)
                                       )

    def forward(self, input_tensor, act=True):
        output_tensor = tf.matmul(input_tensor, self._weights) + self._biases
        if self._activation is not None and act is True:
            output_tensor = self._activation(output_tensor)
        return output_tensor

    def get_weight_decay_loss(self):
        return tf.nn.l2_loss(t=self._weights)

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases
