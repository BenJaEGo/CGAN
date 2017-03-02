from layers import *


class Generator(object):
    def __init__(self, n_input, n_units, n_output, n_label, hidden_activation, output_activation):
        self._n_input = n_input
        self._n_units = n_units
        self._n_output = n_output
        self._n_label = n_label
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation
        self._weight_decay_loss = 0.0
        self._parameters = list()

        self._n_layer = len(n_units)
        self._hidden_layers = list()
        for layer_idx in range(self._n_layer):
            layer_name = "generator_hidden_layer_" + str(layer_idx + 1)
            if layer_idx is 0:
                n_layer_input = n_input + n_label
            else:
                n_layer_input = n_units[layer_idx - 1]
            n_unit = n_units[layer_idx]
            self._hidden_layers.append(
                AffinePlusNonlinearLayer(layer_name, n_layer_input, n_unit, hidden_activation))

        layer_name = "generator_output_layer"
        self._output_layer = AffinePlusNonlinearLayer(layer_name, n_units[-1], n_output, output_activation)

        for layer_idx in range(self._n_layer):
            self._weight_decay_loss += self._hidden_layers[layer_idx].get_weight_decay_loss()
        self._weight_decay_loss += self._output_layer.get_weight_decay_loss()

        for layer_idx in range(self._n_layer):
            self._parameters.append(self._hidden_layers[layer_idx].weights)
            self._parameters.append(self._hidden_layers[layer_idx].biases)

        self._parameters.append(self._output_layer.weights)
        self._parameters.append(self._output_layer.biases)

    def forward(self, input_tensor_latent, input_tensor_label):
        input_tensor = tf.concat(values=[input_tensor_latent, input_tensor_label], axis=1)
        output_tensor = input_tensor
        for layer_idx in range(self._n_layer):
            output_tensor = self._hidden_layers[layer_idx].forward(output_tensor)
        output_tensor = self._output_layer.forward(output_tensor)

        return output_tensor

    @property
    def wd_loss(self):
        return self._weight_decay_loss

    @property
    def parameters(self):
        return self._parameters

