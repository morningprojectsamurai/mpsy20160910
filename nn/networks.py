# This file is part of "MPS Yokohama Deep Learning Series Day 09/10/2016"
#
# "MPS Yokohama Deep Learning Series Day 09/10/2016"
# is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "MPS Yokohama Deep Learning Series Day 09/10/2016"
# is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
# (c) Junya Kaneko <jyuneko@hotmail.com>


import numpy as np
from nn.layers import RectifierLayer, TanhLayer, LogisticLayer


class Network:
    def __init__(self, name, n_input, d_error_func, epsilon):
        self._name = name
        self._n_input = n_input
        self._layers = []
        self._d_error_func = d_error_func
        self._epsilon = epsilon

    @property
    def name(self):
        return self._name

    @property
    def layers(self):
        return self._layers

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    def add_layer(self, type, n_output):
        n_prev_output = self._layers[-1].n_output if self._layers else self._n_input
        if type == 'logistic':
            layer = LogisticLayer(n_output, n_prev_output)
        elif type == 'tanh':
            layer = TanhLayer(n_output, n_prev_output)
        elif type == 'rectifier':
            layer = RectifierLayer(n_output, n_prev_output)
        else:
            raise Exception('Layer type must be sigmoid, tanh or rectifier.')
        self._layers.append(layer)

    def propagate_forward(self, input_datum):
        output = input_datum
        for layer in self._layers:
            output = layer.propagate_forward(output)
        return output

    def propagate_backward(self, input_datum, teaching_datum):
        delta = self._d_error_func(teaching_datum, self.propagate_forward(input_datum))

        next_layer = None
        for layer in reversed(self._layers):
            delta = layer.propagate_backward(delta, next_layer.W if next_layer is not None else None)
            next_layer = layer

    def update(self, input_datum):
        prev_layer = None
        for layer in self._layers:
            layer.update(input_datum if prev_layer is None else prev_layer.y, self._epsilon)
            prev_layer = layer


class Classifier(Network):
    def get_class(self):
        return np.argmax(self._layers[-1].y)
