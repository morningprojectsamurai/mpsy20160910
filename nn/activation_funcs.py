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


def logistic(s):
    return 1/(1 + np.exp(-s))


def d_logistic(y):
    return y * (1 - y)


def tanh(s):
    return np.tanh(s)


def d_tanh(s):
    return 1 - np.power(np.tanh(s), 2)


def rectifier(s):
    return np.max(np.array([s, np.zeros(shape=s.shape)]), axis=0)


def d_rectifier(s):
    d = np.zeros(shape=s.shape)
    d[s > 0] = 1.0
    return d