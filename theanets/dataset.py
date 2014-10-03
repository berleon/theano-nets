# Copyright (c) 2014 Leon Sixt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''This file contains a class for handling batched datasets.'''

import climate
import numpy.random as rng
import numpy as np
from sklearn.datasets import fetch_mldata
logging = climate.get_logger(__name__)


class Dataset(object):
    def __init__(self, data, target=None):
        self.size = target.shape[0]
        self.data = data
        self.target = target

    def split(self, *ratios):
        assert sum(ratios) == 1, "ratios must sum to 1"
        ratio2N = lambda r: r*self.size
        last = 0
        for r in ratios:
            yield Dataset(data=self.data[ratio2N(last):ratio2N(r+last)],
                          target=self.target[ratio2N(last):ratio2N(r+last)])
            last += r

    def reshape(self, shape):
        self.data = self.data.reshape((self.size,) + shape)

    def shuffle(self):
        rng_state = rng.get_state()
        rng.shuffle(self.data)
        rng.set_state(rng_state)
        rng.shuffle(self.target)

    def minibatches(self, size=20):
        assert size < self.size
        for s, e in zip(range(0, self.size-size, size),
                        range(size, self.size, size)):
            yield self.data[s:e], self.target[s:e]

    @staticmethod
    def fetch_mldata(dataname, target_name='label', data_name='data',
                 transpose_data=True, data_home=None):
        raw_dataset = fetch_mldata(dataname, target_name=target_name,
                               data_name=data_name,
                               transpose_data=transpose_data,
                               data_home=data_home)

        dataset = Dataset(data=raw_dataset.data,
                          target=np.asarray(raw_dataset.target, np.int32))
        return dataset

