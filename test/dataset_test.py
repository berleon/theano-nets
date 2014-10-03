import unittest

from theanets.dataset import Dataset
import numpy as np


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.N = 100
        data = np.random.randn(self.N, 10)
        target = np.random.rand(self.N) > 0.5
        self.dataset = Dataset(data=data, target=target)

    def test_data_size(self):
        self.assertEqual(len(self.dataset.data), self.N)
        self.assertEqual(len(self.dataset.target), self.N)

    def test_split(self):
        a, b, c = self.dataset.split(0.2, 0.3, 0.5)

        self.assertEqual(a.size, self.N * 0.2)
        self.assertEqual(b.size, self.N * 0.3)
        self.assertEqual(c.size, self.N * 0.5)

    def test_minibatch(self):
        for batch, _ in self.dataset.minibatches(size=30):
            self.assertEqual(batch.shape[0], 30)

    def test_fetch_mldata(self):
        mnist = Dataset.fetch_mldata("MNIST (original)")
        self.assertEqual(mnist.size, 70000)
        self.assertEqual(mnist.target.shape, (70000, ))

        self.assertEqual(mnist.data.shape, (70000, 784))
        mnist.reshape((1, 28, 28))
        self.assertEqual(mnist.data.shape, (70000, 1, 28, 28))
