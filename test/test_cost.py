from unittest import TestCase
import math
import numpy as np
import theano
import theano.tensor as TT

from theanets.cost import NegativeLogLikelihood, QuadraticCost


class SupervisedCostTest(object):
    def setUp(self):
        nn_out = TT.matrix("nn_out")
        self.cost_fn = theano.function([nn_out] + self.cost.inputs,
                                    self.cost.cost(nn_out))

    def test_regularization(self):
        params = [TT.matrix()]
        self.cost.l1 = 0.
        self.cost.l2 = 0.
        self.assertEqual(self.cost._regularization(params), 0.)

        self.cost.l1 = 0.5
        self.cost.l2 = 0.5
        self.assertNotEqual(self.cost._regularization(params), 0.)

        self.cost.l1 = 0.
        self.cost.l2 = 0.

    def test_inputs(self):
        self.assertEqual(type(self.cost.inputs), list)
        self.assertEqual(type(self.cost.inputs[0]), type(TT.matrix()))

    def test_cost(self):
        identity = np.identity(10, dtype=np.int32)
        self.assertEqual(self.cost_fn(identity.astype(np.float32),
                                      np.arange(10, dtype=np.int32)), 0)

        rand = np.random.random((10, 10))
        rand /= np.sum(rand, axis=1)
        self.assertNotEqual(self.cost_fn(rand, np.arange(10, dtype=np.int32)), 0)

    def test_accuracy(self):
        output = TT.matrix("output")
        acc = self.cost.accuracy(output)
        acc_fn = theano.function([output] + self.cost.inputs, acc)

        rand = np.random.random((1000, 10))
        target = np.asarray(np.argmax(np.random.random((1000, 10)), axis=1),
                            dtype=np.int32)
        self.assertLess(acc_fn(rand, target), 20.)

        output = np.asarray([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                             [1, 0, 0]], dtype=np.float32)

        target = np.asarray([4] * 5, dtype=np.int32)
        self.assertEqual(acc_fn(output, target), 0.)

        target = np.asarray([0] * 5, dtype=np.int32)
        self.assertEqual(acc_fn(output, target), 100.)


class TestNegativeLogLikelihood(TestCase, SupervisedCostTest):
    def setUp(self):
        self.cost = NegativeLogLikelihood()
        SupervisedCostTest.setUp(self)

    def test_NLL(self):
        p_y_given_x = np.asarray(
            [[0.8, 0.2, 0],
             [0.5, 0.5, 0],
             [0.001, 0, 0]])
        y = np.asarray( [0, 0, 0], dtype=np.int32)
        self.assertEqual(self.cost_fn(p_y_given_x, y),
                         1.0/3.0*(-math.log(0.8) + -math.log(0.5)
                                  + -math.log(0.001)))

#
# class TestQuadraticCost(TestCase, SupervisedCostTest):
#    def setUp(self):
#        self.cost = QuadraticCost()
#        SupervisedCostTest.setUp(self)
