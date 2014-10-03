
import operator
import unittest
import inspect
import tempfile
import cPickle

import numpy as np
from theanets.cost import NegativeLogLikelihood
from theanets.dataset import Dataset
import theano
import theano.tensor as TT

import theanets
import theanets.layers as layers

# theano.config.exception_verbosity = 'high'
# theano.config.optimizer = 'fast_compile'


class LayerTest(object):
    def test_properties_types(self):
        tensor_type = type(TT.matrix())
        self.assertEqual(type(self.layer.input), tensor_type)
        self.assertTrue(inspect.ismethod(self.layer.output))
        self.assertEqual(type(self.layer.params()), list)
        self.assertEqual(type(self.layer.monitors), dict)

    def test_chain(self):
        n_out = reduce(operator.mul, self.layer.output_shape[1:])
        layer_ok = layers.PerceptonLayer(name="chain_test_ok",
                                         input_shape=(n_out,),
                                         output_shape=(5, ))

        self.layer.chain(layer_ok)

        layer_fail = layers.PerceptonLayer(name="chain_test_fail",
                                         input_shape=(n_out + 10,),
                                         output_shape=(5, ))

        self.assertRaises(AssertionError, self.layer.chain, layer_fail)

    def test_pickle(self):
        with tempfile.NamedTemporaryFile() as f:
            cPickle.dump(self.layer, f)
            f.flush()
            f.seek(0)
            print(f.name)
            unpickled = cPickle.load(f)
            self.assertEqual(self.layer, unpickled)


class PerceptionLayerTest(unittest.TestCase, LayerTest):
    def setUp(self):
        self.layer = layers.PerceptonLayer(name="perception_test",
                                           input_shape=(20,),
                                           output_shape=(10,))


class ConvPoolingLayerTest(unittest.TestCase, LayerTest):
    def setUp(self):
        self.layer = layers.ConvPoolingLayer(name="test",
                                             image_shape=(20, 20),
                                             n_input_fmaps=3,
                                             n_output_fmaps=3,)


class NetworkTest(unittest.TestCase):
    def setUp(self):
        super(NetworkTest, self).setUp()

        self.network = layers.Network(
            layers=[
                layers.ConvPoolingLayer(name="1",
                                        image_shape=(28, 28),
                                        n_input_fmaps=1,
                                        n_output_fmaps=5,
                                        filter_shape=(5, 5)),
                layers.ConvPoolingLayer(name="2",
                                        image_shape=(12, 12),
                                        n_input_fmaps=5,
                                        n_output_fmaps=50,
                                        filter_shape=(5, 5)),
                layers.PerceptonLayer(name="3", input_shape=50*4*4,
                                      output_shape=500, activation=TT.tanh),
                layers.PerceptonLayer(name="4", input_shape=500,
                                      output_shape=10, activation=TT.tanh),
            ],
            cost=NegativeLogLikelihood()
        )

    def test_input_output(self):
        self.assertListEqual(self.network.inputs, [self.network.layers[0].input])

    def test_predict(self):
        print(self.network.perdict(np.random.randn(1, 1, 28, 28)))

    def test_output_shapes(self):
        layer_output = theano.function(
            self.network.inputs,
            [l.output() for l in self.network.layers],
            givens={l.batchsize: 20 for l in self.network.layers},
            on_unused_input='ignore'
        )
        self.assertListEqual(
            [o.shape for o in layer_output(np.random.random((20, 1, 28, 28)))],
            [(20, 5, 12, 12), (20, 50, 4, 4), (20, 500), (20, 10)]
        )