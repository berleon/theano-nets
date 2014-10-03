import operator

import numpy as np
import theano
import theano.tensor as TT
import theano.tensor.signal.downsample as downsample
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


FLOAT = theano.config.floatX


class Layer(object):
    '''
    The Superclass of all layers. Use on of its subclasses (`ConvPoolingLayer`,
    `PerceptionLayer`) or extend this class and implement the `output()` method.

    Parameters
    ----------
    name : str
        The name of this layer. It is used as a suffix for the weights and bias
        of this layer.

    input_shape : tuple
        The shape of the input of the layer.

    output_shape : tuple
        The shape of the output of the layer.

    weight_shape : tuple, optional
        The shape of the weights. Defaults are `(input_shape, output_shape)`

    bias_shape : tuple, optional
        The shape of the biases. Defaults are `(output_shape, )`

    activation : str, optional
        The name of an activation function to use on hidden network units.
        Defaults to 'sigmoid'.

    rng : theano RandomStreams object, optional
        Use a specific Theano random number generator. A new one will be created
        if this is None.

    noise : float, optional
        Standard deviation of desired noise to inject into input.

    dropouts : float in [0, 1], optional
        Proportion of input units to randomly set to 0.

    sparse : float in [0, 1], optional
        If given, ensure that the weight matrix for the layer has only this
        proportion of nonzero entries.

    Attributes
    ----------

    activation_fn : function
        The activation function.

    input : TensorType
        The input of this layer.
    '''

    def __init__(self, name, input_shape, output_shape, activation=None,
                 weight_shape=None, bias_shape=None, sparse=0., dropout=0.,
                 noise=0.,
                 rng=None):
        self.name = name
        self.batchsize = TT.lscalar("batchsize_{}".format(name))
        if type(input_shape) == int:
            input_shape = (input_shape, )
        self.input_shape = (self.batchsize, ) + input_shape

        if type(output_shape) == int:
            output_shape = (output_shape, )
        self.output_shape = (self.batchsize, ) + output_shape

        if weight_shape is None:
            self.weight_shape = input_shape + output_shape
        else:
            self.weight_shape = weight_shape

        if bias_shape is None:
            self.bias_shape = (output_shape)
        else:
            self.bias_shape = bias_shape

        if activation is None:
            self.activation_fn = TT.nnet.sigmoid
        else:
            self.activation_fn = activation

        self._weights = []
        self._biases = []

        if rng is not None:
            self.rng = rng
        else:
            self.rng = RandomStreams()

        self.sparse = sparse
        self.dropout = dropout
        self.noise = noise

        # layer input
        self._input = self._tensor("input_{}".format(name), self.input_shape)

    def __getstate__(self):
        return {
            "name": self.name,
            "sparse": self.sparse,
            "dropout": self.dropout,
            "noise": self.noise,
            "weights": self._weights,
            "biases": self._biases,
            "input_shape": self.input_shape[1:],
            "output_shape": self.output_shape[1:],
            "weight_shape": self.weight_shape,
            "bias_shape": self.bias_shape,
            "activation": self.activation_fn
        }

    def __setstate__(self, s):
        Layer.__init__(self,
                       name=s["name"],
                       input_shape=s["input_shape"],
                       output_shape=s["output_shape"],
                       activation=s["activation"],
                       weight_shape=s["weight_shape"],
                       bias_shape=s["bias_shape"],
                       sparse=s["sparse"],
                       dropout=s["dropout"],
                       noise=s["noise"])

    def __eq__(self, other):
        return (self.name == other.name and
                self.input_shape[1:] == other.input_shape[1:] and
                self.output_shape[1:] == other.output_shape[1:] and
                self.activation_fn == other.activation_fn and
                self.weight_shape == other.weight_shape and
                self.bias_shape == other.bias_shape and
                self.sparse == other.sparse and
                self.noise == other.noise)

    def output(self):
        raise NotImplementedError("Don't initialize Layer directly. Create a "
                                  "subclass and implement the output method")

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, input):
        self._input = input

    @property
    def monitors(self):
        return {}

    def chain(self, next_layer):
        n_out = reduce(operator.mul, self.output_shape[1:])
        n_in = reduce(operator.mul, next_layer.input_shape[1:])

        assert n_out == n_in, \
            "Shapes differs in size!\n" \
            "Layer '{}' has shape {} #{}.\n" \
            "Layer '{}' has shape {} #{}." \
                .format(self.name, self.output_shape[1:], n_out,
                        next_layer.name, next_layer.input_shape[1:], n_in)
        output = self.output()
        next_layer.input = output.reshape(next_layer.input_shape)
        return output

    def params(self):
        return self._weights + self._biases

    def _add_noise(self, x):
        '''Add noise and dropouts to elements of x as needed.

        Parameters
        ----------
        x : Theano array
            Input array to add noise and dropouts to.
        sigma : float
            Standard deviation of gaussian noise to add to x. If this is 0, then
            no gaussian noise is added to the values of x.
        rho : float, in [0, 1]
            Fraction of elements of x to set randomly to 0. If this is 0, then
            no elements of x are set randomly to 0. (This is also called
            "masking noise" (for inputs) or "dropouts" (for hidden units).)

        Returns
        -------
        Theano array
            The parameter x, plus additional noise as specified.
        '''

        if self.noise > 0:
            noise = self.rng.normal(size=x.shape, std=self.noise, dtype=FLOAT)
            x = x + noise

        if self.dropout > 0:
            mask = self.rng.binomial(size=x.shape, n=1, p=1 - self.dropout,
                                     dtype=FLOAT)
            x *= mask

        return x

    @staticmethod
    def random_weight(weight_shape, bias_shape, suffix, sparse=None):
        '''Create a layer of weights and bias values.

        Parameters
        ----------
        weight_shape: tuple

        bias_shape: tuple
            Number of columns of the weight matrix -- equivalently, the number
            of "output" units that the weight matrix connects.
        suffix : str
            A string suffix to use in the Theano name for the created variables.
            This string will be appended to 'W_' (for the weights) and 'b_' (for
            the biases) parameters that are created and returned.
        sparse : float in (0, 1)
            If given, ensure that the weight matrix for the layer has only this
            proportion of nonzero entries.

        Returns
        -------
        weight : Theano shared array
            A shared array containing Theano values representing the weights
            connecting each "input" unit to each "output" unit.
        bias : Theano shared array
            A shared array containing Theano values representing the bias
            values on each of the "output" units.
        '''

        fan_in = np.prod(weight_shape)
        arr = np.random.uniform(
            low=-np.sqrt(3. / fan_in),
            high=np.sqrt(3. / fan_in),
            size=weight_shape)
        if sparse is not None:
            arr *= np.random.binomial(n=1, p=sparse, size=weight_shape)

        weight = theano.shared(arr.astype(FLOAT),
                               name='W_{}'.format(suffix),
                               borrow=True)
        # TODO: check if zeros initialized bias is good!
        bias = theano.shared(np.zeros(bias_shape, FLOAT),
                             name='b_{}'.format(suffix),
                             borrow=True)
        # logging.info('weights for layer %s: %s x %s', suffix, input_shape,
        # output_shape)
        return weight, bias

    @staticmethod
    def _tensor(name, dim):
        tensors = [TT.vector, TT.matrix, TT.tensor3, TT.tensor4]
        return tensors[len(dim) - 1](name)


class PerceptonLayer(Layer):
    def __init__(self, name, input_shape, output_shape, activation=None,
                 sparse=0., dropout=0., noise=0., rng=None):
        super(PerceptonLayer, self). \
            __init__(name, input_shape, output_shape, activation=activation,
                     weight_shape=None, sparse=sparse, dropout=dropout,
                     noise=noise, rng=rng)

    def output(self):
        W, b = self.random_weight(self.weight_shape, self.bias_shape, self.name,
                                  sparse=self.sparse)
        self._weights.append(W)
        self._biases.append(b)
        return self.activation_fn(TT.dot(self._add_noise(self._input), W) + b)


class ConvPoolingLayer(Layer):
    '''
    A convolutional neuronal network layer.
    The `input_shape` is determined by
    Parameters:
    ----------

    filter_shape : tuple, optional
        Shape of the filter. Default is (3,3)

    n_input_fmaps : int
        Number of input feature maps

    n_output_fmpas : int
        Number of output feature maps.

    image_shape: tuple
        The size (weight, height) of the input image.

    poolsize: tuple
        The number of pixels that get pooled together


    '''

    def __init__(self, name, n_output_fmaps, image_shape, n_input_fmaps,
                 filter_shape=(3, 3), activation=None, dropout=0., noise=0.,
                 rng=None, poolsize=(2, 2)):
        input_shape = (n_input_fmaps,) + image_shape
        weight_shape = (n_output_fmaps,) + (n_input_fmaps,) + filter_shape
        bias_shape = (n_output_fmaps,)
        output_shape = (n_output_fmaps,) + \
                       ((image_shape[0] - filter_shape[0] + 1) / poolsize[0],
                        (image_shape[1] - filter_shape[1] + 1) / poolsize[1])

        self.poolsize = poolsize
        self.image_shape = image_shape
        self.filter_shape = filter_shape

        super(ConvPoolingLayer, self). \
            __init__(name, input_shape, output_shape, activation=activation,
                     weight_shape=weight_shape, bias_shape=bias_shape,
                     dropout=dropout, noise=noise, rng=rng)

    def output(self):
        W, b = self.random_weight(self.weight_shape, self.bias_shape, self.name,
                                  sparse=self.sparse)
        conv_out = TT.nnet.conv2d(input=self._input, filters=W)
        pool_out = downsample.max_pool_2d(input=conv_out, ds=self.poolsize,
                                          ignore_border=False)
        self._weights.append(W)
        self._biases.append(b)
        return self.activation_fn(pool_out + b.dimshuffle('x', 0, 'x', 'x'))

    def __setstate__(self, s):
        super(ConvPoolingLayer, self).__setstate__(s)
        self.poolsize = s.get("poolsize", (2, 2))

    def __getstate__(self):
        s = super(ConvPoolingLayer, self).__getstate__()
        return s


class Network(object):
    def __init__(self, layers, cost):
        self._cost_needs_compile = True
        assert len(layers) >= 1

        first_layer = layers[0]
        last_layer = layers[-1]

        self.layer_outputs = [
            layer.chain(next_layer)
            for layer, next_layer in zip(layers[:-1], layers[1:])
        ]


        self._input = first_layer.input
        self.output = last_layer.output()
        # TODO: check if self.params need to be flattened
        self._params = sum([l.params() for l in layers], [])
        self._layers = layers
        self._cost_class = cost
        # lazy initialization in self.cost()
        self._cost_fn = None
        self._monitors = {"acc": self.cost_class.accuracy(self.output)}

    @property
    def inputs(self):
        return [self._input]

    @property
    def layers(self):
        return self._layers

    def params(self):
        return self._params

    def accuracy(self, data, target):
        '''Compute the percent correct classifications.'''
        if not hasattr(self, '_accuracy_fn'):
            accuracy = self._cost_class.accuracy(self.output)
            self._accuracy_fn = theano.function(
                self.inputs + self.cost_class.inputs,
                accuracy,
                givens=self._batchsizes(len(data)),
                on_unused_input='ignore')

        return self._accuracy_fn(data, target)

    @property
    def monitors(self):
        for name, monitor in self._monitors.iteritems():
            yield name, monitor
        for l in self._layers:
            for name, monitor in l.monitors.iteritems():
                yield name, monitor

    def _compile(self):
        '''If needed, compile the Theano function for this network.'''
        if getattr(self, '_func', None) is None:
            self._func = theano.function(self.inputs,
                                         [self.output],
                                         givens=self._batchsizes(1),
                                         on_unused_input='ignore')

    def _batchsizes(self, batchsize=1):
        return {l.batchsize: batchsize for l in self._layers}

    def perdict(self, x):
        self._compile()
        return self._func(x)

    @property
    def cost_class(self):
        return self._cost_class

    @cost_class.setter
    def cost_class(self, clss):
        self._cost_class = clss
        self._cost_needs_compile = True

    def cost(self, input, target=None):
        if self._cost_needs_compile:
            cost = self._cost_class.cost(self.output, self.params())
            self._cost_fn = theano.function(
                self.inputs + self._cost_class.inputs,
                cost)

        if target is None:
            return self._cost_fn(input)
        else:
            return self._cost_fn(input, target)

    def save(self, path):
        import cPickle

        with open(path, "w+") as f:
            cPickle.dump(self, f)

    @classmethod
    def load(cls, path):
        import cPickle

        with open(path, "r") as f:
            return cPickle.load(f)

    def __getstate__(self):
        return {
            "cost": self.cost_class,
            "layers": self.layers
        }

    def __setstate__(self, state):
        Network.__init__(self, state["layers"], state["cost"])

