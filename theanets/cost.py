from theano import tensor as TT, tensor
import theano
from theano.printing import Print


class SupervisedCost():
    def __init__(self, l1_factor=0., l2=0.):
        self.target = TT.ivector("target")
        self.l1 = l1_factor
        self.l2 = l2

    def _regularization(self, params):
        regul = 0.
        if self.l1 != 0.:
            regul += self.l1*(sum([TT.sum(p) for p in params]))
        if self.l2 != 0.:
            regul += self.l2*(sum([TT.sqrt(p**2) for p in params]))

        return regul

    def cost(self, outputs, params=()):
        return NotImplementedError("")

    def accuracy(self, output):
        return 100 * TT.mean(TT.eq(self.target, TT.argmax(output, axis=1)))

    @property
    def inputs(self):
        return [self.target]


class NegativeLogLikelihood(SupervisedCost):
    def cost(self, outputs, params=()):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type outputs: theano.tensor.TensorType
        :param outputs: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        return -TT.mean(
            TT.log(outputs[TT.arange(self.target.shape[0]), self.target])) + \
            self._regularization(params)


class QuadraticCost(SupervisedCost):
    def cost(self, output, params=()):
        return 0.5*(TT.sum(output - self.target))**2 + \
            self._regularization(params)