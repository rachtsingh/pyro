from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.transformed_distribution import Bijector, TransformedDistribution
from pyro.distributions.util import get_probs_and_logits, _get_clamping_buffer

class LogitRelaxedBernoulli(Distribution):
    """
    An LogitRelaxedBernoulli distribution is the log-transform of the RelaxedBernoulli 
    distribution. Each element of the vector of samples is in the range (-\infty, 0].

    The intended use is to downstream compute the exp of these values, which is 
    then from a RelaxedOneHotCategorical distribution. 

    :param torch.Tensor temperature: Relaxation temperature. Should be a 
        positive, 1-dimensional Tensor.
    :param torch.autograd.Variable ps: Probabilities. Should lie in the
        interval `[0,1]`.
    :param logits: Log odds, i.e. :math:`\\log(\\frac{p}{1 - p})`. Either `ps` or
        `logits` should be specified, but not both.
    :param batch_size: The number of elements in the batch used to generate
        a sample. The batch dimension will be the leftmost dimension in the
        generated sample.
    :param log_pdf_mask: Tensor that is applied to the batch log pdf values
        as a multiplier. The most common use case is supplying a boolean
        tensor mask to mask out certain batch sites in the log pdf computation.
    """
    enumerable = False

    def __init__(self, temperature, ps=None, logits=None, batch_size=None, log_pdf_mask=None, *args, **kwargs):
        if (ps is None) == (logits is None):
            raise ValueError("Got ps={}, logits={}. Either `ps` or `logits` must be specified, "
                             "but not both.".format(ps, logits))
        if temperature[0] <= 0 or len(temperature.size()) > 1:
            raise ValueError("temperature should be a 1-dimensional tensor with positive value.")
        self.temperature = temperature
        self.ps, self.logits = get_probs_and_logits(ps=ps, logits=logits, is_multidimensional=False)
        self.log_pdf_mask = log_pdf_mask
        if self.ps.dim() == 1 and batch_size is not None:
            self.ps = self.ps.expand(batch_size, self.ps.size(0))
            self.logits = self.logits.expand(batch_size, self.logits.size(0))
            if log_pdf_mask is not None and log_pdf_mask.dim() == 1:
                self.log_pdf_mask = log_pdf_mask.expand(batch_size, log_pdf_mask.size(0))
        super(LogitRelaxedBernoulli, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`.
        """
        event_dim = 1
        ps = self.ps
        if x is not None:
            if x.size()[-event_dim] != ps.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.ps.size()[-1], but got {} vs {}".format(
                                     x.size(-1), ps.size(-1)))
            try:
                ps = self.ps.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `ps` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(ps.size(), x.size(), str(e)))

        return ps.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`.
        """
        event_dim = 1
        return self.ps.size()[-event_dim:]

    def sample(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`.
        """
        uniforms = torch.zeros(self.logits.data.size()).uniform_()
        eps = _get_clamping_buffer(uniforms)
        uniforms = uniforms.clamp(min=eps, max=1-eps)
        logistic = uniforms.log() - (-uniforms).log1p()
        return Variable((logistic + self.logits.data)/self.temperature)
        

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        u = -self.temperature * x + self.logits
        log_prob = self.temperature.log() + u - 2 * torch.log(1 + torch.exp(u))
        # XXX this allows for the user to mask out certain parts of the score, for example
        # when the data is a ragged tensor. also useful for KL annealing. this entire logic
        # will likely be done in a better/cleaner way in the future
        if self.log_pdf_mask is not None:
            log_prob = log_prob * self.log_pdf_mask
        return torch.sum(log_prob, -1).contiguous().view(batch_log_pdf_shape)

class Sigmoid(Bijector):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def __call__(self, x, *args, **kwargs):
        return torch.sigmoid(x)

    def inverse(self, y, *args, **kwargs):
        eps = _get_clamping_buffer(y)
        return torch.log(y.clamp(min=eps)) + torch.log1p((-y).clamp(max=-eps))

    def log_det_jacobian(self, y, *args, **kwargs):
        sx = torch.sigmoid(x)
        return (sx.log() + (-sx).log1p()).sum().abs()

class RelaxedBernoulli(Distribution):
    def __new__(cls, *args, **kwargs):
        return TransformedDistribution(LogitRelaxedBernoulli(*args, **kwargs), Sigmoid())