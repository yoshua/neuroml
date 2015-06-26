from collections import OrderedDict
from functools import partial

import numpy
from blocks.bricks import Initializable, Random, Activation
from blocks.bricks.base import application, lazy
from blocks.initialization import Constant
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans, dict_union
from fuel.datasets import IndexableDataset
from fuel.schemes import IterationScheme
from picklable_itertools import chain, repeat, imap
from theano import tensor
from theano.gradient import disconnected_grad


class Repeat(IterationScheme):
    """Sends repeated requests.

    Parameters
    ----------
    iteration_scheme : :class:`fuel.schemes.IterationScheme`
        Iteration scheme providing requests to repeat.
    times : int
        Number of repeats.

    """
    def __init__(self, iteration_scheme, times):
        self.iteration_scheme = iteration_scheme
        self.times = times

    def get_request_iterator(self):
        request_iterator = self.iteration_scheme.get_request_iterator()
        return chain.from_iterable(
            imap(partial(repeat, times=self.times), request_iterator))


class Toy2DGaussianDataset(IndexableDataset):
    """A toy dataset of 2D gaussian samples.

    Samples are drawn upon instantiation and kept fixed afterwards.

    Parameters
    ----------
    mean : numpy.ndarray
        2D mean vector.
    covariance_matrix : numpy.ndarray
        2x2 covariance matrix.
    num_examples : int
        Number of examples to generate.
    rng : numpy.random.RandomState, optional
        Numpy RNG used to draw from the distribution defined by `mean`
        and `covariance_matrix`. Defaults to `None`, in which case a
        RandomState with seed Toy2DGaussianDataset._default_seed will
        be used.

    """
    _default_seed = 1234

    def __init__(self, mean, covariance_matrix, num_examples, rng=None):
        if not rng:
            rng = numpy.random.RandomState(self._default_seed)
        data = rng.multivariate_normal(mean, covariance_matrix, num_examples)
        super(Toy2DGaussianDataset, self).__init__(
            indexables={'features': data})


class Rho(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return (tensor.switch(input_ + 0.5 > 0, input_, 0) -
                tensor.switch(input_ - 0.5 > 0, input_, 0))


class FivEM(Initializable, Random):
    """Implementation of the 5EM model.

    The model this brick represents is a simple bipartite, energy-based,
    undirected graphical model.

    Parameters
    ----------
    nvis : int
        Number of visible units.
    nhid : int
        Number of hidden units.
    epsilon : float
        Step size.
    batch_size : int
        Batch size, used for initializing the persistent states h_prev
        and h.

    """
    @lazy(allocation=['nvis', 'nhid'])
    def __init__(self, nvis, nhid, epsilon, batch_size, noise_scaling=1., **kwargs):
        super(FivEM, self).__init__(**kwargs)
        self.nvis = nvis
        self.nhid = nhid
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.h_init = Constant(0)
        self.h_prev_init = Constant(0)
        self.energy_prev_init = Constant(0)
        self.energy_new_init = Constant(0)
        self.rho = Rho()
        self.noise_scaling = noise_scaling
        self.children = [self.rho]

    def _allocate(self):
        W = shared_floatx_nans((self.nvis, self.nhid), name='W')
        self.params.append(W)
        add_role(W, WEIGHT)
        self.h = shared_floatx_nans((self.batch_size, self.nhid), name='h')
        self.h_prev = shared_floatx_nans((self.batch_size, self.nhid),
                                         name='h_prev')
        self.energy_prev = shared_floatx_nans((),name="energy_prev")
        self.energy_new = shared_floatx_nans((),name="energy_new")

    def _initialize(self):
        W, = self.params
        self.weights_init.initialize(W, self.rng)
        self.h_init.initialize(self.h, self.rng)
        self.h_prev_init.initialize(self.h_prev, self.rng)
        self.energy_prev_init.initialize(self.energy_prev, self.rng)
        self.energy_new_init.initialize(self.energy_new, self.rng)

    @property
    def W(self):
        return self.params[0]

    def energy(self, x, h):
        """Computes the energy function.

        Parameters
        ----------
        x : tensor variable
            Batch of visible states.
        h : tensor variable
            Batch of hidden states.

        """
        return (0.5 * (tensor.dot(x, x.T) + tensor.dot(h, h.T)) -
                (tensor.dot(self.rho.apply(x), self.W) *
                 self.rho.apply(h)).sum(axis=1))

    def langevin_update(self, x, h, update_x=False):
        """Computes state updates according to Langevin dynamics.

        Parameters
        ----------
        x : tensor variable
            Batch of visible states.
        h : tensor variable
            Batch of hidden states.
        update_x : bool, optional
            Whether to return updates for visible states as well. Defaults
            to `False`.

        """
        if update_x:
            return (
                self.corrupt(x) - self.epsilon * tensor.grad(
                    self.energy(x, h).sum(), x),
                self.corrupt(h) - self.epsilon * tensor.grad(
                    self.energy(x, h).sum(), h))
        else:
            return (self.corrupt(h) - self.epsilon *
                    tensor.grad(self.energy(x, h).sum(), h))

    def corrupt(self, var):
        """Adds zero-mean gaussian noise to the input variable.

        Parameters
        ----------
        var : tensor variable
            Input.

        """
        return var + 2 * self.epsilon * self.noise_scaling * self.theano_rng.normal(
            size=var.shape, dtype=var.dtype)

    @application(inputs=['x'], outputs=['value'])
    def cost(self, x, application_call):
        """Computes the loss function.

        Parameters
        ----------
        x : tensor variable
            Batch of visible states.

        Notes
        -----
        The `application_call` argument is an effect of the `application`
        decorator and isn't visible to users. It's used internally to
        set an updates dictionary for `h_prev` and `h` that's
        discoverable by `ComputationGraph`.

        """
        h_prev = self.h_prev
        h = self.h
        h_next = disconnected_grad(self.langevin_update(x, h))
        old_energy = self.energy_prev
        new_energy = self.energy(x,h_next).mean(dtype=old_energy.dtype)
        delta_energy = old_energy - new_energy
        updates = OrderedDict([(h_prev, h), (h, h_next),(self.energy_prev,self.energy_new),(self.energy_new,new_energy)])
        application_call.updates = dict_union(application_call.updates,
                                              updates)
        application_call.add_auxiliary_variable(self.energy_prev, name="energy_prev")
        application_call.add_auxiliary_variable(self.energy_new, name="energy_new")
        application_call.add_auxiliary_variable(self.energy_prev - self.energy_new, name="energy_decrease")
        energy_sqrt = (h_next - h_prev + self.epsilon *
                       tensor.grad(self.energy(x, h_prev).sum(), h_prev))
        return tensor.dot(energy_sqrt, energy_sqrt.T).mean()
