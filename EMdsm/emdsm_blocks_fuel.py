from collections import OrderedDict
from functools import partial

import numpy
import theano
from blocks.bricks import Initializable, Random, Activation
from blocks.bricks.base import application, lazy
from blocks.extensions import SimpleExtension
from blocks.initialization import Constant
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.utils import shared_floatx_nans, dict_union
from fuel.datasets import IndexableDataset
from fuel.schemes import IterationScheme
from picklable_itertools import chain, repeat, imap
from theano import tensor
from theano.gradient import disconnected_grad

class BurnIn(SimpleExtension):
    """Resets FivEM's h and h_prev vars and performs a burn-in.

    Parameters
    ----------
    model_brick : FivEM
        FivEM model.
    num_steps : int, optional
        Number of burn-in steps. Default to 1.
    """
    def __init__(self, model_brick, num_steps=1, **kwargs):
        super(BurnIn, self).__init__(**kwargs)
        self.model_brick = model_brick
        self.num_steps = num_steps
        self._compile_update_function()

    def _compile_update_function(self):
        x = tensor.matrix('x')
        self.update_function = theano.function(
            inputs=[x],
            updates=OrderedDict([
                (self.model_brick.h_prev, self.model_brick.h),
                (self.model_brick.h,
                 self.model_brick.langevin_update(x, self.model_brick.h))]))

    def do(self, which_callback, *args):
        if which_callback == 'before_batch':
            batch, = args
            self.model_brick.h_prev.set_value(
                0 * self.model_brick.h_prev.get_value())
            self.model_brick.h.set_value(
                0 * self.model_brick.h.get_value())
            for i in range(self.num_steps):
                self.update_function(batch)

class GenerateAndPlot2DSamples(SimpleExtension):
    """Generate and plot 2-D samples from FivEM model

    Parameters
    ----------
    model_brick : FivEM
        FivEM model.
    num_samples : int, optional
        Number of generated samples. Default to 100.
    """
    def __init__(self, model_brick, num_samples=1, **kwargs):
        super(GenerateAndPlot2DSamples, self).__init__(**kwargs)
        self.model_brick = model_brick
        self.num_samples = num_samples
        self._compile_update_function()

   # TODO

    def _compile_update_function(self):
        x = tensor.matrix('x')
        self.update_function = theano.function(
            inputs=[x],
            updates=OrderedDict([
                (self.model_brick.h_prev, self.model_brick.h),
                (self.model_brick.h,
                 self.model_brick.langevin_update(x, self.model_brick.h))]))

    def do(self, which_callback, *args):
        if which_callback == 'before_batch':
            batch, = args
            self.model_brick.h_prev.set_value(
                0 * self.model_brick.h_prev.get_value())
            self.model_brick.h.set_value(
                0 * self.model_brick.h.get_value())
            for i in range(self.num_steps):
                self.update_function(batch)


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

    def __init__(self, mean, covariance_matrix, num_examples, squash=False, rng=None):
        if not rng:
            rng = numpy.random.RandomState(self._default_seed)
        data = rng.multivariate_normal(mean, covariance_matrix, num_examples)
        if squash:
            data = 0.5*numpy.tanh(-data)
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
    def __init__(self, nvis, nhid, epsilon, batch_size, noise_scaling=1.0,
                 lateral_x=False, lateral_h=False, debug=0, **kwargs):
        super(FivEM, self).__init__(**kwargs)
        self.nvis = nvis
        self.nhid = nhid
        self.lateral_x = lateral_x
        self.lateral_h = lateral_h
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.states_init = Constant(0)
        self.rho = Rho()
        self.noise_scaling = noise_scaling
        self.children = [self.rho]
        self.debug = debug

    def pp(self,var,name,level=1):
        if self.debug >= level:
            return theano.printing.Print(name)(var)
        else:
            return var
        
    def _allocate(self):
        Wxh = shared_floatx_nans((self.nvis, self.nhid), name='Wxh')
        self.parameters.append(Wxh)
        add_role(Wxh, WEIGHT)
        b = shared_floatx_nans((self.nhid), name='b')
        self.parameters.append(b)
        add_role(b, BIAS)
        c = shared_floatx_nans((self.nvis), name='c')
        self.parameters.append(c)
        add_role(c, BIAS)
        Whh = shared_floatx_nans((self.nhid, self.nhid), name='Whh')
        self.parameters.append(Whh)
        add_role(Whh, WEIGHT)
        Wxx = shared_floatx_nans((self.nvis, self.nvis), name='Wxx')
        self.parameters.append(Wxx)
        add_role(Wxx, WEIGHT)
        self.h = shared_floatx_nans((self.batch_size, self.nhid), name='h')
        self.h_prev = shared_floatx_nans((self.batch_size, self.nhid),
                                         name='h_prev')
        self.energy_prev = shared_floatx_nans((),name="energy_prev")
        self.energy_new = shared_floatx_nans((),name="energy_new")

    def _initialize(self):
        Wxh,b,c,Whh,Wxx = self.parameters
        self.weights_init.initialize(Wxh, self.rng)
        self.biases_init.initialize(b, self.rng)
        self.biases_init.initialize(c, self.rng)
        self.weights_init.initialize(Whh, self.rng)
        self.weights_init.initialize(Wxx, self.rng)
        self.states_init.initialize(self.h, self.rng)
        self.states_init.initialize(self.h_prev, self.rng)
        self.states_init.initialize(self.energy_prev, self.rng)
        self.states_init.initialize(self.energy_new, self.rng)

    @property
    def Wxh(self):
        return self.parameters[0]
    
    @property
    def b(self):
        return self.parameters[1]

    @property
    def c(self):
        return self.parameters[2]

    @property
    def Whh(self):
        return self.parameters[3]
    
    @property
    def Wxx(self):
        return self.parameters[4]
    
    def energy(self, x, h):
        """Computes the energy function.

        Parameters
        ----------
        x : tensor variable
            Batch of visible states.
        h : tensor variable
            Batch of hidden states.

        """
        rx = self.rho.apply(x)
        rh = self.rho.apply(h)
        energy = 0.5 * ((x*x).sum(axis=1) + (h*h).sum(axis=1)) - \
                (tensor.dot(rx, tensor.tanh(self.Wxh)) * rh).sum(axis=1) - \
                tensor.dot(rx,self.c) + tensor.dot(rh,self.b)
        if self.lateral_x:
            energy = energy + (tensor.dot(rx, tensor.tanh(self.Wxx)) * rx).sum(axis=1)
        if self.lateral_h:
            energy = energy + (tensor.dot(rh, tensor.tanh(self.Whh)) * rh).sum(axis=1)
        return energy
                
                

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

    def map_update(self, x, h):
        """Computes h update going down the energy gradient, given x.

        Parameters
        ----------
        x : tensor variable
            Batch of visible states.
        h : tensor variable
            Batch of hidden states.
        """
        return (h - self.epsilon * tensor.grad(self.energy(x, h).sum(), h))

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
        h_next = self.pp(disconnected_grad(self.langevin_update(self.pp(x,"x",3),
                                                                self.pp(h,"h",2))),
                        "h_next",2)
        old_energy = self.pp(self.energy_prev,"old_energy",2)
        new_energy = self.pp(self.energy(x,h_next).sum(dtype=old_energy.dtype),"new_energy",1)
        delta_energy = self.pp(old_energy - new_energy,"delta_energy",1)
        updates = OrderedDict([(h_prev, h), (h, h_next),(self.energy_prev,self.energy_new),(self.energy_new,new_energy)])
        application_call.updates = dict_union(application_call.updates,
                                              updates)
        h_prediction_residual = (h_next - self.pp(h_prev,"h_prev",3) + self.epsilon *
                                 tensor.grad(self.energy(x, h_prev).sum(), h_prev))
        J_h = self.pp((h_prediction_residual*h_prediction_residual).sum(axis=1).mean(axis=0),"J_h",1)
        x_prediction_residual = self.pp(tensor.grad(self.energy(x, h_prev).sum(), x),"x_residual",2)
        J_x = self.pp((x_prediction_residual*x_prediction_residual).sum(axis=1).mean(axis=0),"J_x",1)
        ## the values printed using add_auxiliary_variables are not reliable, so use printing.Print instead
        #application_call.add_auxiliary_variable(self.energy_prev, name="energy_prev")
        #application_call.add_auxiliary_variable(self.energy_new, name="energy_new")
        #application_call.add_auxiliary_variable(self.energy_prev - self.energy_new, name="energy_decrease")
        if self.debug>0:
           application_call.add_auxiliary_variable(J_x, name="J_x")
           application_call.add_auxiliary_variable(J_h, name="J_h")
        #application_call.add_auxiliary_variable(x*1, name="x")
        #application_call.add_auxiliary_variable(h_prev, name="h_prev")
        #application_call.add_auxiliary_variable(h, name="h")
        #application_call.add_auxiliary_variable(h_next, name="h_next")
        if self.debug>1:
           application_call.add_auxiliary_variable(self.Wxh*1., name="Wxh")
           application_call.add_auxiliary_variable(self.Whh*1., name="Whh")
           application_call.add_auxiliary_variable(self.Wxx*1., name="Wxx")
           application_call.add_auxiliary_variable(self.b*1, name="b")
           application_call.add_auxiliary_variable(self.c*1, name="c")
        # YB: the commented lines below make blocks crash, not sure why
        #application_call.add_auxiliary_variable(self.params[0], name="W")
        #application_call.add_auxiliary_variable(self.params[1], name="b")
        #application_call.add_auxiliary_variable(self.params[2], name="c")
        return self.pp(J_x + J_h,"total_cost")


def update_val_maker(n_inference_steps)
    def update_val(n_it, old_value):
        if n_it % n_inference_steps == 0:
            # return 0 * old_value
            return old_value+numpy.random.normal(0,0.1,size=old_value.shape)
        else:
            return old_value
    return update_val
