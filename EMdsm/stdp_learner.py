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
                 lateral_x=False, lateral_h=False, debug=0, n_inference_steps=3,
                 initial_noise=0.1, **kwargs):
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
        self.n_inference_steps = n_inference_steps
        self.initial_noise = initial_noise

    def pp(self,var,name,level=1):
        if self.debug >= level:
            return theano.printing.Print(name)(var)
        else:
            return var
        
    def _allocate(self):
        Wxh = shared_floatx_nans((self.nvis, self.nhid), name='Wxh')
        self.params.append(Wxh)
        add_role(Wxh, WEIGHT)
        b = shared_floatx_nans((self.nhid), name='b')
        self.params.append(b)
        add_role(b, BIAS)
        c = shared_floatx_nans((self.nvis), name='c')
        self.params.append(c)
        add_role(c, BIAS)
        Whh = shared_floatx_nans((self.nhid, self.nhid), name='Whh')
        self.params.append(Whh)
        add_role(Whh, WEIGHT)
        Wxx = shared_floatx_nans((self.nvis, self.nvis), name='Wxx')
        self.params.append(Wxx)
        add_role(Wxx, WEIGHT)
        self.h = shared_floatx_nans((self.batch_size, self.nhid), name='h')
        x = tensor.matrix()
        h = tensor.matrix()
        self.generate_step_f = theano.function(inputs=[x,h],outputs=self.langevin_update(x,h,update_x=True))

    def _initialize(self):
        Wxh,b,c,Whh,Wxx = self.params
        self.weights_init.initialize(Wxh, self.rng)
        self.biases_init.initialize(b, self.rng)
        self.biases_init.initialize(c, self.rng)
        self.weights_init.initialize(Whh, self.rng)
        self.weights_init.initialize(Wxx, self.rng)
        self.states_init.initialize(self.h, self.rng)

    @property
    def Wxh(self):
        return self.params[0]
    
    @property
    def b(self):
        return self.params[1]

    @property
    def c(self):
        return self.params[2]

    @property
    def Whh(self):
        return self.params[3]
    
    @property
    def Wxx(self):
        return self.params[4]
    
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

    def map_update(self, x, h, update_x=False):
        """Computes h update going down the energy gradient, given x.

        Parameters
        ----------
        x : tensor variable
            Batch of visible states.
        h : tensor variable
            Batch of hidden states.
        """
        if update_x:
            return ((x - self.epsilon * tensor.grad(self.energy(x, h).sum(), x)),
                    (h - self.epsilon * tensor.grad(self.energy(x, h).sum(), h)))
        else:
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

    @application(inputs=['given_x'], outputs=['value'])
    def cost(self, given_x, application_call):
        """Computes the loss function.

        Parameters
        ----------
        given_x : tensor variable
                  Batch of given visible states from dataset.

        Notes
        -----
        The `application_call` argument is an effect of the `application`
        decorator and isn't visible to users. It's used internally to
        set an updates dictionary for  `h` that's
        discoverable by `ComputationGraph`.

        """
        x = self.pp(given_x,"given_x",2)
        h_prev = self.h + self.initial_noise * \
                 self.theano_rng.normal(size=self.h.shape,dtype=self.h.dtype)
        h = h_next = h_prev
        old_energy = self.pp(self.energy(x,h).sum(),"old_energy",1)

        # try to go towards a fixed point, near the given_x
        for iteration in range(self.n_inference_steps):
            h_prev = h
            h = h_next
            new_x, new_h = self.map_update(self.pp(x,"x",3), self.pp(h_next,"h",2), update_x=True)
            x, h_next = self.pp(disconnected_grad(new_x),"new mapped x",2), disconnected_grad(new_h)
            new_energy = self.pp(self.energy(x,h_next).sum(),"map_new_energy",1)
            delta_energy = self.pp(old_energy - new_energy,"map_delta_energy",1)
            old_energy = new_energy

        # now move back towards given_x and let h settle accordingly
        for iteration in range(self.n_inference_steps):
            h_prev = h
            h = h_next
            x = (1-self.epsilon)*x + self.epsilon*given_x
            h_next = self.pp(disconnected_grad(self.langevin_update(self.pp(x,"x",3),
                                                                    self.pp(h_next,"h",2))),
                        "h_next",2)
            new_energy = self.pp(self.energy(x,h_next).sum(),"new_energy",1)
            delta_energy = self.pp(old_energy - new_energy,"delta_energy",1)
            old_energy = new_energy
            h_prediction_residual = (h_next - self.pp(h_prev,"h_prev",3) + self.epsilon *
                                    tensor.grad(self.energy(x, h_prev).sum(), h_prev))
            J_h = self.pp((h_prediction_residual*h_prediction_residual).sum(axis=1).mean(axis=0),"J_h",1)
            x_prediction_residual = self.pp(tensor.grad(self.energy(given_x, h_prev).sum(), given_x),
                                            "x_residual",2)
            J_x = self.pp((x_prediction_residual*x_prediction_residual).sum(axis=1).mean(axis=0),"J_x",1)
            if self.debug>1:
                application_call.add_auxiliary_variable(J_x, name="J_x"+str(iteration))
                application_call.add_auxiliary_variable(J_h, name="J_h"+str(iteration))
            if iteration == 0:
                total_cost = J_h + J_x
            else:
                total_cost = total_cost + J_h + J_x
                
        per_iteration_cost = total_cost / self.n_inference_steps

        updates = OrderedDict([(self.h, h_next)])
        application_call.updates = dict_union(application_call.updates, updates)

        if self.debug>0:
           application_call.add_auxiliary_variable(per_iteration_cost, name="per_iteration_cost")
        if self.debug>1:
           application_call.add_auxiliary_variable(self.Wxh*1., name="Wxh")
           application_call.add_auxiliary_variable(self.Whh*1., name="Whh")
           application_call.add_auxiliary_variable(self.Wxx*1., name="Wxx")
           application_call.add_auxiliary_variable(self.b*1, name="b")
           application_call.add_auxiliary_variable(self.c*1, name="c")

        return self.pp(total_cost,"total_cost")
            
