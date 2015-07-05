import functools
import itertools
import logging
from collections import OrderedDict
from argparse import ArgumentParser

import numpy
import theano
import pdb
from blocks.algorithms import GradientDescent, Adam, Scale, Momentum, AdaDelta, RMSProp, AdaGrad
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.training import SharedVariableModifier
from blocks.extensions.saveload import Checkpoint
from blocks.serialization import load
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.utils import shared_floatx, shared_floatx_zeros
from blocks.extensions.monitoring import TrainingDataMonitoring
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme,ConstantScheme,SequentialScheme
from fuel.transformers import Flatten
from fuel.datasets import MNIST
from theano import tensor

from emdsm_blocks_fuel import FivEM, Toy2DGaussianDataset, Repeat

    
def create_main_loop(dataset, nvis, nhid, num_epochs, debug_level=0):
    seed = 188229
    n_inference_steps = 5
    num_examples = dataset.num_examples
    batch_size = num_examples

    train_loop_stream = Flatten(DataStream.default_stream(
        dataset=dataset,
        iteration_scheme= #Repeat(
          SequentialScheme(dataset.num_examples, batch_size)
         #, n_inference_steps)
#            ShuffledScheme(dataset.num_examples, batch_size), n_inference_steps))
            ), which_sources=('features',))

    model_brick = FivEM(
        nvis=nvis, nhid=nhid, epsilon=.001, batch_size=batch_size,
        weights_init=IsotropicGaussian(0.1), biases_init=Constant(0),
        noise_scaling=1, debug=debug_level, lateral_x=False, lateral_h=False,
        n_inference_steps=n_inference_steps)
    model_brick.initialize()

    x = tensor.matrix('features')

    cost = model_brick.cost(x)
    computation_graph = ComputationGraph([cost])
    model = Model(cost)
    #step_rule = Adam(learning_rate=2e-5, beta1=0.1, beta2=0.001, epsilon=1e-8,
    #                 decay_factor=(1 - 1e-8))
    step_rule = Momentum(learning_rate=1e-2,momentum=0.95)
    #step_rule = AdaDelta()
    #step_rule = RMSProp(learning_rate=0.01)
    #step_rule = AdaGrad(learning_rate=1e-4)
    algorithm = GradientDescent(
        cost=cost, parameters=computation_graph.parameters, step_rule=step_rule)
    algorithm.add_updates(computation_graph.updates)

    extensions = [
        Timing(),
        FinishAfter(after_n_epochs=num_epochs),
        TrainingDataMonitoring([cost]+computation_graph.auxiliary_variables,
                               after_batch=False, after_epoch=False,
                               every_n_epochs=1),
        Printing(after_epoch=False, every_n_epochs=1,after_batch=False),
        # Checkpoint(path="./fivem.zip",every_n_epochs=10,after_training=True)
    ]
    main_loop = MainLoop(model=model, data_stream=train_loop_stream,
                         algorithm=algorithm, extensions=extensions)
    return main_loop


import matplotlib.pyplot as mp
def plot_generated_samples(x,data, data_category = 'toy', plot_config = None):
    if data_category is 'toy':
        mp.hold(True)
        fig=mp.figure()
        #mp.plot(x[:,0],x[:,1],'bo')
        #mp.show()
        #pdb.set_trace()
        n=min(100,data.shape[0]) 
        mp.plot(x[:,0],x[:,1],'bo')
        #mp.draw()
        mp.plot(data[:n,0],data[:n,1],'ro')
        mp.draw()
        mp.axes().set_aspect('equal')
        #pl.quiver(prev_x[:,0],prev_x[:,1],x[:,0]-prev_x[:,0],x[:,1]-prev_x[:,1])
        #pl.show()
        mp.show()

def plot_energy_surface(model):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt


    (x1,x2) = numpy.meshgrid(numpy.arange(-.5,.5,.05),numpy.arange(-.5,.5,.05))
    x = shared_floatx(numpy.vstack((x1.flatten(),x2.flatten())).T)
    h = shared_floatx(numpy.zeros((x.get_value().shape[0],model.nhid)))
    map_f = theano.function([],updates=OrderedDict([(h,model.map_update(x,h))]))
    energy_f = theano.function([],[model.energy(x,h)])
    
    for i in range(100):
        map_f()
    (E_,) = energy_f()
    E_ = E_.reshape(x1.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x1, x2, E_, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    ax.set_zlim(numpy.min(E_), numpy.max(E_))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show() 
    plt.savefig('E.png')


def dataset_from_data_stream(data_stream):
    if hasattr(data_stream, 'dataset'):
        return data_stream.dataset
    else:
        return dataset_from_data_stream(data_stream.data_stream)


def show_samples(samples, sample_shape):
    from matplotlib import pyplot
    from matplotlib import cm

    nrows, ncols = 10, 10
    samples = samples[:nrows * ncols].reshape((-1,) + sample_shape)
    figure, axes = pyplot.subplots(nrows=nrows, ncols=ncols)
    for n, (i, j) in enumerate(itertools.product(range(nrows),
                                                 range(ncols))):
        ax = axes[i][j]
        ax.axis('off')
        ax.imshow(samples[n], cmap=cm.Greys_r, interpolation='nearest')
    pyplot.show()

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR,filename="train_emdsm.log")
    parser = ArgumentParser("Train a FivEM movel.")
    parser.add_argument("-d", "--dataset", type=str, choices=('toy', 'mnist'),
                        dest="which_dataset", default="toy",
                        help="Which dataset to use.")
    parser.add_argument("--nhid", type=int, default=2,
                        help="Number of hidden units.")
    parser.add_argument("--nepochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=188229, help="RNG seed.")
    parser.add_argument("--debug", type=int, default=0, help="Debugging level.")
    parser.add_argument("--reload", dest="main_loop_path", type=str,
                        default=None, help="Reload a pickled main loop.")
    args = parser.parse_args()

    if args.main_loop_path:
        main_loop = load(args.main_loop_path)
        dataset = dataset_from_data_stream(data_stream)
        which_dataset = 'toy' if 'Toy' in str(dataset.__class__) else 'mnist'
    else:
        which_dataset = args.which_dataset
        if which_dataset == "toy":
            batch_size = 2
            num_examples = batch_size
            mean = numpy.array([0, 0])
            covariance_matrix = 0.1 * numpy.array([[3.0, 1.5],
                                                   [1.5, 1.0]])
            dataset = Toy2DGaussianDataset(
                 mean, covariance_matrix, num_examples, squash=True,
                 rng=numpy.random.RandomState(args.seed))
            print "data cov:"
            data = dataset.indexables[0]
            print(numpy.cov(data, rowvar=0))
            print dataset.indexables[0]
            nvis = len(mean)
            if num_examples<=10:
                print "data x:",data 
        else:
            dataset = MNIST(("train",), sources=('features',))
            num_examples = dataset.num_examples
            batch_size = num_examples
            nvis = 784
        main_loop = create_main_loop(dataset, nvis, args.nhid,args.nepochs,args.debug)
    main_loop.run()
    model, = main_loop.model.top_bricks

    if which_dataset == 'toy':
       print "show energy function" 
       plot_energy_surface(model)
    
    print "generate samples"
    n_generated = 1000
    x = shared_floatx_zeros((n_generated, model.nvis))
    h = shared_floatx_zeros((n_generated, model.nhid))
    h.set_value(numpy.random.normal(0,scale=0.5,size=(n_generated,model.nhid)))
    model.noise_scaling=1
    new_x, new_h = model.langevin_update(x, h, update_x=True)
    generate_f = theano.function(
        inputs=[], updates=OrderedDict([(x, new_x), (h, new_h)]))

    for i in range(1000):
        generate_f()
    xx=x.get_value()

    if which_dataset == 'toy':
        print(numpy.cov(xx, rowvar=0))
        plot_generated_samples(xx,dataset.indexables[0])
    else:
        show_samples(samples=xx, sample_shape=(28, 28))
