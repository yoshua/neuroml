import logging
from collections import OrderedDict

import numpy
import theano
import pdb
from blocks.algorithms import GradientDescent, Adam, Scale, Momentum, AdaDelta, RMSProp, AdaGrad
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.training import SharedVariableModifier
from blocks.extensions.saveload import Checkpoint
from blocks.serialization import load
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.utils import shared_floatx, shared_floatx_zeros
from blocks.extensions.monitoring import DataStreamMonitoring
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme,ConstantScheme,SequentialScheme
from fuel.transformers import Flatten
from fuel.datasets import MNIST
from theano import tensor

from emdsm_blocks_fuel import FivEM, Toy2DGaussianDataset, Repeat

    
def create_main_loop():
    seed = 188229
    num_epochs = 10000
    n_inference_steps = 10
    dataname = "toy"
    if dataname=="toy":
       batch_size = 2
       num_examples = batch_size
       mean = numpy.array([0, 0])
       covariance_matrix = 0.1*numpy.array([[3.0, 1.5],
                                            [1.5, 1.0]])
       dataset = Toy2DGaussianDataset(mean, covariance_matrix, num_examples,squash=True,
                                      rng=numpy.random.RandomState(seed))
       print "data cov:"
       print(numpy.cov(dataset.indexables[0], rowvar=0))
       print dataset.indexables[0]
       nhid = 4
       nvis = len(mean)
    else:
       dataset = MNIST(("train",), sources=('features',))
       nhid = 100
       num_examples = dataset.num_examples
       batch_size = num_examples
       nvis = 784

    train_loop_stream = DataStream.default_stream(
        dataset=dataset,
        iteration_scheme=SequentialScheme(dataset.num_examples, batch_size)
        #Repeat(
#            ShuffledScheme(dataset.num_examples, batch_size), n_inference_steps))
            #, n_inference_steps)
            )
    monitoring_stream = DataStream.default_stream(
        dataset=dataset,
        iteration_scheme=SequentialScheme(dataset.num_examples, batch_size)
            #Repeat(
            #, n_inference_steps)
            #ShuffledScheme(dataset.num_examples, batch_size), n_inference_steps))
            )

    if dataname=='mnist':
        train_loop_stream = Flatten(train_loop_stream,which_sources=('features',))
        monitoring_stream = Flatten(monitoring_stream,which_sources=('features',))

    model_brick = FivEM(
        nvis=nvis, nhid=nhid, epsilon=.001, batch_size=batch_size,
        weights_init=IsotropicGaussian(0.1), noise_scaling=1, debug=0,lateral_x=False,lateral_h=True)
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
        cost=cost, params=computation_graph.parameters, step_rule=step_rule)
    algorithm.add_updates(computation_graph.updates)

    def update_val(n_it, old_value):
        if n_it % n_inference_steps == 0:
            # return 0 * old_value 
            return old_value#+numpy.random.normal(0,0.05,size=old_value.shape)
        else:
            return old_value

    extensions = [
        Timing(),
        FinishAfter(after_n_epochs=num_epochs),
        DataStreamMonitoring([cost]+computation_graph.auxiliary_variables,
                             monitoring_stream, after_batch=False, every_n_epochs=100),
        #SharedVariableModifier(
        #    model_brick.h_prev,
        #    update_val,
        #    after_batch=False, before_batch=True),
        #SharedVariableModifier(
        #    model_brick.h,
        #    update_val,
        #    after_batch=False, before_batch=True),
        Printing(after_epoch=False, every_n_epochs=100,after_batch=False),
        Checkpoint(path="./",every_n_epochs=100,after_training=True)
    ]
    main_loop = MainLoop(model=model, data_stream=train_loop_stream,
                         algorithm=algorithm, extensions=extensions)
    return main_loop,dataset,dataname


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
        #print "total E=",inferencer.energyfn.E_fn(x,h)
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
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR,filename="train_emdsm.log")
    #logging.basicConfig(level=logging.DEBUG,filename="train_emdsm.log",filemode='w')
    reload = None # "tmpt3YKfp"
    main_loop,dataset,dataname = create_main_loop()
    if reload != None:
        main_loop = load(reload)
    
    main_loop.run()
    model, = main_loop.model.top_bricks

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
    if dataname=="toy":
      print(numpy.cov(xx, rowvar=0))
      plot_generated_samples(xx,dataset.indexables[0])

    #main_loop.run()
    #for i in range(1000):
    #    f()
    #print(numpy.cov(xx, rowvar=0))
    #plot_generated_samples(xx,dataset.indexables[0])
    
