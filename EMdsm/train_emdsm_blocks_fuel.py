import logging
from collections import OrderedDict

import numpy
import theano
import pdb
from blocks.algorithms import GradientDescent, Adam, Scale
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.training import SharedVariableModifier
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.utils import shared_floatx_zeros
from blocks.extensions.monitoring import DataStreamMonitoring
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme,ConstantScheme,SequentialScheme
from fuel.datasets import MNIST
from theano import tensor

from emdsm_blocks_fuel import FivEM, Toy2DGaussianDataset, Repeat


def create_main_loop():
    seed = 188229
    batch_size = 1000
    num_epochs = 10000
    num_examples = batch_size
    n_inference_steps = 1
    mean = numpy.array([0, 0])
    covariance_matrix = 0.1*numpy.array([[3.0, 1.5],
                                         [1.5, 1.0]])
    nvis = len(mean)
    nhid = 5

    dataset = Toy2DGaussianDataset(mean, covariance_matrix, num_examples,squash=True,
                                   rng=numpy.random.RandomState(seed))
    #dataset = MNIST(("train",))
    #pdb.set_trace()
    #print dataset.indexables[0]
    train_loop_stream = DataStream(
        dataset=dataset,
        iteration_scheme=#Repeat(
#            ShuffledScheme(dataset.num_examples, batch_size), n_inference_steps))
            SequentialScheme(dataset.num_examples, batch_size))#, n_inference_steps))

    monitoring_stream = DataStream(
        dataset=dataset,
        iteration_scheme=#Repeat(
            SequentialScheme(dataset.num_examples, batch_size))#, n_inference_steps))
            #ShuffledScheme(dataset.num_examples, batch_size), n_inference_steps))

    model_brick = FivEM(
        nvis=nvis, nhid=nhid, epsilon=1e-2, batch_size=batch_size,
        weights_init=IsotropicGaussian(0.1), noise_scaling=1, debug=3,lateral_x=True,lateral_h=True)
    model_brick.initialize()

    x = tensor.matrix('features')

    cost = model_brick.cost(x)
    computation_graph = ComputationGraph([cost])
    model = Model(cost)
    #step_rule = Adam(learning_rate=2e-5, beta1=0.1, beta2=0.001, epsilon=1e-8,
    #                 decay_factor=(1 - 1e-8))
    step_rule = Scale(learning_rate=1e-2)
    algorithm = GradientDescent(
        cost=cost, params=computation_graph.parameters, step_rule=step_rule)
    algorithm.add_updates(computation_graph.updates)

    def update_val(n_it, old_value):
        if n_it % n_inference_steps == 0:
            return 0 * old_value
        else:
            return old_value

    extensions = [
        Timing(),
        FinishAfter(after_n_epochs=num_epochs),
        DataStreamMonitoring([cost]+computation_graph.auxiliary_variables,
                             monitoring_stream, after_batch=False),
        #SharedVariableModifier(
        #    model_brick.h_prev,
        #    update_val,
        #    after_batch=False, before_batch=True),
        #SharedVariableModifier(
        #    model_brick.h,
        #    update_val,
        #    after_batch=False, before_batch=True),
        Printing(after_epoch=False, every_n_epochs=10,after_batch=False)
    ]
    main_loop = MainLoop(model=model, data_stream=train_loop_stream,
                         algorithm=algorithm, extensions=extensions)
    return main_loop


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,filename="train_emdsm.log")
    #logging.basicConfig(level=logging.DEBUG,filename="train_emdsm.log",filemode='w')
    main_loop = create_main_loop()

    main_loop.run()

    model, = main_loop.model.top_bricks
    x = shared_floatx_zeros((1000, model.nvis))
    h = shared_floatx_zeros((1000, model.nhid))
    new_x, new_h = model.langevin_update(x, h, update_x=True)
    f = theano.function(
        inputs=[], updates=OrderedDict([(x, new_x), (h, new_h)]))
    for i in range(1000):
        f()
    print(numpy.cov(x.get_value(), rowvar=0))
