import theano, os, time, pickle
import numpy as np
from theano import tensor as T
from theano import printing
#from util import *
import pdb
# from theano.tensor import nnet
from theano.tensor.nnet import softplus,sigmoid
from theano.sandbox.rng_mrg import MRG_RandomStreams

# for interactive things
import matplotlib.pyplot as mp
import pylab as pl
import matplotlib.cm as cm


def sharedX(x) : return theano.shared( theano._asarray(x, dtype=theano.config.floatX) ) 
def softplus(x) : return T.nnet.softplus(x)
def relu(x) : return x * (x > 1e-15)
def rho(x) : return relu(x+0.5) - relu(x-0.5)

debug_printing = False

def pp(s,x) : 
    if debug_printing:
       return printing.Print(s)(x)
    return x

 
RNG = MRG_RandomStreams(max(np.random.RandomState(1364).randint(2 ** 15), 1))
def gaussian(x, std, rng=RNG) : return x + rng.normal(std=std, size=x.shape, dtype=x.dtype)

class adam(object):
      def __init__(self, alpha=0.001,beta1=0.9,beta2=.999,epsilon=1e-8,lambd=(1-1e-8)):
          self.alpha=alpha
          self.beta1=beta1
          self.beta2=beta2
          self.epsilon=epsilon
          self.lambd=lambd
          self.set_up = False

      def set_updates(self, params, cost):
          self.set_up = True
          self.auxs = []
          for i in xrange(len(params)):
              self.auxs.append(self.initialize_aux(params[i].get_value().shape))
          self.updates = []
          for (param, aux) in zip(params, self.auxs):
              [new_param, new_aux] = self.update(param, T.grad(cost, param), aux) 
              self.updates.append((param, new_param))
              for i_ in range(len(aux)):
                  assert aux[i_].ndim == new_aux[i_].ndim
                  self.updates.append((aux[i_],new_aux[i_]))

      def get_updates(self):
          if not self.set_up:
              raise ValueError("call set_updates before doing updates!")
          return self.updates
 
      def update(self,theta,gradient,aux):
          (m,v,t)=aux
          new_t = t+1.
          beta_t = self.beta1*T.pow(self.lambd,(t-1.))
          new_m = beta_t*m + (1-beta_t)*gradient
          mm = m*(1./(1.-beta_t))
          new_v = self.beta2*v + (1-self.beta2)*gradient*gradient
          vv = self.epsilon+T.sqrt(new_v*(1./(1.-self.beta2)))
          return (theta-self.alpha*mm/vv, (new_m,new_v,new_t))

      def initialize_aux(self,shape):
          m = sharedX(np.zeros(shape))
          v = sharedX(np.zeros(shape))
          t = sharedX(np.zeros(()))
          return (m,v,t)


class sgd(object):
      def __init__(self, lrate=0.01):
          self.lrate=lrate
          self.set_up = False

      def set_updates(self, params, cost):
          self.set_up = True
          self.auxs = []
          for i in xrange(len(params)):
              self.auxs.append(self.initialize_aux(params[i].get_value().shape))
          self.updates = []
          for (param, aux) in zip(params, self.auxs):
              [new_param, new_aux] = self.update(param, T.grad(cost, param), aux) 
              self.updates.append((param, new_param))
              for i_ in range(len(aux)):
                  assert aux[i_].ndim == new_aux[i_].ndim
                  self.updates.append((aux[i_],new_aux[i_]))

      def get_updates(self):
          if not self.set_up:
              raise ValueError("call set_updates before doing updates!")
          return self.updates
 
      def update(self,theta,gradient,aux):
          return (theta-self.lrate*gradient, aux)

      def initialize_aux(self,shape):
          return ()

def get_ll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(numpy.ceil(float(len(inds)) / batch_size))

    times = []
    lls = []
    for i in range(n_batches):
        begin = time.time()
        ll = parzen(x[inds[i::n_batches]])
        end = time.time()
        times.append(end-begin)
        lls.extend(ll)

        #if i % 10 == 0:
        #    print i, numpy.mean(times), numpy.mean(nlls)

    return numpy.array(lls)

def displaynetwork(data, nx, ny, width) :
    i = 0
    data_ = np.zeros((nx*width, ny*width))
    for x in range(nx) :
        for y in range(ny) :
            data_[ x*width : (x+1)*width, y*width : (y+1)*width ] = data[i].reshape(width,width)#/data[i].min
            i += 1
    #plt.rcParams['figure.figsize'] = (16,16)
    mp.imshow(data_, cmap = cm.Greys_r)
    mp.savefig('test.png')

def plot_generated_samples(prev_x,x,data, data_category = 'toy', plot_config = None):
    if data_category is 'toy':
        mp.hold(True)
        fig=mp.figure()
        #mp.plot(x[:,0],x[:,1],'bo')
        #mp.show()   
        n=min(100,data.shape[0]) 
        mp.plot(data[:n,0],data[:n,1],'ro')
        mp.draw()
        mp.plot(x[:,0],x[:,1],'bo')
        mp.draw()
        mp.axes().set_aspect('equal')
        #pl.quiver(prev_x[:,0],prev_x[:,1],x[:,0]-prev_x[:,0],x[:,1]-prev_x[:,1])
        #pl.show()
        mp.show()

    if data_category is 'mnist':
        (nx, ny, width) = plot_config
        displaynetwork(x, nx, ny, width) 

class LatentEnergyFn(object):
      def __init__(self,sigma=0.01,nx=2,nh=1,corrupt_factor=1.,delta_factor=2-np.sqrt(2.)):
          self.sigma=sigma
          self.nx=nx
          self.nh=nh
          self.x = T.matrix()
          self.x_n = T.matrix()
          self.h = T.matrix()
          self.h_n = T.matrix()
          self.delta_factor=delta_factor
          self.corrupt_factor=corrupt_factor
          # subclasss must define 
          #  self.E as a function mapping Theano variable for the state to a Theano variable for the energy
          #  self.param as a Theano shared variable for the parameters

      def set_rest(self):
          self.dEdh = T.grad(self.E,self.h)
          self.dEdx = T.grad(self.E,self.x)
          self.Rh = pp("Rh:",self.h - self.sigma**2 * self.dEdh)
          self.Rx = pp("Rx:",self.x - self.sigma**2 * self.dEdx)
          self.dEdh_n = pp("dEdh_n:",T.grad(self.E_n, self.h_n))
          self.dEdx_n = pp("dEdx_n:",T.grad(self.E_n, self.x_n))
          self.Rh_n = pp("Rh_n:",self.h_n - self.sigma**2 * self.dEdh_n)
          self.Rx_n = pp("Rx_n:",self.x_n - self.sigma**2 * self.dEdx_n)
          self.delta_x = pp("delta_x:",self.Rx_n - self.x)
          self.delta_h = pp("delta_h:",self.Rh_n - self.h)
          self.new_x = pp("new_x:",self.x + self.delta_factor*self.delta_x)
          self.new_h = pp("new_h:",self.h + self.delta_factor*self.delta_h)
          self.new_x_n = pp("new_x_n:",self.new_x + gaussian(0.*self.x, 1)*self.sigma*self.corrupt_factor)
          self.new_clamped_x_n = pp("new_clamped_x_n:",self.x + gaussian(0.*self.x, 1)*self.sigma*self.corrupt_factor)
          self.new_h_n = pp("new_x_n:",self.new_h + gaussian(0.*self.h, 1)*self.sigma*self.corrupt_factor)

      def penalty(self):
          return 0

class GaussianEnergy(LatentEnergyFn):
      def __init__(self,nx,nh,sigma=0.01,inithsigma=0.1,initxsigma=0.1,initwsigma=0.1):
          super(GaussianEnergy, self).__init__(sigma,nx,nh)
          self.hprec_pre=sharedX(np.random.normal(0,inithsigma,nh))
          self.hprec=sigmoid(self.hprec_pre)
          self.xprec_pre=sharedX(np.random.normal(0,initxsigma,nx))
          self.xprec=sigmoid(self.xprec_pre)
          self.w=sharedX(np.random.normal(0,initwsigma,(nx,nh)))
          self.params=[self.w,self.xprec_pre,self.hprec_pre]

          self.E = pp("E:",T.sum(T.dot(self.x*self.x,self.xprec)+T.dot(self.h*self.h,self.hprec) \
                          -T.sum(self.h*T.dot(self.x,self.w),axis=1)))

          self.E_n = pp("E_n:",T.sum(T.dot(self.x_n*self.x_n,self.xprec)+T.dot(self.h_n*self.h_n,self.hprec) \
                              -T.sum(self.h_n*T.dot(self.x_n,self.w),axis=1)))

          self.set_rest()

          self.monitor = theano.function([], [self.hprec, self.xprec, self.w])

      def penalty(self):
          
          self.determinant = self.xprec[0]*self.xprec[1]*self.hprec[0]-0.25*(self.xprec[0]*self.w[1,0]**2+self.xprec[1]*self.w[0,0]**2)
          #pdb.set_trace()
          return 0.001*softplus(5*(0.1-self.determinant)) + 0.01*softplus(100*(0.01- self.determinant))

      def params_monitor(self):
          return self.monitor()


class NeuroEnergy(LatentEnergyFn):
      def __init__(self, nx, nh, sigma=0.01, inithsigma=.1, initxsigma=.1, initwsigma=.1, corrupt_factor=1.):
         super(NeuroEnergy, self).__init__(sigma, nx, nh,corrupt_factor) 
         self.pp_h = sharedX(np.abs(np.random.normal(0,inithsigma,nh)))
         self.p_h = sigmoid(self.pp_h)
         self.b_h = sharedX(np.zeros(nh))
         self.pp_x = sharedX(np.abs(np.random.normal(0,initxsigma,nx)))
         self.p_x = sigmoid(self.pp_x)
         self.b_x = sharedX(np.zeros(nx))
         r = initwsigma/max(nx,nh)
         self.w = sharedX(np.random.uniform(-r,r,(nx,nh)))
         #self.w = sharedX([[3.],[3.]])
         self.params = [self.w, self.b_h, self.b_x, self.pp_h, self.pp_x]

         self.E_fn = lambda x,h,b_h,b_x,w,p_h,p_x: \
                       0.5*(T.dot(p_x,(x*x).sum(axis=0)) + T.dot(p_h,(h*h).sum(axis=0))) \
                        -T.sum(rho(h)*T.dot(rho(x),self.w)) \
                       - T.dot(rho(h), b_h).sum() - T.dot(rho(x), b_x).sum()
         # /(2*sigma**2) 

         self.E = pp("E:",self.E_fn(self.x,self.h,self.b_h,self.b_x,self.w,self.p_h,self.p_x))
         self.E_n = pp("E_n:",self.E_fn(self.x_n,self.h_n,self.b_h,self.b_x,self.w,self.p_h,self.p_x))

         self.set_rest()
         self.monitor = theano.function([], self.params)

      def params_monitor(self):
          return ["w","b_h","b_x","pp_h","pp_x",self.monitor()]


class EMinferencer(object):
      def __init__(self,energyfn):
          self.energyfn=energyfn


class LangevinEMinferencer(EMinferencer):
      def __init__(self, energyfn, epsilon=0.1, n_inference_it=3): # ,corrupt_factor=1):
          super(LangevinEMinferencer, self).__init__(energyfn)
          self.n_inference_it=n_inference_it

          self.set_infer = False

      def set_inference(self, observed_x, generated_x, h, x_n, h_n, batchsize):
          self.set_infer = True
          self.index = T.iscalar()
          self.update_h = theano.function(
              [self.index], [self.energyfn.new_h],
              updates = {(h, self.energyfn.new_h),(x_n, self.energyfn.new_clamped_x_n), (h_n, self.energyfn.new_h_n)},
              givens = {self.energyfn.x : observed_x[self.index*batchsize : (self.index+1)*batchsize], self.energyfn.h : h,
                        self.energyfn.x_n : x_n, self.energyfn.h_n : h_n})
          self.generate_s = theano.function([], [],
              updates = {(generated_x, self.energyfn.new_x), (x_n, self.energyfn.new_x_n),
                         (h, self.energyfn.new_h), (h_n, self.energyfn.new_h_n)},
              givens = {self.energyfn.x_n : observed_x, self.energyfn.x : generated_x,
                        self.energyfn.h : h, self.energyfn.x_n : x_n, self.energyfn.h_n : h_n})


      def inference_h(self, ind):
          if not self.set_infer:
             raise ValueError("Call set_inference before doing inference!")
          #for n_init in xrange(self.n_inference_it):
          (h,)=self.update_h(ind)
          #print "t=",n_init,"h=",h[0]

      def reset_h(self, h, h_new):
          if h.get_value().shape != h_new.shape:
             raise ValueError("Shape mismatch when reseting h!")
          h.set_value(h_new)

      def generate_step(self):
          return self.generate_s()

class EMmodels(object):
      def __init__(self):
          pass

      def mainloop(self):
          pass

      def set_monitor(self):
          pass 


class EMdsm(EMmodels):
      def __init__(self, x, minibatchsize, energyfn, optimizer, inferencer, n_params_update_it = 1):
          super(EMdsm, self).__init__()
          self.x = x
          self.batchsize = minibatchsize
          self.n_params_update_it = n_params_update_it
          self.n_batch = x.get_value().shape[0]/self.batchsize 
          self.energyfn = energyfn
          self.h = sharedX(np.zeros((self.batchsize, self.energyfn.nh)))
          self.generated_x = sharedX(np.random.normal(0,0.3,((self.batchsize, self.energyfn.nx))))
          self.x_n = sharedX(self.generated_x.get_value())
          self.h_n = sharedX(self.h.get_value())

          # set cost
          self.reconstruction_cost = T.sum(self.energyfn.delta_x**2)
          self.cost = self.reconstruction_cost + T.sum(self.energyfn.delta_h**2) + self.energyfn.penalty()

          # set optimizer
          self.optimizer = optimizer
          self.optimizer.set_updates(self.energyfn.params, self.cost)

          # set inferencer
          self.inferencer = inferencer
          self.inferencer.set_inference(self.x, self.generated_x, self.h, self.x_n, self.h_n, minibatchsize) 

          # define parameter update function
          self.index = T.iscalar()
          self.update_p_names = ["cost","reconstruction_cost","E"]
          self.update_p = theano.function(
               [self.index], [self.cost/minibatchsize,self.reconstruction_cost/minibatchsize,self.energyfn.E/minibatchsize],
               updates = self.optimizer.get_updates(),
               givens = {self.energyfn.x : x[self.index*self.batchsize : (self.index+1)*self.batchsize], 
                         self.energyfn.h : self.h, self.energyfn.x_n : self.x_n, self.energyfn.h_n : self.h_n})
          self.costs = theano.function(
               [self.index], [self.cost/minibatchsize,self.reconstruction_cost/minibatchsize,self.energyfn.E/minibatchsize],
               givens = {self.energyfn.x : x[self.index*self.batchsize : (self.index+1)*self.batchsize], 
                         self.energyfn.h : self.h, self.energyfn.x_n : self.x_n, self.energyfn.h_n : self.h_n})

          # may be useful for plotting the shape of the score function
          self.dEdh_ = theano.function([self.index],[self.energyfn.dEdh],
               givens = {self.energyfn.x : x[self.index*self.batchsize : (self.index+1)*self.batchsize], self.energyfn.h : self.h})
          self.dEdx_ = theano.function([self.index],[self.energyfn.dEdx],
               givens = {self.energyfn.x : x[self.index*self.batchsize : (self.index+1)*self.batchsize], self.energyfn.h : self.h})

          # return monitoring values
          # e.g. monitoring "energy"
          self.monitor_values = theano.function(
               [self.index], [self.energyfn.E],
               givens = {self.energyfn.x : x[self.index*self.batchsize : (self.index+1)*self.batchsize], self.energyfn.h : self.h})
               
      # seems not used
      def reset_x(self, x):
          self.x = x

      def update_params(self, ind):
          values = []
          for t in xrange(self.n_params_update_it):
              v = self.update_p(ind)
              values.append(v)
          return v,values

      def mainloop(self, max_epoch = 100, detailed_monitoring=False, burn_in=10, update_params_during_inference=0, plot_every=1000, plot_data_category = 'toy', plot_config = None):
         try:
           for e in xrange(max_epoch):
              values = []
              costs = np.zeros(len(self.update_p_names))
              for k in xrange(self.n_batch):
                   # initial h is the state from the previous minibatch, which is probably better than random or 0
                   #self.inferencer.reset_h(self.h, np.zeros((self.batchsize, self.energyfn.nh)))
                   if detailed_monitoring: 
                      print "costs (||dE/ds||^2,||dE/dx||^2,E) before inference:    ",self.costs(k)
                      print self.h.get_value()[0,:]
                      print "params=",self.energyfn.params_monitor()
                   #pdb.set_trace()
                   for t in xrange(self.inferencer.n_inference_it):
                       self.inferencer.inference_h(k)
                       if update_params_during_inference>0 and t%update_params_during_inference == update_params_during_inference-1: #and t>self.inferencer.n_inference_it/2 
                          (last,list)=self.update_params(k)
                          if detailed_monitoring: 
                             print "params=",self.energyfn.params_monitor()
                   if detailed_monitoring: 
                      print "costs (||dE/ds||^2,||dE/dx||^2,E) after inference:     ",self.costs(k)
                      print self.h.get_value()[0,:]
                   if update_params_during_inference==0:
                      (last,list) = self.update_params(k)
                   costs += last
                   values.append(list)
                   if detailed_monitoring: print "costs (||dE/ds||^2,||dE/dx||^2,E) after params update: ",self.costs(k)
              costs *= 1./self.n_batch
              if e % 100 == 0:
                  print "epoch = ", e
                  #print self.update_p_names,costs,"params=",self.energyfn.params_monitor()
                  print self.update_p_names,costs
                  #self.print_monitor()
                  if e % plot_every == 0:
                     previous_x = self.generated_x.get_value()
                     minx=np.min(previous_x)
                     maxx=np.max(previous_x)
                     self.generated_x.set_value(np.random.uniform(minx,maxx,((self.batchsize, self.energyfn.nx))))
                     previous_x = self.generated_x.get_value()
                     for t in range(burn_in):
                        self.inferencer.generate_step()
                     new_x=self.generated_x.get_value()
                     plot_generated_samples(previous_x,new_x,self.x.get_value(), plot_data_category, plot_config)
                  #mp.show()
         except (KeyboardInterrupt, EOFError):
            pass
         for t in range(burn_in):
            self.inferencer.generate_step()
         new_x=self.generated_x.get_value()
         plot_generated_samples(previous_x,new_x,self.x.get_value(), plot_data_category, plot_config)

      def monitor(self):
          [E, w_h, w_x, w] = self.energyfn.params_monitor(self)

          nx = self.energyfn.nx
          nh = self.energyfn.nh

          precision = np.zeros((nx+nh, nx+nh))
          precision[:nx, :nx] = np.eye(nx)*w_x
          precision[nx:nx+nh, nx:nx+nh] = np.eye(nh)*w_h
          precision[:nx, nx:nx+nh] = w/2
          precision[nx:nx+nh, :nx] = w.T/2
          
          precision = precision*2
          cov = np.linalg.inv(precision) 
          params = [w_h, w_x, w]

          return precision, cov, params

#      def print_monitor(self):
#          [precision, cov, params] = self.monitor()
#          print "model precision = "
#          print precision
#          print "model cov = "
#          print cov
#          print "model params = "
#          print params

def gaussian_mixture_sample(means,covariances,weights,n):
    means = np.array(means)
    ncomp=means.shape[0]
    d=means.shape[1]
    x=np.zeros((n,d))
    components = np.argmax(np.random.multinomial(1,weights,n),axis=1)
    for i in range(n):
        x[i,:] = np.random.multivariate_normal(means[components[i]],covariances[components[i]],1)
    return x

def exp1():
    # information about x
    # TOY GAUSSIAN DATA 2D
    toy_num = 1000
    toy_mean = [0, 0]
    toy_nstd = [[0.1, .1],[.07, .1]]
    #toy_nstd = [[2]]
    x=np.random.multivariate_normal(toy_mean, toy_nstd, toy_num)
    maxx=np.max(np.abs(x))
    train_x = sharedX(x/maxx)
    #mp.hold(True)
    #mp.figure(1)
    x=train_x.get_value()
    #mp.plot(x[:,0],x[:,1],'bo')
    #mp.show()
    max_epoch = 10000
    batchsize = 100
    nx, nh = 2, 3
    sigma = 1
    #energyfn = GaussianEnergy(nx, nh, sigma=sigma)
    energyfn = NeuroEnergy(nx, nh, sigma=sigma, corrupt_factor=1)
    opt = adam()
    #opt = sgd(.1)
    inferencer = LangevinEMinferencer(energyfn, epsilon=0.25/(sigma*sigma), 
                                      n_inference_it=10)
    model = EMdsm(train_x, batchsize, energyfn, opt, inferencer)
    model.mainloop(max_epoch)

def exp2():
    # information about x
    # gaussian mixture in 2D
    n_examples = 1000
    means = [[-1.5, 0],[0,4],[0,0]]
    covs = [[[0.02, -.3],[.11, -.3]],[[1.01,.99],[.99,1.01]],[[1.05,-.95],[-.95,1.05]]]
    weights = [.15,.5,.35]
    x=gaussian_mixture_sample(means,covs,weights,n_examples)
    maxx=np.max(np.abs(x))
    train_x = sharedX(x/maxx)
    #mp.hold(True)
    #mp.figure(1)
    x=train_x.get_value()
    #mp.plot(x[:,0],x[:,1],'bo')
    #mp.show()
    max_epoch = 20000
    batchsize = 100
    nx, nh = 2, 10
    sigma = 0.1

    #energyfn = GaussianEnergy(nx, nh, sigma=sigma)
    energyfn = NeuroEnergy(nx, nh, sigma=sigma, corrupt_factor=0.1)
    #opt = adam()
    opt = sgd(.0001)
    inferencer = LangevinEMinferencer(energyfn, epsilon=0.25/(sigma*sigma), 
                                      n_inference_it=1)
    model = EMdsm(train_x, batchsize, energyfn, opt, inferencer)
    model.mainloop(max_epoch,update_params_during_inference=0,detailed_monitoring=False, burn_in=10, plot_every=5000)


def exp_mnist():
    # information about x
    # MNIST
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = np.load('/data/lisatmp3/saizheng/problems/targetprop/mnist.pkl')

    def prep(x):
        # just add any preprocess you want
        return sharedX(x/np.max(np.abs(x)))

    train_x, valid_x, test_x = prep(train_x), prep(valid_x), prep(test_x)

    x=train_x.get_value()
    #mp.plot(x[:,0],x[:,1],'bo')
    #mp.show()
    max_epoch = 20000
    batchsize = 100
    nx, nh = 784, 400
    sigma = 0.1

    #energyfn = GaussianEnergy(nx, nh, sigma=sigma)
    energyfn = NeuroEnergy(nx, nh, sigma=sigma, corrupt_factor=0.1)
    #opt = adam()
    opt = sgd(.0001)
    inferencer = LangevinEMinferencer(energyfn, epsilon=0.25/(sigma*sigma), 
                                      n_inference_it=1)
    model = EMdsm(train_x, batchsize, energyfn, opt, inferencer)
    model.mainloop(max_epoch,update_params_during_inference=0,
                   detailed_monitoring=False,
                   burn_in=10,
                   plot_every=5000,
                   plot_data_category='mnist',
                   plot_config = [10, 10, 28])



def plot_energy():
    # information about x
    # TOY GAUSSIAN DATA 2D
    toy_num = 1000
    toy_mean = [0, 0]
    toy_nstd = [[3, 1.5],[1.5, 1]]
    #toy_nstd = [[2]]
    x=np.random.multivariate_normal(toy_mean, toy_nstd, toy_num)
    maxx=np.max(np.abs(x))
    train_x = sharedX(x/maxx)

    max_epoch = 1000
    batchsize = 256
    nx, nh = 2, 3
    sigma = 1
    #energyfn = GaussianEnergy(nx, nh, sigma=0.005)
    energyfn = NeuroEnergy(nx, nh, sigma=sigma,corrupt_factor=0.01)
    opt = adam()
    inferencer = LangevinEMinferencer(energyfn, epsilon=0.25/(sigma*sigma), n_inference_it=3, corrupt_factor=0.1)
    model = EMdsm(train_x, batchsize, energyfn, opt, inferencer)
    model.mainloop(max_epoch)


    # grid data
    x1 = np.arange(-3, 3, 0.2)
    x2 = np.arange(-3, 3, 0.2)
    x1_, x2_ = np.meshgrid(x1, x2)
    grid_shape = x1_.shape
    x1__ = x1_.reshape((x1_.shape[0]*x1_.shape[1]))
    x2__ = x2_.reshape((x2_.shape[0]*x2_.shape[1]))
    x_grid = sharedX(np.concatenate((x1__.reshape(x1__.shape[0], 1), x2__.reshape(x2__.shape[0], 1)), axis=1))
    num_grid = x1__.shape[0]   
 
    inferencer_plot = LangevinEMinferencer(energyfn, epsilon=0.56, n_inference_it=3)
    model_plot = EMdsm(x_grid, 1, energyfn, opt, inferencer_plot)
    E = []
    for ind in xrange(num_grid):
        model_plot.inferencer.reset_h(model_plot.h, np.zeros((model_plot.batchsize, model_plot.energyfn.nh)))
        model_plot.inferencer.inference_h(ind)
        E.append(model_plot.monitor_values(ind))

    E_ = np.exp(-np.asarray(E).reshape(grid_shape))

    # plot
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x1_, x2_, E_, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    ax.set_zlim(-1., 4.)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show() 
    plt.savefig('test.png')

    
if __name__ == "__main__":
    #exp_mnist()
    exp2() 
    #plot_energy()

