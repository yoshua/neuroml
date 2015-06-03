import theano, os, time, pickle
import numpy as np
from theano import tensor as T
#from util import *
import pdb
# from theano.tensor import nnet
from theano.tensor.nnet import softplus, sigmoid
from theano.sandbox.rng_mrg import MRG_RandomStreams

# for interactive things
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mp
import pylab as pl

def sharedX(x) : return theano.shared( theano._asarray(x, dtype=theano.config.floatX) ) 
def softplus(x) : return T.nnet.softplus(x)
def relu(x) : return x * (x > 1e-15)
def rho(x) : return relu(x+0.5) - relu(x-0.5)
 
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


def plot_generated_samples(prev_x,x,data):
    mp.hold(True)
    fig=mp.figure()
    #mp.plot(x[:,0],x[:,1],'bo')
    #mp.show()   
    n=min(1000,data.shape[0]) 
    mp.plot(data[:n,0],data[:n,1],'ro')
    mp.draw()
    mp.plot(x[:,0],x[:,1],'bo')
    mp.draw()
    mp.axes().set_aspect('equal')
    #pl.quiver(prev_x[:,0],prev_x[:,1],x[:,0]-prev_x[:,0],x[:,1]-prev_x[:,1])
    #pl.show()
    mp.show()
    mp.savefig('test.png')



class LatentEnergyFn(object):
      def __init__(self,sigma=0.01,nx=2,nh=1,corrupt_factor=1.):
          self.sigma=sigma
          self.nx=nx
          self.nh=nh
          #self.s=T.matrix() # minibatch: n_examples x state_dim
          #self.x=self.s[:,0:nx]
          #self.h=self.s[:,nx:(nx+nh)]
          self.x = T.matrix()
          self.h = T.matrix()
          self.x_n = self.x + gaussian(0.*self.x, 1)*self.sigma*corrupt_factor
          self.h_n = self.h + gaussian(0.*self.h, 1)*self.sigma*corrupt_factor

          # subclasss must define 
          #  self.E as a function mapping Theano variable for the state to a Theano variable for the energy
          #  self.param as a Theano shared variable for the parameters

      def set_R(self):
          self.dEdh = T.grad(self.E,self.h)
          self.dEdx = T.grad(self.E,self.x)
          self.Rh = self.h - self.sigma**2 * self.dEdh
          self.Rx = self.x - self.sigma**2 * self.dEdx

      def set_R_noise(self):
          self.dEdh_n = T.grad(self.E_n, self.h_n)
          self.dEdx_n = T.grad(self.E_n, self.x_n)
          self.Rh_n = self.h_n - self.sigma**2 * self.dEdh_n
          self.Rx_n = self.x_n - self.sigma**2 * self.dEdx_n

      def penality(self):
          return 0

class GaussianEnergy(LatentEnergyFn):
      def __init__(self,nx,nh,sigma=0.01,inithsigma=0.1,initxsigma=0.1,initwsigma=0.1):
          super(GaussianEnergy, self).__init__(sigma,nx,nh)
          self.hprec_pre=sharedX(np.random.normal(0,inithsigma,nh))
          self.hprec=softplus(self.hprec_pre)
          self.xprec_pre=sharedX(np.random.normal(0,initxsigma,nx))
          self.xprec=softplus(self.xprec_pre)
          self.w=sharedX(np.random.normal(0,initwsigma,(nx,nh)))
          self.params=[self.w,self.xprec_pre,self.hprec_pre]

          self.E = T.sum(T.dot(self.x*self.x,self.xprec)+T.dot(self.h*self.h,self.hprec)
                  -T.sum(self.h*T.dot(self.x,self.w),axis=1))

          self.E_n = T.sum(T.dot(self.x_n*self.x_n,self.xprec)+T.dot(self.h_n*self.h_n,self.hprec)
                   -T.sum(self.h_n*T.dot(self.x_n,self.w),axis=1))

          self.set_R()
          self.set_R_noise()

          self.monitor = theano.function([], [self.hprec, self.xprec, self.w])

      def penality(self):
          
          self.determinant = self.xprec[0]*self.xprec[1]*self.hprec[0]-0.25*(self.xprec[0]*self.w[1,0]**2+self.xprec[1]*self.w[0,0]**2)
          #pdb.set_trace()
          return 0.001*softplus(5*(0.1-self.determinant)) + 0.01*softplus(100*(0.01- self.determinant))

      def params_monitor(self):
          return self.monitor()


class NeuroEnergy(LatentEnergyFn):
      def __init__(self, nx, nh, sigma=0.01, inithsigma=1, initxsigma=1, initwsigma=.1, corrupt_factor=1.):
         super(NeuroEnergy, self).__init__(sigma, nx, nh,corrupt_factor) 
         self.pp_h = sharedX(np.abs(np.random.normal(0,inithsigma,nh)))
         self.p_h = sigmoid(self.pp_h)
         self.b_h = sharedX(np.zeros(nh))
         self.pp_x = sharedX(np.abs(np.random.normal(0,initxsigma,nx)))
         self.p_x = sigmoid(self.pp_x)
         self.b_x = sharedX(np.zeros(nx))
         self.w = sharedX(np.random.normal(1.,initwsigma,(nx,nh)))
         #self.w = sharedX([[3.],[3.]])
         self.params = [self.w, self.b_h, self.b_x, self.pp_h, self.pp_x]

         self.E_fn = lambda x,h,b_h,b_x,w,p_h,p_x: \
                       0.5*(T.dot(p_x,(x*x).sum(axis=0)) + T.dot(p_h,(h*h).sum(axis=0))) \
                        -T.sum(rho(h)*T.dot(rho(x),self.w)) \
                       - T.dot(rho(h), b_h).sum() - T.dot(rho(x), b_x).sum()
         # /(2*sigma**2) 

         self.E = self.E_fn(self.x,self.h,self.b_h,self.b_x,self.w,self.p_h,self.p_x)
         self.E_n = self.E_fn(self.x_n,self.h_n,self.b_h,self.b_x,self.w,self.p_h,self.p_x)

         self.set_R()
         self.set_R_noise()
         self.monitor = theano.function([], self.params)

      def params_monitor(self):
          return ["w","b_h","b_x","pp_h","pp_x",self.monitor()]


def exp_direct():
    # information about x
    # TOY GAUSSIAN DATA 2D
    toy_num = 1000
    toy_mean = [0, 0]
    toy_nstd = np.array([[3, 1.5],[1.5, 1]])
    x=np.random.multivariate_normal(toy_mean, toy_nstd, toy_num)
    train_x = sharedX(x)

    epoch = 100000
    batchsize = 100
    samplesize = 200
    nx, nh = 2, 20
    n_batch = train_x.get_value().shape[0]/batchsize
    n_iter_infer = 20
    n_iter_gene = 20

    sigma_dn = 0.1 # noise level for denoising score matching
    sigma_inf = 0.1 # noise level for Langevin MCMC

    #W = sharedX(np.linalg.inv(np.array(toy_nstd))/2)
    index = T.iscalar()
    #E = NeuroEnergy(nx, nh, sigma=sigma_dn, corrupt_factor=1.)
    E = GaussianEnergy(nx, nh, sigma=sigma_dn)
    h = sharedX(np.random.normal(0., 0.1, (batchsize, nh)))

    generated_h = sharedX(np.random.normal(0., 0.1, (samplesize, nh)))
    generated_x = sharedX(np.random.normal(0., 0.1, (samplesize, nx)))

    h_new = E.h - sigma_inf**2/2 * E.dEdh + gaussian(E.h*0., sigma_inf)
    x_new = E.x - sigma_inf**2/2 * E.dEdx + gaussian(E.x*0., sigma_inf)

    cost = T.sum((E.Rh_n - E.h)**2) + T.sum((E.Rx_n - E.x)**2)
    #set optimizer
    opt = sgd(0.001)
    opt.set_updates(E.params, cost) 

    infer_h = theano.function([index], [],
              updates = {(h, h_new)}, givens={E.x : train_x[batchsize*index : batchsize*(index+1)], E.h : h})
    generate_h = theano.function([], [],
                 updates = {(generated_h, h_new)}, givens={E.x : generated_x, E.h : generated_h})
    generate_x = theano.function([], [],
                 updates = {(generated_x, x_new)}, givens={E.x : generated_x, E.h : generated_h})
    reset_h = theano.function([], [],
              updates = {(h, sharedX(np.random.normal(0., 0.1, (batchsize, nh))))})
    reset_generated_h = theano.function([], [],
                        updates = {(generated_h, sharedX(np.random.normal(0., 0.1, (samplesize, nh))))})
    reset_generated_x = theano.function([], [],
                        updates = {(generated_x, sharedX(np.random.normal(0., 0.1, (samplesize, nx))))})
    params_update = theano.function([index], [], 
                    updates = opt.get_updates(), givens = {E.x : train_x[batchsize*index : batchsize*(index+1)], E.h : h})

    for e in xrange(epoch):
        for k in xrange(n_batch):
            reset_h()
            for t in xrange(n_iter_infer):
                infer_h(k)
            params_update(k)

        if e % 100 == 0:
           print "epoch = ", e
           print E.params_monitor()

        if e % 100 == 0:
           reset_generated_x()
           reset_generated_h()
           for t in xrange(n_iter_gene):
               generate_x()
               generate_h()
           x_ = generated_x.get_value()
           plot_generated_samples(x_, x_,train_x.get_value())
    #generated_X = sharedX(np.random.normal(0., 0.1, (1000,nx)))
    #E = T.sum(X*T.dot(X, W))
    #dEdX = T.grad(E, X)
    #X_new = X - sigma**2/2 * dEdX + gaussian(0.*X, sigma)
    #Rh_n = self.h_n - self.sigma**2 * self.dEdh_n 
    #Rx_n = 

    #generate = theano.function([],[],
    #           updates = {(generated_X, X_new)}, givens = {X:generated_X})

    #for k in xrange(10):
    #    generate()

    #x_ = generated_X.get_value()
    #plot_generated_samples(x_,x_,x)

   
if __name__ == "__main__":

    exp_direct()

