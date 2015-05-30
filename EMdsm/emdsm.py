import theano, os, time, pickle
import numpy as np
from theano import tensor as T
#from util import *
import pdb
# from theano.tensor import nnet
# from theano.tensor.nnet import softplus
from theano.sandbox.rng_mrg import MRG_RandomStreams

def sharedX(x) : return theano.shared( theano._asarray(x, dtype=theano.config.floatX) ) 
def softplus(x) : return T.nnet.softplus(x)
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


class LatentEnergyFn(object):
      def __init__(self,sigma=0.01,nx=2,nh=1):
          self.sigma=sigma
          self.nx=nx
          self.nh=nh
          #self.s=T.matrix() # minibatch: n_examples x state_dim
          #self.x=self.s[:,0:nx]
          #self.h=self.s[:,nx:(nx+nh)]
          self.x = T.matrix()
          self.h = T.matrix()
          self.x_n = self.x + gaussian(0.*self.x, 1)*self.sigma
          self.h_n = self.h + gaussian(0.*self.h, 1)*self.sigma

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


class GaussianEnergy(LatentEnergyFn):
      def __init__(self,nx,nh,sigma=0.01,inithsigma=0.1,initxsigma=0.1,initwsigma=0.1):
          super(GaussianEnergy, self).__init__(sigma,nx,nh)
          self.hprec_pre=sharedX(np.random.normal(0,inithsigma,nh))
          self.hprec=softplus(self.hprec_pre)
          self.xprec_pre=sharedX(np.random.normal(0,inithsigma,nx))
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

      def params_monitor(self):
          return self.monitor()


class EMinferencer(object):
      def __init__(self,energyfn):
          self.energyfn=energyfn


class LangevinEMinferencer(EMinferencer):
      def __init__(self, energyfn, epsilon=0.5, n_inference_it=3):
          super(LangevinEMinferencer, self).__init__(energyfn)
          self.n_inference_it=n_inference_it
          self.new_h = self.energyfn.h - epsilon * self.energyfn.sigma**2 * self.energyfn.dEdh + gaussian(0.*self.energyfn.h,1)*energyfn.sigma
          self.set_infer = False

      def set_inference(self, x, h, batchsize):
          self.set_infer = True
          self.index = T.iscalar()
          self.update_h = theano.function(
              [self.index], [self.new_h],
              updates = {(h, self.new_h)},
              givens = {self.energyfn.x : x[self.index*batchsize : (self.index+1)*batchsize], self.energyfn.h : h})

      def inference_h(self, ind):
          if not self.set_infer:
             raise ValueError("Call set_inference before doing inference!")
          for n_init in xrange(self.n_inference_it):
              self.update_h(ind)

      def reset_h(self, h, h_new):
          if h.get_value().shape != h_new.shape:
             raise ValueError("Shape mismatch when reseting h!")
          h.set_value(h_new)


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

          # set cost
          # TODO: do we need determinant in cost?
          self.cost = T.mean((self.energyfn.Rh_n - self.energyfn.h)**2) \
                      + T.mean((self.energyfn.Rx_n - self.energyfn.x)**2)

          # set optimizer
          self.optimizer = optimizer
          self.optimizer.set_updates(self.energyfn.params, self.cost)

          # set inferencer
          self.inferencer = inferencer
          self.inferencer.set_inference(self.x, self.h, minibatchsize) 

          # define parameter update function
          self.index = T.iscalar()
          self.update_p = theano.function(
               [self.index], [self.cost],
               updates = self.optimizer.get_updates(),
               givens = {self.energyfn.x : x[self.index*self.batchsize : (self.index+1)*self.batchsize], self.energyfn.h : self.h})


      def update_params(self, ind):
          for t in xrange(self.n_params_update_it):
              self.update_p(ind)

      def mainloop(self, max_epoch = 100):
          for e in xrange(max_epoch):
              for k in xrange(self.n_batch):
                   self.inferencer.reset_h(self.h, np.zeros((self.batchsize, self.energyfn.nh)))
                   self.inferencer.inference_h(k)
                   self.update_params(k)
              if e % 100 == 0:
                  print
                  print "epoch = ", e
                  self.print_monitor()
           
      def monitor(self):
          [w_h, w_x, w] = self.energyfn.params_monitor()

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

      def print_monitor(self):
          [precision, cov, params] = self.monitor()
          print "model precision = "
          print precision
          print "model cov = "
          print cov
          print "model params = "
          print params


def exp():
    # information about x
    # TOY GAUSSIAN DATA 2D
    batchsize = 128
    toy_num = batchsize*10
    toy_mean = [0, 0]
    toy_nstd = [[3, 1.5],[1.5, 1]]
    #toy_nstd = [[2]]
    train_x = sharedX(np.random.multivariate_normal(toy_mean, toy_nstd, toy_num))

    max_epoch = 1000000
    nx, nh = 2, 1

    energyfn = GaussianEnergy(nx, nh, sigma=0.01)
    opt = adam()
    inferencer = LangevinEMinferencer(energyfn, epsilon=0.5, n_inference_it=3)
    model = EMdsm(train_x, batchsize, energyfn, opt, inferencer)
    model.mainloop(max_epoch)

    
if __name__ == "__main__":
    exp()

