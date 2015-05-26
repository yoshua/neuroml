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

# TOY GAUSSIAN DATA 1D
toy_num = 10000
toy_nstd = 3
train_x = sharedX(np.random.normal(0, scale = toy_nstd, size = toy_num))

# corruption noise
sigma = 0.001

# TOY GAUSSIAN DATA 2D
toy_num = 10000
toy_mean = [0, 0]
toy_nstd = [[3, 1.5],[1.5, 1]]
#toy_nstd = [[2]]
train_x = sharedX(np.random.multivariate_normal(toy_mean, toy_nstd, toy_num))
dlog2pi = 2*np.log(2*np.pi)

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
              self.auxs.append(self.intialize_aux(params[i].get_value().shape))
          self.updates = []
          for (param, aux) in zip(params, self.auxs):
              [new_param, new_aux] = self.update(param, T.grad(cost, param), aux) 
              self.updates.append((param, new_param))
              for i_ in range(len(aux)):
                  assert aux[i_].ndim == new_aux[i_].ndim
                  self.updates.append((aux[i_],new_aux[i_]))

      def updates():
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

          # subclasss must define 
          #  self.E as a function mapping Theano variable for the state to a Theano variable for the energy
          #  self.param as a Theano shared variable for the parameters

      def set_R(self):
          self.dEdh = T.grad(self.E,self.h)
          self.dEdx = T.grad(self.E,self.x)
          self.Rh = self.h - self.sigma**2 * self.dEdh
          self.Rx = self.x - self.sigma**2 * self.dEdx

      def set_R_noise(self, noise_std):
          self.x_n = self.x + gaussian(0.*self.x, 1)*noise_std
          self.h_n = self.h + gaussian(0.*self.x, 1)*noise_std
          self.dEdh_n = T.grad(self.E_n, self.h_n)
          self.dEdx_n = T.grad(self.E_n, self.x_n)
          self.Rh_n = self.h_n - self.sigma**2 * self.dEdh_n
          self.Rx_n = self.x_n - self.sigma**2 * self.dEdx_n


class GaussianEnergy(LatentEnergyFn):
      def __init__(self,nx,nh,sigma=0.01,inithsigma=0.1,initxsigma=0.1,initwsigma=0.1):
          super().__init__(sigma,nx,nh)
          self.hprec_pre=sharedX(np.random.normal(0,inithsigma,nh))
          self.hprec=softplus(self.hprec_pre)
          self.xprec_pre=sharedX(np.random.normal(0,inithsigma,nx))
          self.xprec=softplus(self.hprec_pre)
          self.w=sharedX(np.random.normal(0,initwsigma,(nx,nh)))
          self.params=[self.w,self.xprec_pre,self.hprec_pre]

          self.E = (T.dot(self.x*self.x,self.xprec)+T.dot(self.h*self.h,self.hprec)
                  -T.sum(self.h*T.dot(self.x,self.w),axis=1)).mean()

          self.E_n = (T.dot(self.x_n*self.x_n,self.xprec)+T.dot(self.h_n*self.h_n,self.hprec)
                   -T.sum(self.h_n*T.dot(self.x_n,self.w),axis=1)).mean()

          self.set_R()
          self.set_R_noise(self.sigma)


class EMinferencer(object):
      def __init__(self,energyfn):
          self.energyfn=energyfn


class LangevinEMinferencer(EMinferencer):
      def __init__(self, energyfn, epsilon=0.1,n_initial_it=3, n_update_it=1):
          self.super(energyfn)
          self.n_initial_it=n_initial_it
          self.n_update_it=n_update_it
          self.new_h = self.energyfn.h - epsilon*self.dEdh + gaussian(0.*self.h,1)*energyfn.sigma
          self.set_infer = False

      def set_inference(self, x, h, batchsize):
          self.set_infer = True
          self.index = T.iscalar()
          self.update_h = theano.function(
              [self.index], [self.new_h],
              updates = {(h, self.new_h)},
              givens = {self.energyfn.x : x[self.index*batchsize, (self.index+1)*batchsize], self.energyfn.h : h]})
           
      def inference_h(self, ind):
          if not self.set_infer:
             raise ValueError("Call set_inference before doing inference!")
          for n_init in xrange(self.n_initial_it):
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

      def monitors(self):
          pass 


class EMdsm(EMmodels):
      def __init__(self, x, minibatchsize, energyfn, optimizer, inferencer, n_params_update):
          self.super()
          self.x = x
          self.batchsize = minibatchsize
          self.n_params_update = n_params_update
          self.n_batch = x.get_value().shape[0]/self.batchsize 
          self.h = sharedX(np.zeros((self.batchsize, self.energyfn.nh)))
          self.energyfn = energyfn

          # set cost
          # TODO: do we need determinant in cost?
          self.cost = T.mean((self.energyfn.Rh_n - self.energyfn.h)**2)
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
               updates = self.optimizer.updates(),
               givens = {self.energyfn.x : x[self.index*batchsize, (self.index+1)*batchsize], self.energyfn.h : h]})

      def update_params(self, ind):
          for t in xrange(self.n_params_update):
              self.update_p(ind)

      def main_loop(max_epoch = 100):
          for e in xrange(max_epoch):
              for k in xrange(self.n_batch):
                   self.inferencer.reset_h(self.h, np.zeros((self.batchsize, self.energyfn.nh)))
                   self.inferencer.inference_h(k)
                   self.update_params(k)


      
def exp():
    # TOY GAUSSIAN DATA 2D
    toy_num = 10000
    toy_mean = [0, 0]
    toy_nstd = [[3, 1.5],[1.5, 1]]
    #toy_nstd = [[2]]
    train_x = sharedX(np.random.multivariate_normal(toy_mean, toy_nstd, toy_num))

    max_epoch = 200
    batchsize = 50
    n_batch = train_x.get_value().shape[0]/batchsize
    nx, nh = 2, 1
    train_h = sharedX(np.zeros((minibatch_size, nh)))

    energyfn = GaussianEnergy(nx, nh) 
    opt = adam()
    
def exp(max_ep = 200, batsize = 10, opt=adam()):
    batch_size = batsize 
    n_batches = toy_num / batch_size 
    max_epochs = max_ep
    train_h = sharedX(np.zeros((batch_size,1)))

    # symbols for trainning
    # initialziation
    w = sharedX(np.random.normal(0, 0.1, (2, 1))) 
    theta_x = sharedX(np.random.normal(0, 0.1, 2))
    w_x = softplus(theta_x) 
    theta_h = sharedX(np.random.normal(0, 0.1))
    w_h = softplus(theta_h)
    w_aux = opt.initialize_aux(w.get_value().shape)
    theta_x_aux = opt.initialize_aux(theta_x.get_value().shape)
    theta_h_aux = opt.initialize_aux(theta_h.get_value().shape)
    cov = sharedX(np.zeros((3,3)))
    precision = sharedX(np.zeros((3,3)))
    #a = sharedX(np.random.normal(0., 0.1, 5))
    #w_x = T.as_tensor_variable([a[0]**2+a[1]**2, a[2]**2])
    #w_h = T.as_tensor_variable(a[3]**2+a[4]**2)
    #w = T.as_tensor_variable([a[1]*a[4], a[2]*a[3]]).reshape((2,1)) 
    
    H = T.matrix() 
    X = T.matrix()
    marginal_precision = T.matrix()
    E = (T.sum(w_x*X*X) + T.sum(w_h*H*H) + T.sum(T.dot(X, T.dot(w, H.transpose()))))/batch_size
    i, e = T.iscalar(), T.iscalar() 
    data = T.matrix()
    determinant = w_x[0]*w_x[1]*w_h-0.25*(w_x[0]*w[1,0]**2+w_x[1]*w[0,0]**2)
    nll_determinant = marginal_precision[0,0]*marginal_precision[1,1]-marginal_precision[1,0]*marginal_precision[0,1]
    avg_nll = 0.5*((T.dot(data,marginal_precision)*data).sum(axis=1).mean()-T.log(nll_determinant)+dlog2pi)


    # "reconstruction" 
    R_h = lambda x,h : h - (2*w_h*h + T.dot(x, w) )*sigma**2 
    R_x = lambda x,h : x - (2*w_x*x + T.dot(h, w.transpose()))*sigma**2 
    eps = 0.1

    #algorithm
    X_0 = X + gaussian(0.*X, 1)*sigma
    H_0 = H + gaussian(0.*H, 1)*sigma

    delta_x = (R_x(X_0, H_0) - X) # X_0 or X?
    delta_h = (R_h(X_0, H_0) - H)

    #dn = H_0 - H
    #dd = delta_h - dn

    H_new = H - eps* (2*w_h*H + T.dot(X, w) ) + gaussian(0.*H_0, 1)*sigma
    #H_new = T.dot(X, w)/(2*w_h)
    H_direct = - T.dot(X, w)/(2*w_h) # + gaussian(0.*H, 1)*T.sqrt(w_h/2)


    #dEdH = (2*w_h*H + T.dot(X, w) )

    # monitoring params
    monitor_params = theano.function([], [w, w_x, w_h])

    # cost, updates
    cost = (T.mean(delta_x**2) + T.mean(delta_h**2))/(sigma**2) + 0.001*softplus(5*(0.1-determinant)) + 0.01*softplus(100*(0.01-determinant))
    updates =[]
    for (param,aux) in [(w,w_aux), (theta_x,theta_x_aux), (theta_h,theta_h_aux)]: 
      (new_param, new_aux) = opt.update(param,T.grad(cost,param),aux)
      updates.append((param, new_param))
      for i_ in range(len(aux)):
          assert aux[i_].ndim == new_aux[i_].ndim
          updates.append((aux[i_],new_aux[i_]))

    # training givens
    givens_train = lambda i : { X : train_x[ i*batch_size : (i+1)*batch_size ], H : train_h }

    # training and testing function
    train_sync = theano.function([i, e], [cost, H_new, H_direct, determinant], 
                                 givens = givens_train(i), on_unused_input='ignore',
                                 updates = updates)

    train_sync_notup = theano.function([i, e], [cost, H_new, H_direct, E,determinant], 
                                       givens = givens_train(i), on_unused_input='ignore')

    marginal_nll = theano.function([data,marginal_precision],[avg_nll])

    print 'epochs train_cost valid_err test_err test_best time'
    sign_error = 0
    det = 0
    xdata=train_x.get_value()
    print "TRUE MODEL marginal NLL=",marginal_nll(xdata,np.linalg.inv(toy_nstd))[0]
    empirical_cov = np.dot(xdata.T,xdata)/toy_num
    print "Empirical covariance = ",empirical_cov
    print "empirical covariance marginal NLL=",marginal_nll(xdata,np.linalg.inv(empirical_cov))[0]
    centered_data=xdata-xdata.mean(axis=0)
    centered_empirical_cov = np.dot(centered_data.T,centered_data)/toy_num
    print "Centered empirical covariance = ",centered_empirical_cov
    print "centered empirical covariance marginal NLL=",marginal_nll(centered_data,np.linalg.inv(centered_empirical_cov))[0]
    # training loop
    if 1:
        min_err = 100
        t = time.time(); monitor = { 'train' : [], 'monitor_params' : [], 'valid' : [], 'test' : [], 'test_ll':[], 'test_ll_base':[] }
        for e in range(1,max_epochs+1):

            monitor['monitor_params'].append(monitor_params())
            [w_, w_x_, w_h_] = monitor['monitor_params'][-1] 
            if e%100==0:
              print "parameters = ",monitor['monitor_params'][-1]

            result = []
            for i in range(n_batches):
                train_h.set_value(np.zeros((batch_size,1)))
                if sign_error>0.8: print "sign error = ",sign_error," # H0 =",h[0,0]
                for t in range(3):
                    [cost, h, hstar, Eng, det] = train_sync_notup(i, e)
                    if sign_error>0.8 and t==0: print "### HSTAR =",hstar[0,0]
                    if sign_error>0.8: print "### H =",h[0,0]
                    #for m in range(50):
                    #    [cost, h, hstar, det] = train_sync(i, e)
                    #    print 'cost: ', cost
                    #pdb.set_trace()
                    train_h.set_value(h)
                    #print h[0]
                    #print 'cost', cost 
                    #print dedh.sum()
                    #print dddn.mean()
                    result = result + [cost]
                sign_error = 0.5*np.mean(np.abs(np.sign(h)-np.sign(hstar)))
                if sign_error>0.8 or det<=0:
                   # print "SIGN=",np.sign(h),np.sign(hstar),0.5*np.abs(np.sign(h)-np.sign(hstar)),sign_error
                   print "H=",hstar[0,0],h[0,0]," %err = ",sign_error, "DET=",det
                train_h.set_value(h)
                [cost, h, hstar, det] = train_sync(i, e)
                result = result + [cost]
            if e%100==0:
              monitor['train'].append( np.array(result).mean(axis=0) )
              #monitor['train'].append(  np.array([ train_sync(i, e) for i in range(n_batches) ]).mean(axis=0)  )
              print "e=",e," train:",monitor['train'][-1], "determinant=",det," precision = ",np.array([[w_x_[0], 0, 0.5*w_[0]],[0, w_x_[1], 0.5*w_[1]],[0.5*w_[0], 0.5*w_[1], w_h_]])*2
              #print np.linalg.inv(np.array([monitor['monitor_params'][-1][1], monitor['monitor_params'][-1][0], monitor['monitor_params'][-1][0], monitor['monitor_params'][-1][2]]).reshape(2,2)*2)
              precision.set_value(np.array([[w_x_[0], 0, 0.5*w_[0]],[0, w_x_[1], 0.5*w_[1]],[0.5*w_[0], 0.5*w_[1], w_h_]])*2)
              # print "precision=",precision.get_value()
              cov.set_value(np.linalg.inv(np.array(precision.get_value())))
              marginal_precision=np.linalg.inv(cov.get_value()[0:2,0:2])
              print "marginal NLL=",marginal_nll(train_x.get_value(),marginal_precision)[0],"C="
              print cov.get_value()
              #print np.linalg.det(np.linalg.inv(np.array([[w_x_[0], 0, 0.5*w_[0]],[0, w_x_[1], 0.5*w_[1]],[0.5*w_[0], 0.5*w_[1], w_h_]])*2)[:0:2][0:2]) #(w_x_[0][1]+w_x_[1][0])/2

if __name__ == "__main__":
    exp(100000, 256)

