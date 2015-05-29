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


def exp_old(max_ep = 200, batsize = 10, opt=adam()):
    batch_size = batsize 
    n_batches = toy_num / batch_size 
    max_epochs = max_ep
    H = sharedX(np.zeros((batch_size,1)))
    X = sharedX(np.zeros((batch_size,2)))

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
    
    #H = T.matrix() 
    obsX = T.matrix() # the observed data
    #X = T.matrix() # the partially free X
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

    # the update of X with obsX not given, i.e., X is totally free
    freeX_new = X - eps * (2*w_x*X + T.dot(H, w.T) ) + gaussian(0.*X_0, 1)*sigma
    # the update of X with obsX given
    clampedX_new = X - eps * (2*w_x*X - obsX) + gaussian(0.*X_0, 1)*sigma

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
    sleep_updates = updates + [(H,H_new),(X,freeX_new)]
    wake_updates = updates + [(H,H_new),(X,clampedX_new)]
    notrain_sleep_updates = [(H,H_new),(X,freeX_new)]
    notrain_wake_updates = [(H,H_new),(X,clampedX_new)]

    # training givens
    givens_wake = lambda i : { obsX : train_x[ i*batch_size : (i+1)*batch_size ]}

    # training and testing function
    
    notrain_sleep = theano.function([i, e], [cost, H_direct, determinant], 
                                  on_unused_input='ignore', updates = notrain_sleep_updates)


    notrain_wake = theano.function([i, e], [cost, H_direct, determinant], 
                                 givens = givens_wake(i), on_unused_input='ignore',
                                 updates = notrain_wake_updates)

    train_sleep = theano.function([i, e], [cost, H_direct, determinant], 
                                  on_unused_input='ignore', updates = sleep_updates)


    train_wake = theano.function([i, e], [cost, H_direct, determinant], 
                                 givens = givens_wake(i), on_unused_input='ignore',
                                 updates = wake_updates)

    train_sync_notup = theano.function([i, e], [cost, H_new, H_direct, E,determinant], 
                                       givens = givens_wake(i), on_unused_input='ignore')

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
                H.set_value(np.zeros((batch_size,1)))
                X.set_value(train_x.get_value()[i*batch_size:(i+1)*batch_size]+np.random.normal(0,scale=0.1,size=X.get_value().shape)) # initialize sleep phase from true data
                #if sign_error>0.8: print "sign error = ",sign_error," # H0 =",h[0,0]
                for t in range(5):
                    [cost, hstar, det] = notrain_sleep(i, e)
                for t in range(5):
                    [cost, hstar, det] = notrain_wake(i, e)
                [cost, hstar, det] = train_wake(i, e)
                result = result + [cost]
                h=H.get_value()
                #sign_error = 0.5*np.mean(np.abs(np.sign(h)-np.sign(hstar)))
                #if sign_error>0.8 or det<=0:
                #   # print "SIGN=",np.sign(h),np.sign(hstar),0.5*np.abs(np.sign(h)-np.sign(hstar)),sign_error
                #   print "H=",hstar[0,0],h[0,0]," %err = ",sign_error, "DET=",det
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
    exp_old(100000, 256)

