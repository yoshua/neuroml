
from collections import OrderedDict
import theano
import pdb
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

np.random.seed(54321)

def sharedX(x) : return theano.shared( theano._asarray(x, dtype=theano.config.floatX) )

def rho(a):
    return (T.switch(a + 0.5 > 0, a+0.5, 0) -
            T.switch(a - 0.5 > 0, a-0.5, 0))

def rho_prime(a):
    return (T.switch(a + 0.5 > 0, 1, 0) -
            T.switch(a - 0.5 > 0, 1, 0))

class LatentPredictor1:
    def __init__(self, n1=1, n2=1, nh=1, n_inference_steps=10, epsilon=0.05, lrate=0.01, batchsize=1, debug_level=1):
        self.epsilon=epsilon
        self.lrate=lrate
        self.n1 = n1
        self.n2 = n2
        self.nh = nh
        self.batchsize = batchsize
        self.debug_level = debug_level
        self.x1 = sharedX(np.zeros((batchsize,n1)))
        self.x2 = sharedX(np.zeros((batchsize,n2)))
        self.h = sharedX(np.zeros((batchsize,nh)))        
        self.prev_h = sharedX(np.zeros((batchsize,nh)))        
        self.n_inference_steps=n_inference_steps
        self.bx1 = sharedX(np.zeros((n1))-.2)
        self.bx2 = sharedX(np.zeros((n2))-.2)
        self.bh = sharedX(np.zeros((nh))-.2)
        self.W1 = sharedX(np.random.uniform(-epsilon,epsilon,((nh,n1))))
        self.W2 = sharedX(np.random.uniform(-epsilon,epsilon,((nh,n2))))
                          

        self.new_h = self.h *(1-self.epsilon) + self.epsilon*rho_prime(self.h)* \
          (self.bh + T.dot(rho(self.x1),self.W1.T) + T.dot(rho(self.x1),self.W2.T))
        self.new_x1 = self.x1*(1-self.epsilon) + self.epsilon*rho_prime(self.x1)* \
          (self.bx1 + T.dot(rho(self.h),self.W1))
        self.new_x2 = self.x2*(1-self.epsilon) + self.epsilon*rho_prime(self.x2)* \
          (self.bx2 + T.dot(rho(self.h),self.W2))
        self.new_bx1 = self.bx1 - self.lrate*(T.sum((self.new_x1-self.x1)*rho_prime(self.x1),axis=0))
        self.new_bx2 = self.bx2 - self.lrate*(T.sum((self.new_x2-self.x2)*rho_prime(self.x2),axis=0))
        self.new_bh = self.bh - self.lrate*(T.sum((self.h-self.new_h)*rho_prime(self.prev_h),axis=0))
        self.new_W1 = self.W1 - self.lrate*(T.dot(((self.h-self.new_h)*rho_prime(self.prev_h)).T,rho(self.x1)) \
                                            +T.dot(rho(self.h).T,(self.new_x1-self.x1)*rho_prime(self.x1)))/self.batchsize
        self.new_W2 = self.W2 - self.lrate*(T.dot(((self.h-self.new_h)*rho_prime(self.prev_h)).T,rho(self.x2)) \
                                            +T.dot(rho(self.h).T,(self.new_x2-self.x2)*rho_prime(self.x2)))/self.batchsize

        self.new_x_f = theano.function([],[self.new_x1,self.new_x2])
        self.inter_f = theano.function([],[-T.dot(((self.h-self.new_h)*rho_prime(self.prev_h)).T,rho(self.x1))*self.lrate/self.batchsize,
                                           -T.dot(rho(self.h).T,(self.new_x1-self.x1)*rho_prime(self.x1))*self.lrate/self.batchsize,
                                           -T.dot(((self.h-self.new_h)*rho_prime(self.prev_h)).T,rho(self.x2))*self.lrate/self.batchsize,
                                           -T.dot(rho(self.h).T,(self.new_x2-self.x2)*rho_prime(self.x2))*self.lrate/self.batchsize,
                                           self.h-self.new_h,
                                           self.new_x1-self.x1,
                                           self.new_x2-self.x2,
                                           rho_prime(self.prev_h)
                                           ])
        self.update_h_params_f = theano.function([],[],
                                            updates = OrderedDict([(self.W1,self.new_W1),
                                                                   (self.W2,self.new_W2),
                                                                   (self.bx1,self.new_bx1),
                                                                   (self.bx2,self.new_bx2),
                                                                   (self.bh,self.new_bh),
                                                                   (self.prev_h,self.h),
                                                                   (self.h,self.new_h)
                                                                   ]))

        self.update_h_f = theano.function([],[],
                                            updates = OrderedDict([
                                                                   (self.prev_h,self.h),
                                                                   (self.h,self.new_h)
                                                                   ]))
        self.update_state_f = theano.function([],[],
                                            updates = OrderedDict([(self.x1,self.new_x1),
                                                                   (self.x2,self.new_x2),
                                                                   (self.prev_h,self.h),
                                                                   (self.h,self.new_h)
                                                                   ]))
    def update_h_params(self,x1,x2):
        self.x1.set_value(x1)
        self.x2.set_value(x2)
        self.prev_h.set_value(self.h.get_value()*0.)
        self.h.set_value(self.h.get_value()*0.)
        if self.debug_level>=3:
            print "x1=",self.x1.get_value()[0:2,:]
            print "x2=",self.x2.get_value()[0:2,:]
        if self.debug_level>=1:
            print "W1=",self.W1.get_value()
            print "W2=",self.W2.get_value()
        for it in range(self.n_inference_steps):
            dw1h,dw1x,dw2h,dw2x,dh,dx1,dx2,rph = self.inter_f()
            if self.debug_level>=2:
                print "it = ",it,
            if self.debug_level>=3:
                print " prev_h=",self.prev_h.get_value()[0:2,:]
                print " h=",self.h.get_value()[0:2,:]
                print " dh=",dh[0:2,:]
                print " drph=",rph[0:2,:]
                print " dx1=",dx1[0:2,:]
                print " dx2=",dx2[0:2,:]
                print " dw1h=",dw1h[0:2,:]
                print " dw1x=",dw1x[0:2,:]
                print " dw2h=",dw2h[0:2,:]
                print " dw2x=",dw2x[0:2,:]
            if it<1:
                self.update_h_f()
            else:
                self.update_h_params_f()
            new_x1, new_x2 = self.new_x_f()
            if self.debug_level>=3:
                print "new_h=",self.h.get_value()[0:2,:]
                print "bx1=",self.bx1.get_value()
                print "bx2=",self.bx2.get_value()
                print "bh=",self.bh.get_value()
                print "new_x1=",new_x1[0:2,:]
                print "new_x2=",new_x2[0:2,:]
            if self.debug_level>=2:
                print "W1=",self.W1.get_value()
                print "W2=",self.W2.get_value()

    def update_state(self):
        for it in range(self.n_inference_steps):
            self.update_state_f()

def data1():
#    data = np.array([[.3,.3],[-.3,-.3],[.2,.2],[-.4,-.4],[-.1,-.1],[.25,.25],[-.2,-.2],[.1,.1]])
    data = np.array([[.3,.3],[-.3,-.3]])
#    data = np.array([[-.3,-.3]])
    return data

def data2():
    x1 = np.random.uniform(-.5,.5,((100,1)))
    x2 = 0.8*x1 + np.random.uniform(-.1,.1,((100,1)))
    data = np.hstack((x1,x2))
    return data

def exp():
    debug_level=3
    nh = 1
    n_inference_steps = 10
    epsilon = 0.1
    lrate = 0.1
    n_epochs = 100
    data = data2()    
    n_examples = data.shape[0]
    nx = data.shape[1]
    nx1 = nx/2
    nx2 = nx-nx1
    x1 = data[:,0:nx1]
    x2 = data[:,nx1:nx]
    n_batches = 1

    batchsize = n_examples/n_batches

    model = LatentPredictor1(n1=nx1,n2=nx2,nh=nh,n_inference_steps=n_inference_steps,
                             batchsize=batchsize, epsilon=epsilon, lrate=lrate, debug_level=debug_level)

    for epoch in range(n_epochs):
        for batch in range(n_batches):
            model.update_h_params(x1[batch*batchsize:(batch+1)*batchsize],x2[batch*batchsize:(batch+1)*batchsize])


if __name__ == "__main__":
    exp()
    
