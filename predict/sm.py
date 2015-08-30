
from collections import OrderedDict
import theano
import pdb
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T
from theano.gradient import Rop
from theano import printing

np.random.seed(54321)

debug_printing = False

def pp(s,x) : 
    if debug_printing:
       return printing.Print(s)(x)
    return x

def sharedX(x) : return theano.shared( theano._asarray(x, dtype=theano.config.floatX) )

def rho(a):
#    return (T.switch(a + 0.5 > 0, a+0.5, 0) -
#            T.switch(a - 0.5 > 0, a-0.5, 0))
#    return (T.switch(a > -1, a, -1) -
#            T.switch(a > 1, a-1, 0))
    return T.tanh(a)

def rho_prime(a):
#    return (T.switch(a + 0.5 > 0, 1, 0) -
#            T.switch(a - 0.5 > 0, 1, 0))
#    return (T.switch(a > -1, 1, 0) -
#            T.switch(a > 1, 1, 0))
     t=T.tanh(a)
     return 1 - t*t

def rho_second(a):
     t=T.tanh(a)
     return -2*t*(1-t*t)

def squared_norm(x): return T.sum(x*x)
    
class LatentPredictor1:
    def __init__(self, n1=1, n2=1, nh=1, n_inference_steps=10, epsilon=0.05, lrate=0.01, batchsize=1, debug_level=1, wd=.1e-3):
        self.epsilon=epsilon
        self.lrate=lrate
        self.n1 = n1
        self.n2 = n2
        self.nh = nh
        self.wd = wd
        self.batchsize = batchsize
        self.debug_level = debug_level
        self.x1 = sharedX(np.zeros((batchsize,n1)))
        self.x2 = sharedX(np.zeros((batchsize,n2)))
        self.h = sharedX(np.zeros((batchsize,nh)))        
        self.prev_h = sharedX(np.zeros((batchsize,nh)))        
        self.n_inference_steps=n_inference_steps
        self.bx1 = sharedX(np.zeros((n1)))
        self.bx2 = sharedX(np.zeros((n2)))
        self.bh = sharedX(np.zeros((nh)))
        self.W1 = sharedX(np.random.uniform(-epsilon,epsilon,((nh,n1))))
        self.W2 = sharedX(np.random.uniform(-epsilon,epsilon,((nh,n2))))

        h_prediction = self.bh + T.dot(rho(self.x1),self.W1.T) + T.dot(rho(self.x2),self.W2.T)
        x1_prediction = self.bx1+T.dot(rho(self.h),self.W1)
        x2_prediction = self.bx2+T.dot(rho(self.h),self.W2)

        # energy of state s=(h,x1,x2)
        # E = .5 ||s||^2 - .5 sum_{i,j} W_ij rho(s_i) rho(s_j) - sum_i b_i rho(s_i)
        # with W_ij = W_ji and non-zero W only for interaction between h and x1 (W1) and between h and x2 (W2), and W_ii=0
        E = 0.5*(squared_norm(self.h)+squared_norm(self.x1)+squared_norm(self.x2)) \
          - T.sum(rho(self.h)*h_prediction) - T.sum(T.dot(rho(self.x1),self.bx1) \
                                                   + T.dot(rho(self.x2),self.bx2))
                                                   
        mean_E = E / self.batchsize
        
        dEdh = T.grad(E,self.h)
        dEdx1 = T.grad(E,self.x1)
        dEdx2 = T.grad(E,self.x2)

        # manually, this gives
        # dE/ds_i = s_i - rho'(s_i) (b_i + sum_{i,j} W_ij rho(s_j))
        # dEdh_ = self.h - rho_prime(self.h)*h_prediction
        # dEdx1_ = self.x1 - rho_prime(self.x1)*x1_prediction
        # dEdx2_ = self.x2 - rho_prime(self.x2)*x2_prediction

        # sum_i d2E/ds_i^2 = sum_i ( 1 - rho''(s_i) (b_i + sum_ij W_ij rho(s_j)) )
        #                  = sum_{t,i} d2E / dh_{t,i}^2 + sum_{t,i} d2E / dx1_{t,i}^2 + sum_{t,i} d2E / dx2_{t,i}^2
        # but ignore the 1, whose derivative wrt parameters is 0
        d2Eds2 = - T.sum(rho_second(self.h)*h_prediction) - T.sum(rho_second(self.x1)*x1_prediction) - T.sum(rho_second(self.x2)*x2_prediction)

        # score matching objective = .5 || dE/ds ||^2 - sum_i d2E/ds_i^2
        J = 0.5*(squared_norm(dEdh)+squared_norm(dEdx1)+squared_norm(dEdx2))/self.batchsize \
            - d2Eds2/self.batchsize

        # check consistency of manually computed derivatives with grad's
        #err1h = squared_norm(dEdh_-dEdh)
        #err1x1 = squared_norm(dEdx1_-dEdx1)
        #err1x2 = squared_norm(dEdx2_-dEdx2)

        # state update
        self.new_h = self.h - self.epsilon*dEdh
        self.new_x1 = self.x1 - self.epsilon*dEdx1
        self.new_x2 = self.x2 - self.epsilon*dEdx2

        # biases update
        dbx1 = -T.grad(J,self.bx1)*self.lrate
        dbx2 = -T.grad(J,self.bx2)*self.lrate
        dbh = -T.grad(J,self.bh)*self.lrate

        # dbx1_ = -self.lrate/self.batchsize*T.sum(-dEdx1*rho_prime(self.x1)+rho_second(self.x1),axis=0)
        # dbx2_ = -self.lrate/self.batchsize*T.sum(-dEdx2*rho_prime(self.x2)+rho_second(self.x2),axis=0)
        # dbh_ = -self.lrate/self.batchsize*T.sum(-dEdh*rho_prime(self.h)+rho_second(self.h),axis=0)

        # check consistency of manually computed derivatives with grad's
        # err2h = squared_norm(dbh_-dbh)
        # err2x1 = squared_norm(dbx1_-dbx1)
        # err2x2 = squared_norm(dbx2_-dbx2)
        
        self.new_bx1 = self.bx1 + dbx1
        self.new_bx2 = self.bx2 + dbx2
        self.new_bh = self.bh + dbh

        h_factor = -dEdh*rho_prime(self.h)+rho_second(self.h)
        dw1 = - self.lrate/self.batchsize*(T.dot(h_factor.T,rho(self.x1)) + \
                                           T.dot(rho(self.h).T,-dEdx1*rho_prime(self.x1)+rho_second(self.x1)))
        self.new_W1 = (1.-self.wd)*self.W1 + dw1
        dw2 = - self.lrate/self.batchsize*(T.dot(h_factor.T,rho(self.x2)) + \
                                           T.dot(rho(self.h).T,-dEdx2*rho_prime(self.x2)+rho_second(self.x2)))
        self.new_W2 = (1.-self.wd)*self.W2 + dw2

        self.new_x_f = theano.function([],[self.new_x1,self.new_x2])
        dh = self.h-self.new_h
        dx1 = self.new_x1-self.x1
        dx2 = self.new_x2-self.x2
        rph = rho_prime(self.h)
        r2h = rho_second(self.h)
        self.inter_f = theano.function([],[
                                           J,
                                           mean_E,
                                           dh,
                                           dx1, 
                                           dx2, 
                                           rph,
                                           r2h,
                                           dbx1,
                                           dbx2,
                                           dw1,
                                           dw2
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

        self.energy_f = theano.function([],[E])
        
    def update_h_params(self,x1,x2):
        self.x1.set_value(x1)
        self.x2.set_value(x2)
        self.prev_h.set_value(self.h.get_value()*0.)
#        self.h.set_value(self.h.get_value()*0.)
#        self.prev_h.set_value(np.random.uniform(-1.,1.,((self.batchsize,self.nh))))
        self.h.set_value(self.prev_h.get_value())
        if self.debug_level>=3:
            print "x1=",self.x1.get_value()[0:2,:]
            print "x2=",self.x2.get_value()[0:2,:]
        if self.debug_level>=1:
            print "W1=",self.W1.get_value()
            print "W2=",self.W2.get_value()
        for it in range(self.n_inference_steps):
#            err1h,err1x1,err1x2,err2h,err2x1,err2x2,J,E,dh,dx1,dx2,rph,r2h,dbx1,dbx2,dw1,dw2 = self.inter_f()
            J,E,dh,dx1,dx2,rph,r2h,dbx1,dbx2,dw1,dw2 = self.inter_f()
            if self.debug_level>=2:
                print "it = ",it,
            if self.debug_level>=3:
                print " prev_h=",self.prev_h.get_value()[0:2,:]
                print " h=",self.h.get_value()[0:2,:]
                print " dh=",dh[0:2,:]
                print " dx1=",dx1[0:2,:]
                print " dx2=",dx2[0:2,:]
#                print " err1=",err1h,err1x1,err1x2
#                print " dEdh=",dEdh[0:2,:]
#                print " dEdh_=",dEdh_[0:2,:]
#                print " err2=",err2h,err2x1,err2x2
                print " J=",J
                print " E=",E
                print " dph=",rph[0:2,:]
                print " r2h=",r2h[0:2,:]
                print " dbx1=",dbx1
                print " dbx2=",dbx2
                print " dw1=",dw1
                print " dw2=",dw2
            if it<self.n_inference_steps-1:
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

        print "final (E,J) before params update = ",E,J
        J,E,dh,dx1,dx2,rph,r2h,dbx1,dbx2,dw1,dw2 = self.inter_f()
        print "final (E,J) after params update = ",E,J
        return E,J

    def update_state(self):
        for it in range(self.n_inference_steps):
            self.update_state_f()
            
    def energy(self,x1,x2,h=NULL):
        self.x1.set_value(x1)
        self.x2.set_value(x2)
        if h=NULL:
            self.prev_h.set_value(self.h.get_value()*0.)
            self.h.set_value(self.prev_h.get_value())
            for it in range(self.n_inference_steps):
                self.update_state_f()
        else:
            self.h.set_value(h)
        return self.energy_f()

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

def plot_energy_surface(model):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt


    (x1,x2) = numpy.meshgrid(numpy.arange(-1.25,1.25,.05),numpy.arange(-1.25,1.25,.05))
    E = model.energy(x1,x2)
    

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

def exp():
    debug_level=3
    nh = 1
    n_inference_steps = 8
    epsilon = 0.1
    lrate = 0.1
    n_epochs = 100
    weight_decay = 1e-3
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
                             batchsize=batchsize, epsilon=epsilon, lrate=lrate, debug_level=debug_level, wd=weight_decay)

    for epoch in range(n_epochs):
        epoch_E=epoch_J=0
        for batch in range(n_batches):
            batch_E,batch_J=model.update_h_params(x1[batch*batchsize:(batch+1)*batchsize],x2[batch*batchsize:(batch+1)*batchsize])
            epoch_E+=batch_E
            epoch_J+=batch_J
        print "EPOCH E,J = ",epoch_E/n_batches,epoch_J/n_batches


    plot_energy_surface(model)
    
if __name__ == "__main__":
    debug_printing = True
    exp()
    
