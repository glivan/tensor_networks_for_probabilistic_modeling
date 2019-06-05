# -*- coding: utf-8 -*-

from .MPSClass import TN
import numpy as np
from sklearn.externals.six.moves import xrange

class RealBorn(TN):
    """Born machine with real parameters
    Probability is the square of the MPS
    Parameters
    ----------
    D : int, optional
        Rank/Bond dimension of the MPS
    learning_rate : float, optional
        Learning rate of the gradient descent algorithm
    batch_size : int, optional
        Number of examples per minibatch.
    n_iter : int, optional
        Number of iterations (epochs) over the training dataset to perform
        during training.
    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.
    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.
    ----------
    Attributes
    ----------
    w : numpy array, shape (m_parameters)
        Parameters of the tensor network
    norm : float
        normalization constant for the probability distribution
    n_samples : int
        number of training samples
    n_features : int
        number of features in the dataset
    d : int
        physical dimension (dimension of the features)
    m_parameters : int
        number of parameters in the network
    history : list
        saves the training accuracies during training
    """
    def __init__(self, D=4, learning_rate=0.1, batch_size=10,
                 n_iter=100, random_state=None, verbose=False):
        self.D = D
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        
    def _probability(self, x):
        """Unnormalized probability of one configuration P(x)
        Parameters
        ----------
        x : numpy array, shape (n_features,)
            One configuration
        Returns
        -------
        probability : float
        """
        w2 = np.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        
        tmp = w2[0,x[0],0,:] #First tensor
        for i in xrange(1,self.n_features-1):
            tmp = np.dot(tmp,w2[i,x[i],:,:]) #MPS contraction  
        probability = np.inner(tmp,
                        w2[self.n_features-1,x[self.n_features-1],:,0])**2
        return probability      

    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """
        w2 = np.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        tmp = np.tensordot(w2[0,:,0,:],w2[0,:,0,:],axes=([0],[0])).reshape(self.D*self.D) #First tensor
        for i in xrange(1,self.n_features-1):
            tmp = np.dot(tmp,np.tensordot(w2[i,:,:,:],w2[i,:,:,:],
                    axes=([0],[0])).transpose((0,2,1,3)).reshape(self.D*self.D,self.D*self.D)) #MPS contraction  
        norm = np.inner(tmp,np.tensordot(w2[self.n_features-1,:,:,0],
                        w2[self.n_features-1,:,:,0],axes=([0],[0])).reshape(self.D*self.D))
        return norm

    def _derivative(self, x):
        """Compute the derivative of P(x)
        Parameters
        ----------
        x : numpy array, shape (n_features,)
            One configuration
        Returns
        -------
        derivative : numpy array, shape (m_parameters,)
        """
        w2 = np.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        derivative = np.zeros((self.n_features,self.d,self.D,self.D))
        
        #Store intermediate tensor contractions for the derivatives: 
        #left to right and right to left
        #tmp stores the contraction of the first i+1 tensors from the left 
        #in tmp[i,:,:], tmp2 the remaining tensors on the right
        #the mps contracted is the remaining contraction tmp[i-1]w[i]tmp2[i+1]
        tmp=np.zeros((self.n_features,self.D))
        tmp2=np.zeros((self.n_features,self.D))
        tmp[0,:]=w2[0,x[0],0,:]
        for i in xrange(1,self.n_features-1):
            tmp[i,:]=np.dot(tmp[i-1,:],w2[i,x[i],:,:])  
        mpscontracted=np.inner(tmp[self.n_features-2,:],
                               w2[self.n_features-1,x[self.n_features-1],:,0])
        
        tmp[self.n_features-1,:]=np.inner(tmp[self.n_features-2,:],
                                w2[self.n_features-1,x[self.n_features-1],:,0])
        tmp2[self.n_features-1,:]=w2[self.n_features-1,x[self.n_features-1],:,0]
        for i in xrange(self.n_features-2,-1,-1):
            tmp2[i,:]=np.dot(w2[i,x[i],:,:],tmp2[i+1,:])
        tmp2[0,:]=np.inner(w2[0,x[0],0,:],tmp2[1,:])
    
        #The derivative of each tensor is the contraction of the other tensors
        derivative[0,x[0],0,:]=2*tmp2[1,:]*mpscontracted
        derivative[self.n_features-1,x[self.n_features-1],:,0]=\
                                2*tmp[self.n_features-2,:]*mpscontracted
        for i in xrange(1,self.n_features-1):
                derivative[i,x[i],:,:]=2*np.outer(tmp[i-1,:],tmp2[i+1,:])*mpscontracted

        return derivative.reshape(self.m_parameters)

    def _derivativenorm(self):
        """Compute the derivative of the norm
        Returns
        -------
        derivative : numpy array, shape (m_parameters,)
        """        
        w2=np.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        derivative=np.zeros((self.n_features,self.d,self.D,self.D)) 
        
        tmp=np.zeros((self.n_features,self.D*self.D))
        tmp2=np.zeros((self.n_features,self.D*self.D))
        tmp[0,:]=np.tensordot(w2[0,:,0,:],w2[0,:,0,:],axes=([0],[0])).reshape(self.D*self.D)
        for i in xrange(1,self.n_features-1):
            tmp[i,:]=np.dot(tmp[i-1,:],np.tensordot(w2[i,:,:,:],w2[i,:,:,:],
                axes=([0],[0])).transpose((0,2,1,3)).reshape(self.D*self.D,self.D*self.D))
        tmp[self.n_features-1,:]=np.inner(tmp[self.n_features-2,:],
            np.tensordot(w2[self.n_features-1,:,:,0],w2[self.n_features-1,:,:,0],axes=([0],[0])).reshape(self.D*self.D))
        
        tmp2[self.n_features-1,:]=np.tensordot(w2[self.n_features-1,:,:,0],
            w2[self.n_features-1,:,:,0],axes=([0],[0])).reshape(self.D*self.D)
        for i in xrange(self.n_features-2,-1,-1):
            tmp2[i,:]=np.dot(np.tensordot(w2[i,:,:,:],
                w2[i,:,:,:],axes=([0],[0])).transpose((0,2,1,3)).reshape(self.D*self.D,self.D*self.D),tmp2[i+1,:])
        tmp2[0,:]=np.inner(np.tensordot(w2[0,:,0,:],w2[0,:,0,:],
                axes=([0],[0])).reshape(self.D*self.D),tmp2[1,:])
        

        for j in xrange(self.d):
            derivative[0,j,0,:]=2*np.dot(tmp2[1,:].reshape(self.D,self.D),
                                                        w2[0,j,0,:])
            derivative[self.n_features-1,j,:,0]=2*np.dot(tmp[self.n_features-2,:].reshape(self.D,self.D),
                                    w2[self.n_features-1,j,:,0])
        for i in xrange(1,self.n_features-1):
            temp1=tmp[i-1,:].reshape(self.D,self.D)
            temp2=tmp2[i+1,:].reshape(self.D,self.D)
            
            for j in xrange(self.d):
                temp3=np.dot(np.dot(temp1,w2[i,j,:,:]),temp2.transpose())
                derivative[i,j,:,:]=2*np.copy(temp3)
                
        return derivative.reshape(self.m_parameters)