# -*- coding: utf-8 -*-

from .MPSClass import MPS
import numpy as np
from sklearn.externals.six.moves import xrange


class ComplexLPS(MPS):
    """Locally purified states (LPS) with complex elements
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
        random permutations generator and of the initial parameters.
        If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.
    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.
    mu : int, optional
        Dimension of the purification link
    ----------
    Attributes
    ----------
    w : numpy array, shape (n_parameters)
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
                 n_iter=100, random_state=None, verbose=False, mu=2):
        self.D = D
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.mu = mu
        
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
        w2 = np.reshape(self.w,(self.n_features,self.d,self.D,self.D,self.mu))
      
        tmp = w2[0,x[0],0,:,:]
        tmp2 = np.einsum('ij,kj->ik',tmp,np.conjugate(tmp)).reshape(self.D*self.D)
        for i in xrange(1,self.n_features-1):
            tmp = np.einsum('imj,klj->ikml',w2[i,x[i],:,:,:],
                            np.conjugate(w2[i,x[i],:,:,:])).reshape((self.D*self.D,self.D*self.D))
            tmp2 = np.dot(tmp2,tmp)

        tmp = np.einsum('ij,kj->ik',w2[self.n_features-1,x[self.n_features-1],:,0,:],
                        np.conjugate(w2[self.n_features-1,
                        x[self.n_features-1],:,0,:])).reshape(self.D*self.D)
        probability = np.abs(np.inner(tmp2,tmp))
        return probability
       
    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """
        w2 = np.reshape(self.w,(self.n_features,self.d,self.D,self.D,self.mu))

        tmp2 = np.einsum('ijk,ilk->jl',w2[0,:,0,:,:],
                         np.conj(w2[0,:,0,:,:])).reshape(self.D*self.D)
        for i in xrange(1,self.n_features-1):
            tmp = np.einsum('pimj,pklj->ikml',w2[i,:,:,:,:],
                            np.conjugate(w2[i,:,:,:,:])).reshape((self.D*self.D,self.D*self.D))
            tmp2 = np.dot(tmp2,tmp)
        tmp = np.einsum('ijk,ilk->jl',w2[self.n_features-1,:,:,0,:],
                        np.conjugate(w2[self.n_features-1,:,:,0,:])).reshape(self.D*self.D)
        norm = np.abs(np.inner(tmp2,tmp))
        return norm

    def _derivative(self, x):
        """Compute the derivative of P(x)
        Parameters
        ----------
        x : numpy array, shape (n_features,)
            One configuration
        Returns
        -------
        derivative : numpy array, shape (n_parameters,)
        """
        w2=np.reshape(self.w,(self.n_features,self.d,self.D,self.D,self.mu))
        derivative=np.zeros((self.n_features,self.d,self.D,self.D,self.mu),
                            dtype=np.complex128)
        
        #Store intermediate tensor contractions for the derivatives: 
        #left to right and right to left
        #tmp stores the contraction of the first i+1 tensors from the left 
        #in tmp[i,:,:], tmp2 the remaining tensors on the right
        #the mps contracted is the remaining contraction tmp[i-1]w[i]tmp2[i+1]
        tmp=np.zeros((self.n_features,self.D*self.D),dtype=np.complex128)
        tmp2=np.zeros((self.n_features,self.D*self.D),dtype=np.complex128)
        tmp[0,:] = np.einsum('ij,kj->ik',w2[0,x[0],0,:,:],
                    np.conjugate(w2[0,x[0],0,:,:])).reshape(self.D*self.D)
        for i in xrange(1,self.n_features-1):
            newtmp = np.einsum('imj,klj->ikml',w2[i,x[i],:,:,:],
                        np.conjugate(w2[i,x[i],:,:,:])).reshape((self.D*self.D,self.D*self.D))
            tmp[i,:]=np.dot(tmp[i-1,:],newtmp)  
        newtmp = np.einsum('ij,kj->ik',w2[self.n_features-1,x[self.n_features-1],:,0,:],
                    np.conjugate(w2[self.n_features-1,x[self.n_features-1],:,0,:])).reshape(self.D*self.D)
        mpscontracted=np.inner(tmp[self.n_features-2,:],newtmp)
        tmp[self.n_features-1,:]=mpscontracted
        
        
        tmp2[self.n_features-1,:]=newtmp
        for i in xrange(self.n_features-2,-1,-1):
            newtmp = np.einsum('imj,klj->ikml',w2[i,x[i],:,:,:],
                      np.conjugate(w2[i,x[i],:,:,:])).reshape((self.D*self.D,self.D*self.D))
            tmp2[i,:]=np.dot(newtmp,tmp2[i+1,:])
        newtmp=np.einsum('ij,kj->ik',w2[0,x[0],0,:,:],
                         np.conjugate(w2[0,x[0],0,:,:])).reshape(self.D*self.D)
        tmp2[0,:]=np.inner(newtmp,tmp2[1,:])
    
        #Now for each tensor, the derivative is the contraction of the rest of the tensors
        
        derivative[0,x[0],0,:,:]=2*np.einsum('ij,il->lj',
                        w2[0,x[0],0,:,:],tmp2[1,:].reshape(self.D,self.D))
        derivative[self.n_features-1,x[self.n_features-1],:,0,:]=\
            2*np.einsum('ij,il->lj',w2[self.n_features-1,x[self.n_features-1],:,0,:],
                        tmp[self.n_features-2,:].reshape(self.D,self.D))
        for i in xrange(1,self.n_features-1):
            temp1=tmp[i-1,:].reshape(self.D,self.D)
            temp2=tmp2[i+1,:].reshape(self.D,self.D)
            derivative[i,x[i],:,:,:]=2*np.einsum('ikm,ij,kl->jlm',w2[i,x[i],:,:,:],temp1,temp2)

        return derivative.reshape(self.m_parameters)

    def _derivativenorm(self):
        """Compute the derivative of the norm
        Returns
        -------
        derivative : numpy array, shape (n_parameters,)
        """   
        
        w2=np.reshape(self.w,(self.n_features,self.d,self.D,self.D,self.mu))
        derivative=np.zeros((self.n_features,self.d,self.D,self.D,self.mu),dtype=np.complex128)
        
        tmp=np.zeros((self.n_features,self.D*self.D),dtype=np.complex128)
        tmp2=np.zeros((self.n_features,self.D*self.D),dtype=np.complex128)
        
        tmp[0,:] = np.einsum('ijk,ilk->jl',w2[0,:,0,:,:],
                        np.conj(w2[0,:,0,:,:])).reshape(self.D*self.D)
        for i in xrange(1,self.n_features-1):
            newtmp = np.einsum('pimj,pklj->ikml',w2[i,:,:,:,:],
                               np.conjugate(w2[i,:,:,:,:])).reshape((self.D*self.D,self.D*self.D))
            tmp[i,:] = np.dot(tmp[i-1,:],newtmp)  
        newtmp = np.einsum('ijk,ilk->jl',w2[self.n_features-1,:,:,0,:],
                           np.conjugate(w2[self.n_features-1,:,:,0,:])).reshape(self.D*self.D)
        mpscontracted=np.inner(tmp[self.n_features-2,:],newtmp)
        tmp[self.n_features-1,:]=mpscontracted

        tmp2[self.n_features-1,:]=newtmp
        for i in xrange(self.n_features-2,-1,-1):
            newtmp = np.einsum('pimj,pklj->ikml',w2[i,:,:,:,:],
                        np.conjugate(w2[i,:,:,:,:])).reshape((self.D*self.D,self.D*self.D))
            tmp2[i,:] = np.dot(newtmp,tmp2[i+1,:])
        newtmp=np.einsum('ijk,ilk->jl',w2[0,:,0,:,:],np.conj(w2[0,:,0,:,:])).reshape(self.D*self.D)
        tmp2[0,:]=np.inner(newtmp,tmp2[1,:])
        
        for j in xrange(self.d):
            derivative[0,j,0,:,:]=2*np.einsum('ij,il->lj',w2[0,j,0,:,:],
                                            tmp2[1,:].reshape(self.D,self.D))
            derivative[self.n_features-1,j,:,0,:]=\
            2*np.einsum('ij,il->lj',w2[self.n_features-1,j,:,0,:],
                            tmp[self.n_features-2,:].reshape(self.D,self.D))
        for i in xrange(1,self.n_features-1):
            temp1=tmp[i-1,:].reshape(self.D,self.D)
            temp2=tmp2[i+1,:].reshape(self.D,self.D)
            for j in xrange(self.d):
                derivative[i,j,:,:,:]=2*np.einsum('ikm,ij,kl->jlm',w2[i,j,:,:,:],temp1,temp2)
        
        return derivative.reshape(self.m_parameters)


    def _weightinitialization(self,rng):
        """Initialize weights w randomly
        Parameters
        ----------
        rng : random number generation
        """
        self.m_parameters = self.n_features*self.d*self.D*self.D*self.mu
        self.w=np.asarray(rng.normal(0, 1, self.m_parameters))\
                +1j*np.asarray(rng.normal(0, 1, self.m_parameters))
        
    def _likelihood_derivative(self, v):
        """Compute derivative of log-likelihood of configurations in v
        Parameters
        ----------
        v : numpy array, shape (n_samples,n_features)
            Configurations
        Returns
        -------
        update_w : numpy array, shape (n_parameters,)
            array of derivatives of the log-likelihood
        """
        update_w=np.zeros(self.m_parameters,dtype=np.complex128)
        for n in xrange(v.shape[0]):
            update_w -= self._logderivative(v[n,:])
        update_w += v.shape[0]*self._logderivativenorm()    
        update_w /= v.shape[0]
        return update_w