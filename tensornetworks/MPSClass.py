# -*- coding: utf-8 -*-

import time
import numpy as np
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from functools import partial
from scipy.optimize import minimize
import itertools

class MPS():
    """Generic Tensor Network Class for Matrix Product States
    This class should not be used directly. Use derived classes instead.
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
        pass
    
    def _computenorm(self): 
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """
        pass
    
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
        pass
    
    def _derivativenorm(self): 
        """Compute the derivative of the norm
        Returns
        -------
        derivative : numpy array, shape (n_parameters,)
        """
        pass
    
    def _logderivative(self, x):
        """Compute the logderivative of P(x)
        Parameters
        ----------
        x : numpy array, shape (n_features,)
            One configuration
        Returns
        -------
        derivative : numpy array, shape (n_parameters,)
        """
        derivative=self._derivative(x)/self._probability(x)
        return derivative

    def _logderivativenorm(self):
        """Compute the logderivative of the norm
        Returns
        -------
        derivative : numpy array, shape (n_parameters,)
        """
        derivative=self._derivativenorm()/self.norm
        return derivative
        
    def _fit(self, v):
        """Inner fit for one mini-batch of training 
        Updatest the parameters and recomputes the norm
        Parameters
        ----------
        v : numpy array, shape (n_samples, n_features)
            The data to use for training.
        """
        update_w = self._likelihood_derivative(v)
        self.w -= self.learning_rate * update_w
        self.norm = self._computenorm()

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
        update_w=np.zeros(self.m_parameters)
        for n in xrange(v.shape[0]):
            update_w -= self._logderivative(v[n,:])
        update_w += v.shape[0]*self._logderivativenorm()    
        update_w /= v.shape[0]
        return update_w
        
    def likelihood(self, v, w=None):
        """Compute averaged negative log-likelihood of configurations in v
        Parameters
        ----------
        v : numpy array, shape (n_samples,n_features)
            dataset to compute the likelihood of
        w : parameters of tensor network (optional)
        Returns
        -------
        loglikelihood : float
            averaged negative log-likelihood of the data in v
        """
        loglikelihood=0
        if w is not None:
            self.w=w
        self.norm=self._computenorm()
        for n in xrange(v.shape[0]):
            loglikelihood+=np.log(max(self._probability(v[n,:])/self.norm,10** (-50)))
        return -loglikelihood/v.shape[0]

    def distance(self, X, w=None):
        """Compute distance (here KL-divergence) between tensor X and MPS
        Parameters
        ----------
        X : array-like, shape (d, d, d, d,...) (dimension d^n_features)
            Tensor to fit
        w : parameters of MPS
        """
        distance=0
        epsilon=10**(-10)
        if w is not None:
            self.w=self.padding_function(w)
        self.norm=self._computenorm()
        for i in itertools.product(np.arange(0,self.d), repeat = self.n_features):
            var=np.array(i)
            b=self._probability(var)/self.norm
            a=X[tuple(var)]
            if a<epsilon:
                distance+=-a*np.log(b)
            else:
                distance+=a*np.log(a)-a*np.log(b)
        return distance

    def function_real_to_complex(self, function, X, w=None):
        derivative=function(X,w.view(self.w.dtype))
        return derivative.view(np.float64)   

    def padding_function(self, w):
        new_w=np.zeros((self.n_features,self.d,self.D,self.D),dtype=w.dtype)
        new_w[0,:,0,:]=w[0:self.D*self.d].reshape(self.d,self.D)
        new_w[1:self.n_features-1,:,:,:]=w[self.D*self.d*2:].reshape((self.n_features-2,self.d,self.D,self.D))
        new_w[self.n_features-1,:,:,0]=w[self.D*self.d:self.D*self.d*2].reshape(self.d,self.D)
        return new_w.reshape(self.m_parameters)

    def unpadding_function(self, w):
        w=w.reshape((self.n_features,self.d,self.D,self.D))
        new_w=np.zeros(self.m_parameters2,dtype=w.dtype)
        new_w[0:self.D*self.d]=w[0,:,0,:].reshape(self.d*self.D)
        new_w[self.D*self.d:self.D*self.d*2]=w[self.n_features-1,:,:,0].reshape(self.d*self.D)
        new_w[self.D*self.d*2:]=w[1:self.n_features-1,:,:,:].reshape((self.n_features-2)*self.d*self.D*self.D)
        return new_w
        
    def _derivativedistance(self, X, w=None):
        """Compute derivative of the distance (here KL-divergence) between tensor X and MPS
        Parameters
        ----------
        X : array-like, shape (d, d, d, d,...) (dimension d^n_features)
            Tensor to fit
        w : parameters of MPS
        """
        derivative=np.zeros(self.m_parameters,dtype=w.dtype)
        
        if w is not None:
            self.w=self.padding_function(w)
            
        self.norm=self._computenorm()
        Zlogderivative=self._logderivativenorm()
        for i in itertools.product(np.arange(0,self.d), repeat = self.n_features):
            var=np.array(i)
            a=X[tuple(var)]
            derivative-=a*(self._logderivative(var)-Zlogderivative)
        return self.unpadding_function(derivative)
        
    def _weightinitialization(self,rng):
        """Initialize weights w randomly
        Parameters
        ----------
        rng : random number generation
        """
        self.w = np.asarray(rng.normal(0, 1, self.m_parameters))

    def _weightinitialization2(self,rng):
        """Initialize weights w randomly
        Parameters
        ----------
        rng : random number generation
        """
        self.m_parameters2=(self.n_features-2)*self.d*self.D*self.D+2*self.D*self.d
        return np.asarray(rng.rand(self.m_parameters2))
        
    def _gen_even_slices(self, batch_size, n_batches, n_samples, rng):
        """Generate batch slices of a dataset
        Parameters
        ----------
        batch_size : int
            batch_size
        n_batches : int
            number of batches
        n_samples : int
            number of samples in the dataset
        rng : random number generation
        """
        start = 0
        array_rand=rng.permutation(n_samples)
        for pack_num in range(n_batches):
            this_n = batch_size // n_batches
            if pack_num < batch_size % n_batches:
                this_n += 1
            if this_n > 0:
                end = start + this_n
                if n_samples is not None:
                    end = min(n_samples, end)
                yield array_rand[np.arange(start, end)]
                start = end
            
    def fit(self, X, w_init=None):
        """Fit the model to the data X, with parameters initialized at w_init
        Parameters
        ----------
        X : {numpy array, integer matrix} shape (n_samples, n_features)
            Training data.
        w_init : {numpy array, float or complex} shape (n_parameters,) (optional)
            Initial value of the parameters
        Returns
        -------
        self : MPS
            The fitted model.
        """

#       Some initial checks of the data, initialize random number generator
        X = check_array(X, dtype=np.int64)
        rng = check_random_state(self.random_state)
#        print(X.shape)
#       Initialize parameters of MPS
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.d = np.max(X)+1
        self.m_parameters = self.n_features*self.d*self.D*self.D
        if w_init is None:
            self._weightinitialization(rng)
        else:
            self.w=w_init
        self.norm=self._computenorm()
        self.history=[]

        n_batches = int(np.ceil(float(self.n_samples) / self.batch_size))
        begin = time.time()
        for iteration in xrange(1, self.n_iter + 1):
            batch_slices = list(self._gen_even_slices(self.batch_size,
                                            n_batches, self.n_samples, rng))
            for batch_slice in batch_slices:
                self._fit(X[batch_slice])  

            end = time.time()
                
            
            if self.verbose:
                train_likelihood=self.likelihood(X)
                print("Iteration %d, likelihood = %.3f,"
                  " time = %.2fs"
                  % (iteration,train_likelihood,
                     end - begin))
                self.history.append(train_likelihood)
            begin = end

        return self

 
    def fit_tensor(self, X, w_init=None):
        """Fit the model to the tensor X, with parameters initialized at w_init
        Parameters
        ----------
        X : {numpy array, integer matrix} shape (d, d, d, d,...) (dimension d^n_features)
            Training data.
        w_init : {numpy array, float or complex} shape (n_parameters,) (optional)
            Initial value of the parameters
        Returns
        -------
        self : MPS
            The fitted model.
        """

#       Some initial checks of the data, initialize random number generator
        rng = check_random_state(self.random_state)
        X=X/np.sum(X) #Tensor needs to be normalized to be a probability mass function
        
#       Initialize parameters of MPS
        self.d = int(X.shape[1])
        self.n_features = len(X.shape)
        
        self.n_samples = self.d**self.n_features
        self.m_parameters = self.n_features*self.d*self.D*self.D
        
        if w_init is None:
            self._weightinitialization(rng)
        else:
            self.w=w_init
        self.norm=self._computenorm()
        self.history=[]

        begin = time.time()
 
        distancepartial = partial(self.function_real_to_complex,self.distance,X)
        derivativedistancepartial = partial(self.function_real_to_complex,self._derivativedistance,X)  

        initial_value=self._weightinitialization2(rng)

        res=minimize(fun=distancepartial,jac=derivativedistancepartial,x0=initial_value.view(np.float64),\
                     method='BFGS',options={'maxiter': self.n_iter},tol=10**(-16))


        self.w=self.padding_function(res.x.view(self.w.dtype))
        
        self.norm=self._computenorm()
        end = time.time()
        print("KL divergence = %.6f, time = %.2fs" % (self.distance(X),end - begin))
        return self

