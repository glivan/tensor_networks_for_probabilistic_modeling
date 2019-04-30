# -*- coding: utf-8 -*-

import time
import numpy as np
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_array
from sklearn.utils import check_random_state

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
      
    def _weightinitialization(self,rng):
        """Initialize weights w randomly
        Parameters
        ----------
        rng : random number generation
        """
        self.w = np.asarray(rng.normal(0, 1, self.m_parameters))


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
