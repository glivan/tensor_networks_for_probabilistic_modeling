# Expressive power of tensor-network factorizations for probabilistic modeling
This is the code accompanying the paper "Expressive power of tensor-network factorizations for probabilistic modeling" (Advances in Neural Information Processing Systems 32, proceedings of the NeurIPS 2019 Conference) which allows for reproduction of its numerical results. If you use this code or these results please cite [1].

## Prerequisite
A working python 2.7, 3.4 or more recent installation with the following python libraries (all included in Anaconda) :
```
numpy, scikit-learn, os, sys, pickle, time
```
For training a hidden Markov model the python library [pomegranate](https://github.com/jmschrei/pomegranate) [2] and its dependencies must also be installed. This is not necessary for running the tensor-network algorithms.

## Overview of the code
### Datasets
The preprocesssed datasets are included in the `datasets` folder. 
Preprocessing transformed the categorical data into a numpy array of integers. Each row corresponds to a training example and each column is an integer feature between 0 and d-1, where d is the number of different categories. As this work is only concerned with the expressivity of different functions, only training sets are used.

Included datasets :  
From the R package TraMineR :  
- Family life states from the Swiss Household Panel biographical survey : `biofam` [3]

From the UCI Machine Learning Repository [4]  
- SPECT Heart Data Set : `spect`
- Lymphography Data Set : `lymphography` [5]
- Primary Tumor Data Set : `tumor` [6]
- Congressional Voting Records Data Set : `votes`
- Solar Flare Data Set : `flare`

### Tensor networks
The `tensornetworks` folder includes a generic tensor network class `MPSClass.py` as well as classes for positive MPS, Born machine MPS and Locally Purified States (LPS) with real or complex tensor elements. These classes include simple methods for performing maximum likelihood estimation on a dataset using batch gradient descent. The training is done by computing the gradients of the log-likelihood over all tensors for each minibatch of training examples and then updating all tensors at once in a gradient descent optimization scheme. This is different from a DMRG-like algorithm where only one (or two) tensor is updated at a time. For this reason canonical forms (that would be different for each class of tensor network) are not used, but they might be required for the numerical stability over much larger datasets. The bond dimension/rank is fixed to the same value for all tensors. Code for approximating a given non-negative tensor representing a probability mass function is also available. Note that the code is not optimized for speed and performance, but is rather a tool demonstrating how the algorithms work.

### HMM
We include a simple script to define a hidden Markov model corresponding to an MPS of a certain bond dimension in the `hmm` folder. This script requires the pomegranate library.

## Running the code
We provide a jupyter notebook `fitdataset.ipynb` that explains how to create a model, load a dataset and train the model on the dataset. We also provide a jupyter notebook `fittensor.ipynb` that explains how to train the model to approximate a given non-negative tensor.

Input parameters of a tensor network (all parameters are optional):
- D : int [default: 2], bond dimension/rank of the tensor networks.
- learning_rate : float [default: 1.0], learning rate for gradient descent.
- batch_size : int [default: 20], number of training examples per minibatch.
- n_iter : int [default: 10], number of epochs over the training dataset to perform, or number of iterations of the optimization for approximating a given tensor.
- random_state : int or numpy.RandomState [default: None], a random number generator instance to define the state of the random permutations generator. If an integer is given, it fixes the seed. Defaults to the global numpy random number generator.
- verbose : int [default: 0], the verbosity level. Zero means silent mode.
- mu : int [default: 2], only for real and complex LPS : the dimension of the purification index.

Experiments in the paper used the following parameters:
- batch size was set to 20
- learning_rate was chosen using a grid search on powers of 10 going from 10<sup>-5</sup> to 10<sup>5</sup>.
- n_iter was set to a maximum of 20000  
Each data point indicated in the paper is the lowest negative log-likelihood obtained from 10 trials with different initial tensors.

For approximating a given non-negative tensor, the optimization is performed by a limited-memory BFGS algorithm. Batch size and learning rate parameters are not used, and experiments in the paper used a maximum number of iterations n_iter of 10000.

We also include code to train a hidden Markov model corresponding to an MPS with positive tensors. The training is performed using the Baum-Welch algorithm by running
```
python runHMM.py lymphography 2 100
```
Input parameters (all parameters are optional):
- datasetload : str [default: lymphography], Name of the dataset file which should be located in the datasets/ folder
- bond_dimension : int [default: 2], bond dimension/rank (here number of hidden states per variable)
- n_iter : int [default: 100], number of epochs over the training dataset to perform

## References
[1] Glasser, I., Sweke, R., Pancotti, N., Eisert, J., Cirac, J. I. (2019) Expressive power of tensor-network factorizations for probabilistic modeling. Advances in Neural Information Processing Systems 32 (Proceedings of the NeurIPS 2019 Conference). [https://papers.nips.cc/paper/8429-expressive-power-of-tensor-network-factorizations-for-probabilistic-modeling]. See also extended version at [arxiv:1907.03741](https://arxiv.org/abs/1907.03741).  
[2] Schreiber, J. (2018). Pomegranate: fast and flexible probabilistic modeling in python. Journal of Machine Learning Research, 18(164), 1-6.  
[3] Müller, N. S., M. Studer, G. Ritschard (2007). Classification de parcours de vie à l'aide de l'optimal matching. In XIVe Rencontre de la Société francophone de classification (SFC 2007), Paris, 5 - 7 septembre 2007, pp. 157–160.  
[4] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.  
[5] This lymphography domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Slovenia. Thanks go to M. Zwitter and M. Soklic for providing the data.  
[6] This primary tumor domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Slovenia. Thanks go to M. Zwitter and M. Soklic for providing the data.  
