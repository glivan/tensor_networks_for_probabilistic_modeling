# Probabilistic modelling with tensor networks
This is the code accompanying the paper "Probabilistic modelling with tensor networks, hidden Markov models and quantum circuits" and allowing to reproduce its numerical results.

## Prerequisite
A working python 2.7 or 3.7 installation with the following python libraries :
```
numpy, scikit-learn, os, sys, pickle, time
```

## Overview of the code
### Datasets
The preprocesssed datasets are included in the `datasets` folder. 
Preprocessing transformed the categorical data into a numpy array of integers. Each row corresponds to a training example and each column is an integer feature between 0 and d-1, where d is the number of different categories. As this work is only concerned with the expressivity of different functions, only training sets are included.

Included datasets :  
From the R package TraMineR :  
- Family life states from the Swiss Household Panel biographical survey : `biofam` [1]

From the UCI Machine Learning Repository [2]  
- SPECT Heart Data Set : `spect`
- Lymphography Data Set : `lymphography` [3]
- Primary Tumor Data Set : `tumor` [4]
- Congressional Voting Records Data Set : `votes`
- Solar Flare Data Set : `flare`

### Tensor networks
The `tensornetworks` folder includes a generic tensor network class `MPSClass.py` as well as classes for positive MPS, Born machine MPS and Locally Purified States (LPS) with real or complex tensor elements. These classes include simple methods for performing maximum likelihood estimation on a dataset using batch gradient descent. The training is done by computing the gradients of the log-likelihood over all tensors for each batch of training example and then updating all tensors at once in a gradient descent optimization scheme. This is different from a DMRG-like algorithm where only one (or two) tensor is updated at a time. For this reason canonical forms (that would be different for each class of tensor network) are not used. Note that the code is not optimized for speed, but rather for simplicity and being easily understandable.

## Running the code
We provide a jupyter notebook `MPStutorial.ipynb` that explains how to create a model, load a dataset and train the model on the dataset, as well as a python script `RunMPSunsupervised.py` to run maximum likelihood estimation with all parameters from command line.

Run Maximum Likelihood Estimation with a tensor network on a dataset:
```
python RunMPSunsupervised.py lymphography 20 1.0 2 10 squarecomplex
```
Input parameters (all parameters are optional):
- datasetload : str [default: lymphography], Name of the dataset file which should be located in the datasets/ folder
- batch_size : int [default: 20], number of training examples per minibatch.
- learning_rate : float [default: 1.0], learning rate for gradient descent
- bond_dimension : int [default: 2], bond dimension/rank of the tensor networks
- n_iter : int [default: 10], number of epochs over the training dataset to perform
- ansatz : str [default: squarecomplex], choice of tensor network ansatz, between 'positive', 'squarereal', 'squarecomplex', 'realLPS', 'complexLPS'
- seed : int [default: None], choice of integer seed for random number generation
- save : int [default: 0], if equal to 1, save the optimized tensor network to disk

Experiments in the paper used the following parameters:
- batch size was set to 20
- learning_rate was chosen using a grid search on powers of 10 going from 10<sup>-5</sup> to 10<sup>5</sup>.
- n_iter was set to a maximum of 20000


Datasets bibliography:

[1] Müller, N. S., M. Studer, G. Ritschard (2007). Classification de parcours de vie à l'aide de l'optimal matching. In XIVe Rencontre de la Société francophone de classification (SFC 2007), Paris, 5 - 7 septembre 2007, pp. 157–160.  
[2] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.  
[3] This lymphography domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia. Thanks go to M. Zwitter and M. Soklic for providing the data.  
[4] This primary tumor domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia. Thanks go to M. Zwitter and M. Soklic for providing the data.
