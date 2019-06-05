# -*- coding: utf-8 -*-

import numpy as np
import sys
import pickle
import time
import pomegranate

def init(datasetload_init='lymphography',
         bond_dimension_init='2',n_iter_init='100'):
    """Initialize parameters :
        ----------
        datasetload : str, path of dataset
        bond_dimension : int, number of hidden states
        n_iter : int, Number of iterations over the training dataset to perform
    """
    global datasetload
    global bond_dimension
    global n_iter
    
    datasetload=str(datasetload_init)
    bond_dimension=int(bond_dimension_init)
    n_iter=int(n_iter_init)
        
def run():
    # Load dataset
    path='datasets/'
    with open(path+datasetload, 'rb') as f:
        a=pickle.load(f)
    X=a[0]
    X=X.astype(int)
    
    # Create HMM
    D=bond_dimension
    N=X.shape[1]
    d=np.max(X+1)
    list_of_states=[]
    for i in xrange(N):
        list_of_states.append([])
        for u in xrange(bond_dimension):
            dictionnary=dict()
            for l in xrange(d):
                dictionnary[str(l)] = np.random.rand()
            list_of_states[i].append(pomegranate.State(pomegranate.DiscreteDistribution(dictionnary)))
    model = pomegranate.HiddenMarkovModel()
    for i in xrange(N-1):
        for d in xrange(D):
            for d2 in xrange(D):
                model.add_transition(list_of_states[i][d],list_of_states[i+1][d2],np.random.rand())
    for d in xrange(D):
        model.add_transition(model.start,list_of_states[0][d],np.random.rand())
    for d in xrange(D):
        model.add_transition(list_of_states[N-1][d],model.end,np.random.rand())
    model.bake()

    # Train HMM
    begin = time.time()  
    sequencetrain=[[str(i) for i in v] for v in X]
    np.random.seed()
    model.fit(sequencetrain,algorithm='baum-welch',stop_threshold=1e-50,min_iterations=1000,\
              max_iterations=n_iter)
    
    u=0
    for i in sequencetrain:
        u+=model.log_probability(i)
    accuracy=-u/len(sequencetrain)

    time_elapsed = time.time()-begin
    
    print("Negative log likelihood = %.3f" % (accuracy))
    print("Time elapsed = %.2fs" %(time_elapsed))

if __name__ == '__main__':
    # Main program : initialize with options from command line and run
    init(*sys.argv[1::])
    run()