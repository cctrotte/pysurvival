import random
import numpy as np
import torch

def make_train_test_split(X, T, E, ID, index_train, index_valid, index_test):


    # Creating the X, T and E inputs
    X_train, X_valid, X_test = X.iloc[index_train], X.iloc[index_valid], X.iloc[index_test]
    T_train, T_valid, T_test = T[index_train],T[index_valid], T[index_test]
    E_train, E_valid, E_test = E[index_train],E[index_valid], E[index_test]
    ID_train, ID_valid, ID_test = ID[index_train],ID[index_valid], ID[index_test]
    
    return {'X_train':X_train, 'T_train':T_train, 'E_train':E_train, 'ID_train':ID_train, 'index_train':index_train,
            'X_valid':X_valid, 'T_valid':T_valid, 'E_valid':E_valid, 'ID_valid':ID_valid, 'index_valid':index_valid,
           'X_test':X_test, 'T_test':T_test, 'E_test':E_test, 'ID_test':ID_test, 'index_test':index_test}

def set_seeds():

    # Set seeds for random, numpy, PyTorch
    seed_value = 42

    # Set seed for the random module
    random.seed(seed_value)

    # Set seed for numpy
    np.random.seed(seed_value)

    # Set seed for PyTorch
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    return