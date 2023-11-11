from __future__ import absolute_import
import torch 
import numpy as np
import copy
import multiprocessing
from pysurvival_mine import HAS_GPU
from pysurvival_mine import utils
from pysurvival_mine.utils import neural_networks as nn
from pysurvival_mine.utils import optimization as opt
from pysurvival_mine.models.multi_task import  BaseMultiTaskModel, norm_diff


class BaseMultiTaskModelMine(BaseMultiTaskModel):


    def fit(self, X, T, E,  X_valid, T_valid, E_valid, init_method = 'glorot_uniform', optimizer ='adam', 
            lr = 1e-4, num_epochs = 1000, dropout = 0.2, l2_reg=1e-2, 
            l2_smooth=1e-2, batch_normalization=False, bn_and_dropout=False,
            verbose=True, extra_pct_time = 0.1, is_min_time_zero=True):
       
        # Checking data format (i.e.: transforming into numpy array)
        X, T, E = utils.check_data(X, T, E)

        # Extracting data parameters
        nb_units, self.num_vars = X.shape
        input_shape = self.num_vars
    
        # Scaling data 
        if self.auto_scaler:
            X = self.scaler.fit_transform( X ) 
            X_valid = self.scaler.transform( X_valid )

        # Building the time axis, time buckets and output Y
        X_cens, X_uncens, Y_cens, Y_uncens \
            = self.compute_XY(X, T, E, is_min_time_zero, extra_pct_time)
       
        # Initializing the model
        model = nn.NeuralNet(input_shape, self.num_times, self.structure, 
                             init_method, dropout, batch_normalization, 
                             bn_and_dropout )

        # Creating the Triangular matrix
        Triangle = np.tri(self.num_times, self.num_times + 1, dtype=np.float32) 
        Triangle = torch.FloatTensor(Triangle)

        # Performing order 1 optimization
        model, loss_values = opt.optimize_mine(self, self.loss_function, model, optimizer, 
            X, T, E, X_valid, T_valid, E_valid, lr, num_epochs, verbose,  num_workers=0, X_cens=X_cens, X_uncens=X_uncens, 
            Y_cens=Y_cens, Y_uncens=Y_uncens, Triangle=Triangle, 
            l2_reg=l2_reg, l2_smooth=l2_smooth)

        # Saving attributes
        self.model = model.eval()
        self.loss_values = loss_values

        return self
    
    def loss_function(self, model, X, T, E, X_cens, X_uncens, Y_cens, Y_uncens, 
            Triangle, l2_reg, l2_smooth):
            """ Computes the loss function of the any MTLR model. 
                All the operations have been vectorized to ensure optimal speed
            """

            # Likelihood Calculations -- Uncensored
            score_uncens = model(X_uncens)
            phi_uncens = torch.exp( torch.mm(score_uncens, Triangle) )
            reduc_phi_uncens = torch.sum(phi_uncens*Y_uncens, dim = 1)

            # Likelihood Calculations -- Censored
            score_cens = model(X_cens)
            phi_cens = torch.exp( torch.mm(score_cens, Triangle) )
            reduc_phi_cens = torch.sum( phi_cens*Y_cens, dim = 1)

            # Likelihood Calculations -- Normalization
            z_uncens = torch.exp( torch.mm(score_uncens, Triangle) )
            reduc_z_uncens = torch.sum( z_uncens, dim = 1)

            z_cens = torch.exp( torch.mm(score_cens, Triangle) )
            reduc_z_cens = torch.sum( z_cens, dim = 1)

            # MTLR cost function
            loss = - (
                        torch.sum( torch.log(reduc_phi_uncens) ) \
                    + torch.sum( torch.log(reduc_phi_cens) )  \

                    - torch.sum( torch.log(reduc_z_uncens) ) \
                    - torch.sum( torch.log(reduc_z_cens) ) 
                    )

            # Adding the regularized loss
            nb_set_parameters = len(list(model.parameters()))
            for i, w in enumerate(model.parameters()):
                loss += l2_reg*torch.sum(w*w)/2.
                
                if i >= nb_set_parameters - 2:
                    loss += l2_smooth*norm_diff(w)
                    
            return loss

    def predict(self, x, t = None, model = None):
        """ Predicting the hazard, density and survival functions
        
        Parameters:
        ----------
        * `x` : **array-like** *shape=(n_samples, n_features)* --
            array-like representing the datapoints. 
            x should not be standardized before, the model
            will take care of it

        * `t`: **double** *(default=None)* --
             time at which the prediction should be performed. 
             If None, then return the function for all available t.
        """
        
        # Convert x into the right format
        x = utils.check_data(x)

        # Scaling the data
        if self.auto_scaler:
            if x.ndim == 1:
                x = self.scaler.transform( x.reshape(1, -1) )
            elif x.ndim == 2:
                x = self.scaler.transform( x )
        else:
            # Ensuring x has 2 dimensions
            if x.ndim == 1:
                x = np.reshape(x, (1, -1))

        # Transforming into pytorch objects
        x = torch.FloatTensor(x)
                
        # Predicting using linear/nonlinear function
        if model is None:
            score_torch = self.model(x)
        else:
            score_torch = model(x)
        score = score_torch.data.numpy()
                
        # Cretaing the time triangles
        Triangle1 = np.tri(self.num_times , self.num_times + 1 )
        Triangle2 = np.tri(self.num_times+1 , self.num_times + 1 )

        # Calculating the score, density, hazard and Survival
        phi = np.exp( np.dot(score, Triangle1) )
        div = np.repeat(np.sum(phi, 1).reshape(-1, 1), phi.shape[1], axis=1)
        density = (phi/div)
        Survival = np.dot(density, Triangle2)
        hazard = density[:, :-1]/Survival[:, 1:]

        # Returning the full functions of just one time point
        if t is None:
            return hazard, density, Survival
        else:
            min_abs_value = [abs(a_j_1-t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_abs_value)
            return hazard[:, index], density[:, index], Survival[:, index]


    def predict_risk(self, x, use_log=False):
        """ Computing the risk score 

        Parameters:
        -----------
        * `x` : **array-like** *shape=(n_samples, n_features)* --
            array-like representing the datapoints. 
            x should not be standardized before, the model
            will take care of it

        * `use_log`: **bool** *(default=True)* -- 
            Applies the log function to the risk values

        """

        risk = super(BaseMultiTaskModel, self).predict_risk(x)
        if use_log:
            return np.log(risk)
        else:
            return risk
        
    def predict_risk_mine(self, model, x, use_log=False):
        """ Computing the risk score 

        Parameters:
        -----------
        * `x` : **array-like** *shape=(n_samples, n_features)* --
            array-like representing the datapoints. 
            x should not be standardized before, the model
            will take care of it

        * `use_log`: **bool** *(default=True)* -- 
            Applies the log function to the risk values

        """
        
        hazard, density, survival = self.predict( x, model = model)
        cumulative_hazard = np.cumsum(hazard, 1)
        risk = np.sum(cumulative_hazard, 1)
        if use_log:
            return np.log(risk)
        else:
            return risk

class LinearMultiTaskModelMine(BaseMultiTaskModelMine):
    """ LinearMultiTaskModel is the original Multi-Task model, 
        a.k.a the Multi-Task Logistic Regression model (MTLR).
        It was first introduced by  Chun-Nam Yu et al. in 
        Learning Patient-Specific Cancer Survival Distributions 
        as a Sequence of Dependent Regressors
        
        Reference:
        ----------
            * http://www.cs.cornell.edu/~cnyu/papers/nips11_survival.pdf
        
        Parameters:
        ----------
            * bins: int 
                Number of subdivisions of the time axis 

            * auto_scaler: boolean (default=True)
                Determines whether a sklearn scaler should be automatically 
                applied

    """
    
    def __init__(self, bins = 100, auto_scaler=True):
        super(LinearMultiTaskModelMine, self).__init__(
            structure = None, bins = bins, auto_scaler=auto_scaler)


    def fit(self, X, T, E, X_valid, T_valid, E_valid, init_method = 'glorot_uniform', optimizer ='adam', 
            lr = 1e-4, num_epochs = 1000, l2_reg=1e-2, l2_smooth=1e-2, 
            verbose=True, extra_pct_time = 0.1, is_min_time_zero=True):

        super(LinearMultiTaskModelMine, self).fit(X=X, T=T, E=E, 
            X_valid=X_valid, T_valid=T_valid, E_valid=E_valid,
            init_method = init_method, optimizer =optimizer, 
            lr = lr, num_epochs = num_epochs, dropout = None, l2_reg=l2_reg, 
            l2_smooth=l2_smooth, batch_normalization=False, 
            bn_and_dropout=False, verbose=verbose, 
            extra_pct_time = extra_pct_time, is_min_time_zero=is_min_time_zero)

        return self

class NeuralMultiTaskModelMine(BaseMultiTaskModelMine):
    def __init__(self, structure, bins = 100, auto_scaler = True):

        # Checking the validity of structure
        structure = nn.check_mlp_structure(structure)

        # Initializing the instance
        super(NeuralMultiTaskModelMine, self).__init__(
            structure = structure, bins = bins, auto_scaler = auto_scaler)
    
    
    def __repr__(self):
        """ Representing the class object """

        if self.structure is None:
            super(NeuralMultiTaskModelMine, self).__repr__()
            return self.name
            
        else:
            S = len(self.structure)
            self.name = self.__class__.__name__
            empty = len(self.name)
            self.name += '( '
            for i, s in enumerate(self.structure):
                n = 'Layer({}): '.format(i+1)
                activation = nn.activation_function(s['activation'], 
                    return_text=True)
                n += 'activation = {}, '.format( s['activation'] )
                n += 'units = {} '.format( s['num_units'] )
                
                if i != S-1:
                    self.name += n + '; \n'
                    self.name += empty*' ' + '  '
                else:
                    self.name += n
            self.name = self.name + ')'
            return self.name

