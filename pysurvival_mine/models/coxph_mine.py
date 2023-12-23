from pysurvival_mine.models.semi_parametric import NonLinearCoxPHModel
import torch
import numpy as np
from pysurvival_mine import utils
from pysurvival_mine.utils import neural_networks as neur_net
from pysurvival_mine.utils import optimization as opt
from pysurvival_mine.models._coxph import _baseline_functions


class MineNonLinearCoxPHModel(NonLinearCoxPHModel):
    def predict_mine(self, x, t=None):
        """
        Predicting the hazard, density and survival functions

        Arguments:
            * x: pd.Dataframe or np.ndarray or list
                x is the testing dataset containing the features
                x should not be standardized before, the model
                will take care of it
            * t: float (default=None)
                Time at which hazard, density and survival functions
                should be calculated. If None, the method returns
                the functions for all times t.
        """

        # Convert x into the right format
        x = utils.check_data(x)

        if self.autoscaler:
            # Scaling the dataset
            if x.ndim == 1:
                x = self.scaler.transform(x.reshape(1, -1))
            elif x.ndim == 2:
                x = self.scaler.transform(x)

        # Calculating risk_score, hazard, density and survival
        score = self.model(torch.FloatTensor(x)).data.numpy().flatten()
        phi = np.exp(score)
        hazard = self.baseline_hazard * phi.reshape(-1, 1)
        survival = np.power(self.baseline_survival, phi.reshape(-1, 1))
        density = hazard * survival
        if t is None:
            return hazard, density, survival
        else:
            min_index = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_index)
            return hazard[:, index], density[:, index], survival[:, index]

    def fit(
        self,
        X,
        T,
        E,
        X_valid,
        T_valid,
        E_valid,
        init_method="glorot_uniform",
        optimizer="adam",
        lr=1e-4,
        num_epochs=1000,
        dropout=0.2,
        batch_normalization=False,
        bn_and_dropout=False,
        l2_reg=1e-5,
        verbose=True,
    ):

        # Checking data format (i.e.: transforming into numpy array)
        X, T, E = utils.check_data(X, T, E)

        # Extracting data parameters
        N, self.num_vars = X.shape
        input_shape = self.num_vars

        # Scaling data
        if self.auto_scaler:
            X_original = self.scaler.fit_transform(X)

        else:
            X_original = X

        # Sorting X, T, E in descending order according to T
        order = np.argsort(-T)
        T = T[order]
        E = E[order]
        X_original = X_original[order, :]
        self.times = np.unique(T[E.astype(bool)])
        self.nb_times = len(self.times)
        self.get_time_buckets()

        # Initializing the model
        model = neur_net.NeuralNet(
            input_shape,
            1,
            self.structure,
            init_method,
            dropout,
            batch_normalization,
            bn_and_dropout,
        )

        # Looping through the data to calculate the loss
        X = torch.FloatTensor(X_original)

        # Computing the Risk and Fail tensors
        Risk, Fail = self.risk_fail_matrix(T, E)
        Risk = torch.FloatTensor(Risk)
        Fail = torch.FloatTensor(Fail)

        # Computing Efron's matrices
        Efron_coef, Efron_one, Efron_anti_one = self.efron_matrix()
        Efron_coef = torch.FloatTensor(Efron_coef)
        Efron_one = torch.FloatTensor(Efron_one)
        Efron_anti_one = torch.FloatTensor(Efron_anti_one)

        # Performing order 1 optimization
        # def optimize_mine(model_wrapper, loss_function, model, optimizer_str, X, T, E, X_valid, T_valid, E_valid, lr=1e-4, nb_epochs=1000,
        #        verbose = True, num_workers = 0, **kargs):
        # Performing order 1 optimization
        horizons = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

        eval_times = np.quantile(T[E == 1], horizons).tolist()
        model, loss_values = opt.optimize_mine(
            self,
            self.loss_function,
            model,
            optimizer,
            X,
            T,
            E,
            X_valid,
            T_valid,
            E_valid,
            eval_times,
            X_original,
            lr,
            num_epochs,
            verbose,
            num_workers=0,
            Risk=Risk,
            Fail=Fail,
            Efron_coef=Efron_coef,
            Efron_one=Efron_one,
            Efron_anti_one=Efron_anti_one,
            l2_reg=l2_reg,
        )

        # Saving attributes
        self.model = model.eval()
        self.loss_values = loss_values

        # Computing baseline functions
        x = X_original
        x = torch.FloatTensor(x)

        # Calculating risk_score
        score = np.exp(self.model(torch.FloatTensor(x)).data.numpy().flatten())
        baselines = _baseline_functions(score, T, E)

        # Saving the Cython attributes in the Python object
        self.times = np.array(baselines[0])
        self.baseline_hazard = np.array(baselines[1])
        self.baseline_survival = np.array(baselines[2])

        return self

    def loss_function(
        self, model, X, T, E, Risk, Fail, Efron_coef, Efron_one, Efron_anti_one, l2_reg
    ):
        """Efron's approximation loss function by vectorizing
        all the quantities at stake
        """

        # Calculating the score
        pre_score = model(X)
        score = torch.reshape(torch.exp(pre_score), (-1, 1))
        max_nb_fails = Efron_coef.shape[1]

        # Numerator calculation
        log_score = torch.log(score)
        log_fail = torch.mm(Fail, log_score)
        numerator = torch.sum(log_fail)

        # Denominator calculation
        risk_score = torch.reshape(torch.mm(Risk, score), (-1, 1))
        risk_score = risk_score.repeat(1, max_nb_fails)

        fail_score = torch.reshape(torch.mm(Fail, score), (-1, 1))
        fail_score = fail_score.repeat(1, max_nb_fails)

        Efron_Fail = fail_score * Efron_coef
        Efron_Risk = risk_score * Efron_one
        log_efron = torch.log(Efron_Risk - Efron_Fail + Efron_anti_one)

        denominator = torch.sum(torch.sum(log_efron, dim=1))

        # Adding regularization
        loss = -(numerator - denominator)
        for w in model.parameters():
            loss += l2_reg * torch.sum(w * w) / 2.0

        return loss

    def predict(self, x, t=None, model=None, T=None, E=None, X_original=None):
        """
        Predicting the hazard, density and survival functions

        Arguments:
            * x: pd.Dataframe or np.ndarray or list
                x is the testing dataset containing the features
                x should not be standardized before, the model
                will take care of it
            * t: float (default=None)
                Time at which hazard, density and survival functions
                should be calculated. If None, the method returns
                the functions for all times t.
        """

        # Convert x into the right format
        x = utils.check_data(x)

        if self.auto_scaler:

            # Scaling the dataset
            if x.ndim == 1:
                x = self.scaler.transform(x.reshape(1, -1))
            elif x.ndim == 2:
                x = self.scaler.transform(x)

        # Calculating risk_score, hazard, density and survival
        if model is not None:
            score = model(torch.FloatTensor(x)).data.numpy().flatten()
            score_orig = model(torch.FloatTensor(X_original)).data.numpy().flatten()
            baselines = _baseline_functions(score_orig, T, E)
            baseline_hazard = np.array(baselines[1])
            baseline_survival = np.array(baselines[2])
            phi = np.exp(score)
            hazard = baseline_hazard * phi.reshape(-1, 1)
            survival = np.power(baseline_survival, phi.reshape(-1, 1))
            density = hazard * survival
            if t is None:
                return hazard, density, survival
            else:
                min_index = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
                index = np.argmin(min_index)
                return hazard[:, index], density[:, index], survival[:, index]
        else:
            score = self.model(torch.FloatTensor(x)).data.numpy().flatten()
            phi = np.exp(score)
            hazard = self.baseline_hazard * phi.reshape(-1, 1)
            survival = np.power(self.baseline_survival, phi.reshape(-1, 1))
            density = hazard * survival
            if t is None:
                return hazard, density, survival
            else:
                min_index = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
                index = np.argmin(min_index)
                return hazard[:, index], density[:, index], survival[:, index]

    def predict_risk_mine(self, model, x, use_log=False):
        """
        Predicting the risk score functions

        Arguments:
            * x: pd.Dataframe or np.ndarray or list
                x is the testing dataset containing the features
                x should not be standardized before, the model
                will take care of it
        """
        # Convert x into the right format
        x = utils.check_data(x)

        # Scaling the data
        if self.auto_scaler:
            if x.ndim == 1:
                x = self.scaler.transform(x.reshape(1, -1))
            elif x.ndim == 2:
                x = self.scaler.transform(x)
        else:
            # Ensuring x has 2 dimensions
            if x.ndim == 1:
                x = np.reshape(x, (1, -1))

        # Transforming into pytorch objects
        x = torch.FloatTensor(x)

        # Calculating risk_score
        score = model(x).data.numpy().flatten()
        if not use_log:
            score = np.exp(score)

        return score

    def predict_survival_mine(self, model, x, t=None, T=None, E=None, X_original=None):
        hazard, density, survival = self.predict(
            x, model=model, T=T, E=E, X_original=X_original
        )
        return survival
