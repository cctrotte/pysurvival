import numpy as np
import torch
import torch.nn as nn
import progressbar
import time
import copy
from pysurvival_mine.utils.metrics import concordance_index_mine, other_metrics


def initialization(init_method, W, is_tensor=True):
    """Initializes the provided tensor.

    Parameters:
    -----------

    * init_method : str (default = 'glorot_uniform')
        Initialization method to use. Here are the possible options:
            * 'glorot_uniform': Glorot/Xavier uniform initializer,
                Glorot & Bengio, AISTATS 2010
                http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
            * 'he_uniform': He uniform variance scaling initializer
               He et al., http://arxiv.org/abs/1502.01852
            * 'uniform': Initializing tensors with uniform (-1, 1) distribution
            * 'glorot_normal': Glorot normal initializer,
            * 'he_normal': He normal initializer.
            * 'normal': Initializing tensors with standard normal distribution
            * 'ones': Initializing tensors to 1
            * 'zeros': Initializing tensors to 0
            * 'orthogonal': Initializing tensors with an orthogonal matrix,

    * W: torch.Tensor
        Corresponds to the Torch tensor

    """

    # Checking the dimensions
    is_one_dim = False

    # Checking if the parameters is a tensor, if not transform it into one
    if not is_tensor:
        W = torch.FloatTensor(W)

    # Creating a column vector if one dimensional tensor
    if len(W.shape) == 1:
        is_one_dim = True
        W = torch.reshape(W, (1, -1))

    # Initializing the weights
    if init_method.lower() == "uniform":
        W = nn.init.uniform_(W)

    elif init_method.lower() == "normal":
        W = nn.init.normal_(W)

    elif init_method.lower().startswith("one"):
        W = nn.init.ones_(W)

    elif init_method.lower().startswith("zero"):
        W = nn.init.zeros_(W)

    elif init_method.lower().startswith("ortho"):
        W = nn.init.orthogonal_(W)

    elif init_method.lower().startswith("glorot") or init_method.lower().startswith(
        "xav"
    ):

        if init_method.lower().endswith("uniform"):
            W = nn.init.xavier_uniform_(W)
        elif init_method.lower().endswith("normal"):
            W = nn.init.xavier_normal_(W)

    elif init_method.lower().startswith("he") or init_method.lower().startswith(
        "kaiming"
    ):

        if init_method.lower().endswith("uniform"):
            W = nn.init.kaiming_uniform_(W)
        elif init_method.lower().endswith("normal"):
            W = nn.init.kaiming_normal_(W)

    else:
        error = " {} isn't implemented".format(init_method)
        raise NotImplementedError(error)

    # Returning a PyTorch tensor
    if is_tensor:
        if is_one_dim:
            return torch.nn.Parameter(W.flatten())
        else:
            return torch.nn.Parameter(W)

    # Returning a Numpy array
    else:
        if is_one_dim:
            return W.data.numpy().flatten()
        else:
            return W.data.numpy()


def optimize(
    loss_function,
    model,
    optimizer_str,
    lr=1e-4,
    nb_epochs=1000,
    verbose=True,
    num_workers=0,
    **kargs
):
    """
    Providing the schema of the iterative method for optimizing a
    differentiable objective function for models that use gradient centric
    schemas (a.k.a order 1 optimization)

    Parameters:
    -----------
        * loss_function: function
            Loss function of the model

        * model: torch object
            Actual model to optimize

        * optimizer_str: str
            Defines the type of optimizer to use. Here are the possible options:
                - adadelta
                - adagrad
                - adam
                - adamax
                - rmsprop
                - sparseadam
                - sgd

        * lr: float (default=1e-4)
            learning reate used in the optimization

        * nb_epochs: int (default=1000)
            The number of iterations in the optimization

        * verbose: bool (default=True)
            Whether or not producing detailed logging about the modeling
    """

    # Choosing an optimizer
    W = model.parameters()
    if optimizer_str.lower() == "adadelta":
        optimizer = torch.optim.Adadelta(W, lr=lr)

    elif optimizer_str.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(W, lr=lr)

    elif optimizer_str.lower() == "adam":
        optimizer = torch.optim.Adam(W, lr=lr)

    elif optimizer_str.lower() == "adamax":
        optimizer = torch.optim.Adamax(W, lr=lr)

    elif optimizer_str.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(W, lr=lr)

    elif optimizer_str.lower() == "sparseadam":
        optimizer = torch.optim.SparseAdam(W, lr=lr)

    elif optimizer_str.lower() == "sgd":
        optimizer = torch.optim.SGD(W, lr=lr)

    elif optimizer_str.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS(W, lr=lr)

    elif optimizer_str.lower() == "rprop":
        optimizer = torch.optim.Rprop(W, lr=lr)

    else:
        error = "{} optimizer isn't implemented".format(optimizer_str)
        raise NotImplementedError(error)

    # Initializing the Progress Bar
    loss_values = []
    if verbose:
        widgets = ["% Completion: ", progressbar.Percentage(), progressbar.Bar("*"), ""]
        bar = progressbar.ProgressBar(maxval=nb_epochs, widgets=widgets)
        bar.start()

    # Updating the weights at each training epoch
    temp_model = None
    for epoch in range(nb_epochs):

        # Backward pass and optimization
        def closure():
            optimizer.zero_grad()
            loss = loss_function(model, **kargs)
            loss.backward()
            return loss

        if "lbfgs" in optimizer_str.lower():
            optimizer.step(closure)
        else:
            optimizer.step()
        loss = closure()
        loss_value = loss.item()

        # Printing error message if the gradient didn't explode
        if np.isnan(loss_value) or np.isinf(loss_value):
            error = "The gradient exploded... "
            error += "You should reduce the learning"
            error += "rate (lr) of your optimizer"
            if verbose:
                widgets[-1] = error
            else:
                print(error)
            break

        # Otherwise, printing value of loss function
        else:
            temp_model = copy.deepcopy(model)
            loss_values.append(loss_value)
            if verbose:
                widgets[-1] = "Loss: {:6.2f}".format(loss_value)

        # Updating the progressbar
        if verbose:
            bar.update(epoch + 1)

    # Terminating the progressbar
    if verbose:
        bar.finish()

    # Finilazing the model
    if temp_model is not None:
        temp_model = temp_model.eval()
        model = copy.deepcopy(temp_model)
    else:
        raise ValueError(error)

    return model, loss_values


def optimize_mine(
    model_wrapper,
    loss_function,
    model,
    optimizer_str,
    X,
    T,
    E,
    X_valid,
    T_valid,
    E_valid,
    eval_times=None,
    X_original=None,
    lr=1e-4,
    nb_epochs=1000,
    verbose=True,
    num_workers=0,
    **kargs
):

    """
    Providing the schema of the iterative method for optimizing a
    differentiable objective function for models that use gradient centric
    schemas (a.k.a order 1 optimization)

    Parameters:
    -----------
        * loss_function: function
            Loss function of the model

        * model: torch object
            Actual model to optimize

        * optimizer_str: str
            Defines the type of optimizer to use. Here are the possible options:
                - adadelta
                - adagrad
                - adam
                - adamax
                - rmsprop
                - sparseadam
                - sgd

        * lr: float (default=1e-4)
            learning reate used in the optimization

        * nb_epochs: int (default=1000)
            The number of iterations in the optimization

        * verbose: bool (default=True)
            Whether or not producing detailed logging about the modeling
    """

    # Choosing an optimizer
    W = model.parameters()
    if optimizer_str.lower() == "adadelta":
        optimizer = torch.optim.Adadelta(W, lr=lr)

    elif optimizer_str.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(W, lr=lr)

    elif optimizer_str.lower() == "adam":
        optimizer = torch.optim.Adam(W, lr=lr)

    elif optimizer_str.lower() == "adamax":
        optimizer = torch.optim.Adamax(W, lr=lr)

    elif optimizer_str.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(W, lr=lr)

    elif optimizer_str.lower() == "sparseadam":
        optimizer = torch.optim.SparseAdam(W, lr=lr)

    elif optimizer_str.lower() == "sgd":
        optimizer = torch.optim.SGD(W, lr=lr)

    elif optimizer_str.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS(W, lr=lr)

    elif optimizer_str.lower() == "rprop":
        optimizer = torch.optim.Rprop(W, lr=lr)

    else:
        error = "{} optimizer isn't implemented".format(optimizer_str)
        raise NotImplementedError(error)

    # Initializing the Progress Bar
    loss_values = []
    if verbose:
        widgets = ["% Completion: ", progressbar.Percentage(), progressbar.Bar("*"), ""]
        bar = progressbar.ProgressBar(maxval=nb_epochs, widgets=widgets)
        bar.start()

    # Updating the weights at each training epoch
    temp_model = None
    metrics_names = ["brs", "ibs", "auc", "ctd"]

    metrics = {"c_index_valid": [], "c_index_train": []}
    for name in metrics_names:
        metrics[name] = []
    for epoch in range(nb_epochs):

        # Backward pass and optimization
        def closure():
            optimizer.zero_grad()
            loss = loss_function(model, X, T, E, **kargs)
            loss.backward()
            return loss

        if "lbfgs" in optimizer_str.lower():
            optimizer.step(closure)
        else:
            optimizer.step()
        loss = closure()
        loss_value = loss.item()

        # Printing error message if the gradient didn't explode
        if np.isnan(loss_value) or np.isinf(loss_value):
            error = "The gradient exploded... "
            error += "You should reduce the learning"
            error += "rate (lr) of your optimizer"
            if verbose:
                widgets[-1] = error
            else:
                print(error)
            break

        # Otherwise, printing value of loss function
        else:
            temp_model = copy.deepcopy(model)
            loss_values.append(loss_value)
            if verbose:
                widgets[-1] = "Loss: {:6.2f}".format(loss_value)

            # def other_metrics(model_wrapper, model, E_train, T_train, X_valid, E_valid, T_valid, eval_times):
            #
            # if epoch%5==0:
            c_index_valid = concordance_index_mine(
                model_wrapper,
                temp_model,
                X=np.array(X_valid),
                T=np.array(T_valid).flatten(),
                E=np.array(E_valid).flatten(),
            )
            c_index_train = concordance_index_mine(
                model_wrapper,
                temp_model,
                X=np.array(X),
                T=np.array(T).flatten(),
                E=np.array(E).flatten(),
            )
            score = other_metrics(
                model_wrapper,
                temp_model,
                E_train=np.array(E),
                T_train=np.array(T),
                X_valid=np.array(X_valid),
                T_valid=np.array(T_valid).flatten(),
                E_valid=np.array(E_valid).flatten(),
                eval_times=eval_times,
                X_original=X_original,
            )
            metrics["c_index_valid"].append(c_index_valid)
            metrics["c_index_train"].append(c_index_train)
            for key in score.keys():
                metrics[key].append(score[key])

        # Updating the progressbar
        if verbose:
            bar.update(epoch + 1)

    model_wrapper.metrics = metrics
    # Terminating the progressbar
    if verbose:
        bar.finish()

    # Finilazing the model
    if temp_model is not None:
        temp_model = temp_model.eval()
        model = copy.deepcopy(temp_model)
    else:
        raise ValueError(error)

    return model, loss_values
