import random
import numpy as np
import pandas as pd
import torch
import copy
from itertools import product
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pysurvival_mine.utils.metrics import concordance_index
from pysurvival_mine.models.survival_forest import (
    ConditionalSurvivalForestModel,
    ExtraSurvivalTreesModel,
    RandomSurvivalForestModel,
)
from AutonSurvival.auton_survival.metrics import survival_regression_metric
from AutonSurvival.auton_survival.models.dsm import DeepSurvivalMachines


def make_train_test_split(X, T, E, ID, index_train, index_valid, index_test):

    # Creating the X, T and E inputs
    X_train, X_valid, X_test = (
        X.iloc[index_train],
        X.iloc[index_valid],
        X.iloc[index_test],
    )
    T_train, T_valid, T_test = T[index_train], T[index_valid], T[index_test]
    E_train, E_valid, E_test = E[index_train], E[index_valid], E[index_test]
    ID_train, ID_valid, ID_test = ID[index_train], ID[index_valid], ID[index_test]

    return {
        "X_train": X_train,
        "T_train": T_train,
        "E_train": E_train,
        "ID_train": ID_train,
        "index_train": index_train,
        "X_valid": X_valid,
        "T_valid": T_valid,
        "E_valid": E_valid,
        "ID_valid": ID_valid,
        "index_valid": index_valid,
        "X_test": X_test,
        "T_test": T_test,
        "E_test": E_test,
        "ID_test": ID_test,
        "index_test": index_test,
    }


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


def get_features(split_dict, vars_, impute=True):
    X_train, X_valid, X_test = (
        split_dict["X_train"][vars_],
        split_dict["X_valid"][vars_],
        split_dict["X_test"][vars_],
    )
    T_train, T_valid, T_test = (
        split_dict["T_train"].reshape(-1, 1),
        split_dict["T_valid"].reshape(-1, 1),
        split_dict["T_test"].reshape(-1, 1),
    )
    E_train, E_valid, E_test = (
        split_dict["E_train"].reshape(-1, 1),
        split_dict["E_valid"].reshape(-1, 1),
        split_dict["E_test"].reshape(-1, 1),
    )
    ID_train, ID_valid, ID_test = (
        split_dict["ID_train"].to_numpy().reshape(-1, 1),
        split_dict["ID_valid"].to_numpy().reshape(-1, 1),
        split_dict["ID_test"].to_numpy().reshape(-1, 1),
    )

    if impute:
        means = X_train.mean()
        X_train = X_train.fillna(means)
        X_valid = X_valid.fillna(means)
        X_test = X_test.fillna(means)
        T_train, T_test, E_train, E_test = (
            T_train.flatten(),
            T_test.flatten(),
            E_train.flatten(),
            E_test.flatten(),
        )
    return (
        X_train,
        X_valid,
        X_test,
        T_train,
        T_valid,
        T_test,
        E_train,
        E_valid,
        E_test,
        ID_train,
        ID_valid,
        ID_test,
    )


def get_folds(split_dict, vars_, folds=None, impute=True):
    fold_dict = {key: {} for key in range(5)}
    if folds is None:
        X_train = pd.concat((split_dict["X_train"], split_dict["X_valid"]))
        idx = X_train["soascaseid"].values
        T_train = X_train["T"]
        E_train = X_train["E"]
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for i, (train_index, valid_index) in enumerate(kf.split(idx)):
        fold_dict[i]["X_train"], fold_dict[i]["X_valid"] = (
            X_train[vars_].iloc[train_index],
            X_train[vars_].iloc[valid_index],
        )
        fold_dict[i]["T_train"], fold_dict[i]["T_valid"] = (
            T_train.iloc[train_index].values,
            T_train.iloc[valid_index].values,
        )
        fold_dict[i]["E_train"], fold_dict[i]["E_valid"] = (
            E_train.iloc[train_index].values,
            E_train.iloc[valid_index].values,
        )
        fold_dict[i]["ID_train"], fold_dict[i]["ID_valid"] = (
            idx[train_index],
            idx[valid_index],
        )
        scaler = StandardScaler()
        fold_dict[i]["X_train_scaled"] = scaler.fit_transform(fold_dict[i]["X_train"])
        fold_dict[i]["X_valid_scaled"] = scaler.transform(fold_dict[i]["X_valid"])

        if impute:
            means = pd.DataFrame(fold_dict[i]["X_train_scaled"]).mean(axis=0)
            fold_dict[i]["X_train_scaled"] = np.array(
                pd.DataFrame(fold_dict[i]["X_train_scaled"]).fillna(means).values
            )
            fold_dict[i]["X_valid_scaled"] = np.array(
                pd.DataFrame(fold_dict[i]["X_valid_scaled"]).fillna(means).values
            )

    return fold_dict, kf


def fit_and_tune(model, param_grid, fold_dict, eval_times, model_name="coxph"):
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    results_df = pd.DataFrame(
        columns=param_names + ["fold", "c_index_train", "c_index_valid"]
    )
    metrics = ["brs", "ibs", "auc", "ctd"]
    models = []
    # Grid Search
    for fold in fold_dict.keys():
        X_train = fold_dict[fold]["X_train_scaled"]
        X_valid = fold_dict[fold]["X_valid_scaled"]
        T_train = fold_dict[fold]["T_train"]
        E_train = fold_dict[fold]["E_train"]
        T_valid = fold_dict[fold]["T_valid"]
        E_valid = fold_dict[fold]["E_valid"]
        et_train = pd.DataFrame(
            np.array(
                [(E_train[i], T_train[i]) for i in range(len(E_train))],
                dtype=[("event", bool), ("time", float)],
            )
        )
        et_val = pd.DataFrame(
            np.array(
                [(E_valid[i], T_valid[i]) for i in range(len(E_valid))],
                dtype=[("event", bool), ("time", float)],
            )
        )
        scores = []
        for i, params in enumerate(param_combinations):
            score = {metric: np.nan for metric in metrics}

            # Set the current hyperparameters
            print(
                f"{model_name} fold {fold} param combi {i} out of {len(param_combinations)}"
            )
            current_params = dict(zip(param_grid.keys(), params))
            if model_name in ["CondRF"]:
                model = ConditionalSurvivalForestModel(
                    num_trees=current_params["num_trees"]
                )
                fit_params = {
                    key: elem
                    for key, elem in current_params.items()
                    if key != "num_trees"
                }
            elif model_name in ["linear_nn", "coxph_nn"]:
                model = model
                model.structure = current_params["structure"]
                fit_params = {
                    key: elem
                    for key, elem in current_params.items()
                    if key != "structure"
                }
            else:
                fit_params = current_params
            if model_name in ["linear", "linear_nn", "coxph_nn"]:
                try:
                    model.fit(
                        X_train,
                        T_train,
                        E_train,
                        X_valid,
                        T_valid,
                        E_valid,
                        **fit_params,
                    )
                except Exception as e:
                    print(f"Error fitting model: {e}")
            else:
                try:
                    model.fit(X_train, T_train, E_train, **fit_params)
                except Exception as e:
                    print(f"Error fitting model: {e}")
            # Compute the C-index for the current hyperparameters
            try:
                c_index_train = concordance_index(model, X_train, T_train, E_train)
                c_index_valid = concordance_index(model, X_valid, T_valid, E_valid)
                for metric in metrics:
                    out_survival = model.predict_survival(
                        X_valid,
                    )
                    indices = []
                    for t in eval_times:
                        min_index = [
                            abs(a_j_1 - t) for (a_j_1, a_j) in model.time_buckets
                        ]
                        indices.append(np.argmin(min_index))
                    out_survival = out_survival[:, indices]
                    score[metric] = np.mean(
                        survival_regression_metric(
                            metric=metric,
                            outcomes_train=et_train,
                            outcomes=et_val,
                            predictions=out_survival,
                            times=eval_times,
                        )
                    )
            except Exception as e:
                print(f"Error computing C-index: {e}")
                c_index_train = c_index_valid = float("nan")
            if model_name in ["linear", "linear_nn", "coxph_nn"]:
                try:
                    best_epoch = np.argmax(model.metrics["c_index_valid"])
                    c_index_valid = np.max(model.metrics["c_index_valid"])
                    c_index_train = model.metrics["c_index_train"][best_epoch]
                except Exception as e:
                    print(f"Error computing C-index: {e}")
                    best_epoch = None
                    c_index_valid = np.nan
                    c_index_train = np.nan

            # Add the results to the DataFrame
            else:
                best_epoch = None
            results_df = results_df.append(
                {
                    **current_params,
                    "fold": fold,
                    "c_index_train": c_index_train,
                    "c_index_valid": c_index_valid,
                    "best_epoch": best_epoch,
                    "name": model_name,
                    **score,
                },
                ignore_index=True,
            )

    return results_df, model


def fit_and_tune_bis(
    model, param_grid, fold_dict, eval_times, model_name="coxph", seed=0
):
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    results_df = pd.DataFrame(
        columns=param_names + ["fold", "c_index_train", "c_index_valid"]
    )
    metrics = ["brs", "ibs", "auc", "ctd"]
    models = []
    # Grid Search
    for fold in fold_dict.keys():
        X_train = fold_dict[fold]["X_train_scaled"]
        X_valid = fold_dict[fold]["X_valid_scaled"]
        T_train = fold_dict[fold]["T_train"]
        E_train = fold_dict[fold]["E_train"]
        T_valid = fold_dict[fold]["T_valid"]
        E_valid = fold_dict[fold]["E_valid"]
        et_train = pd.DataFrame(
            np.array(
                [(E_train[i], T_train[i]) for i in range(len(E_train))],
                dtype=[("event", bool), ("time", float)],
            )
        )
        et_val = pd.DataFrame(
            np.array(
                [(E_valid[i], T_valid[i]) for i in range(len(E_valid))],
                dtype=[("event", bool), ("time", float)],
            )
        )
        scores = []
        for i, params in enumerate(param_combinations):
            score = {metric: np.nan for metric in metrics}

            # Set the current hyperparameters
            print(
                f"{model_name} fold {fold} param combi {i} out of {len(param_combinations)}"
            )
            current_params = dict(zip(param_grid.keys(), params))
            if model_name in ["CondRF"]:
                model = ConditionalSurvivalForestModel(
                    num_trees=current_params["num_trees"]
                )
                fit_params = {
                    key: elem
                    for key, elem in current_params.items()
                    if key != "num_trees"
                }
            elif model_name in ["linear_nn", "coxph_nn"]:
                model = model
                model.structure = current_params["structure"]
                fit_params = {
                    key: elem
                    for key, elem in current_params.items()
                    if key != "structure"
                }
            elif model_name in ["surv_machine"]:
                model = DeepSurvivalMachines(
                    k=current_params["k"],
                    distribution=current_params["distribution"],
                    layers=current_params["layers"],
                    random_seed=seed,
                )
            else:
                fit_params = current_params
            if model_name in ["linear", "linear_nn", "coxph_nn"]:
                try:
                    model.fit(
                        X_train,
                        T_train,
                        E_train,
                        X_valid,
                        T_valid,
                        E_valid,
                        **fit_params,
                    )
                except Exception as e:
                    print(f"Error fitting model: {e}")
            elif model_name in ["surv_machine"]:
                try:
                    model.fit(
                        X_train,
                        T_train,
                        E_train,
                        val_data=(X_valid, T_valid, E_valid),
                        pat_thresh=3,
                        metric_name="ctd",
                        iters=2000,
                        learning_rate=current_params["learning_rate"],
                    )
                except Exception as e:
                    print(f"Error fitting model: {e}")
            else:
                try:
                    model.fit(X_train, T_train, E_train, **fit_params)
                except Exception as e:
                    print(f"Error fitting model: {e}")
            # Compute the C-index for the current hyperparameters
            try:
                if model_name in ["surv_machine"]:
                    out_survival = model.predict_survival(X_valid, eval_times)
                else:
                    out_survival = model.predict_survival(
                        X_valid,
                    )
                    indices = []
                    for t in eval_times:
                        min_index = [
                            abs(a_j_1 - t) for (a_j_1, a_j) in model.time_buckets
                        ]
                        indices.append(np.argmin(min_index))
                    out_survival = out_survival[:, indices]
                for metric in metrics:

                    score[metric] = np.mean(
                        survival_regression_metric(
                            metric=metric,
                            outcomes_train=et_train,
                            outcomes=et_val,
                            predictions=out_survival,
                            times=eval_times,
                            random_seed=seed,
                        )
                    )
                score["overall_ctd"] = np.max(model.metrics["ctd"])

            except Exception as e:
                print(f"Error computing C-index: {e}")

            results_df = results_df.append(
                {**current_params, "fold": fold, "name": model_name, **score},
                ignore_index=True,
            )
            models.append(model)

    return results_df, models
