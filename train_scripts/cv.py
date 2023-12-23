import sys

sys.path.append("/cluster/work/medinfmk/STCS_swiss_transplant/code_ct/pysurvival_mine/")
sys.path.append(
    "/cluster/work/medinfmk/STCS_swiss_transplant/code_ct/pysurvival_mine/AutonSurvival"
)
#### 1 - Importing packages
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pysurvival_mine.models.simulations import SimulationModel
from pysurvival_mine.models.coxph_mine import MineNonLinearCoxPHModel
from pysurvival_mine.utils.metrics import concordance_index
from pysurvival_mine.utils.metrics import concordance_index_mine
from sklearn.preprocessing import StandardScaler
from pysurvival_mine.utils.display import integrated_brier_score
from pysurvival_mine.utils.display import compare_to_actual
from pysurvival_mine.models.semi_parametric import CoxPHModel
from pysurvival_mine.models.coxph_mine import MineNonLinearCoxPHModel
from pysurvival_mine.models.survival_forest import (
    ConditionalSurvivalForestModel,
    ExtraSurvivalTreesModel,
    RandomSurvivalForestModel,
)
from pysurvival_mine.models.multi_task_mine import (
    LinearMultiTaskModelMine,
    NeuralMultiTaskModelMine,
)


from utils_mine.preprocessing import *
from utils_mine.ml_prepro import *

# variables available at transplant
baseline_vars = [
    "DQB1_MM",
    "dbp",
    "donage",
    "A_MM",
    "C_MM",
    "DRB345_MM",
    "MFI",
    "height",
    "HLA_MM_Total",
    "weight",
    "ColdIschemia",
    "warmisch",
    "DRB1_MM",
    "HLA_MM_Class II",
    "DPB1_MM",
    "HLA_MM_Class I",
    "B_MM",
    "DSAcount",
    "sbp",
    "DSAcumulativeMFI",
    "age_at_transp",
    "transp_num",
    "sex_Male",
    "dontype_Living unrelated",
    "dontype_Living related",
    "dontype_Brain dead",
    "dontype_NHBD",
    "dontype_Unknown",
    "HLARiskGroup_No HLA-DSA",
    "HLARiskGroup_HLA-DSA",
    "HLARiskGroup_Other risk",
    "LD_DD_DD",
    "HLA_antibodies_yes",
    "simult_Double Tpx",
    "donsex_Female",
    "C/R_NoDSA",
    "C/R_ Current DSA",
    "C/R_ Remote DSA",
    "DSAclass_NoDSA",
    "DSAclass_II",
    "DSAclass_I",
    "DSAclass_I+II",
]


def compute_coxph(split_dict, vars_, drop=True):
    X_train, X_valid = split_dict["X_train"][vars_], split_dict["X_valid"][vars_]
    T_train, T_valid = split_dict["T_train"].reshape(-1, 1), split_dict[
        "T_valid"
    ].reshape(-1, 1)
    E_train, E_valid = split_dict["E_train"].reshape(-1, 1), split_dict[
        "E_valid"
    ].reshape(-1, 1)
    ID_train, ID_valid = split_dict["ID_train"].to_numpy().reshape(-1, 1), split_dict[
        "ID_valid"
    ].to_numpy().reshape(-1, 1)
    miss_ = np.round(
        np.isnan(X_train.to_numpy()).any(axis=1).sum() / len(X_train) * 100, 2
    )
    print(f"Dropping {miss_} % of the values because missing")

    # imputation or drop
    if drop:
        not_null_train = ~np.isnan(X_train.to_numpy()).any(axis=1)
        not_null_valid = ~np.isnan(X_valid.to_numpy()).any(axis=1)
        X_train, T_train, E_train, ID_train = (
            X_train[not_null_train],
            T_train[not_null_train].flatten(),
            E_train[not_null_train].flatten(),
            ID_train[not_null_train],
        )
        X_valid, T_valid, E_valid, ID_valid = (
            X_valid[not_null_valid],
            T_valid[not_null_valid].flatten(),
            E_valid[not_null_valid].flatten(),
            ID_valid[not_null_valid],
        )
    # impute with mean or majority class
    else:
        means = X_train.mean()
        X_train = X_train.fillna(means)
        X_valid = X_valid.fillna(means)
        T_train, T_valid, E_train, E_valid = (
            T_train.flatten(),
            T_valid.flatten(),
            E_train.flatten(),
            E_valid.flatten(),
        )
    coxph = CoxPHModel()
    coxph.fit(X_train, T_train, E_train, lr=0.5, l2_reg=1e-2, init_method="zeros")

    #### 5 - Cross Validation / Model Performances
    c_index = concordance_index(coxph, X_valid, T_valid, E_valid)
    print("C-index: {:.2f}".format(c_index))

    ibs = integrated_brier_score(
        coxph, X_valid, T_valid, E_valid, t_max=10, figure_size=(20, 6.5)
    )
    print("IBS: {:.2f}".format(ibs))
    summary = coxph.summary
    summary["c_index"] = c_index
    summary["ibs"] = ibs
    if drop:
        summary["prop_missing"] = miss_

    summary.p_values = summary.p_values.astype(float)
    return summary


if __name__ == "__main__":
    save_path = "/cluster/work/medinfmk/STCS_swiss_transplant/to_save_ct/"
    set_seeds()
    df_s_small_enc, df_s_small, df_t = create_dfs()
    DF0 = T_E_outcomes(df_s_small)
    t_col = "T_rejection"
    e_col = "E_rejection"
    T = DF0[t_col].to_numpy() + 0.01
    E = DF0[e_col].to_numpy()
    Tnotisnan = ~np.isnan(T)
    T = T[Tnotisnan]
    E = E[Tnotisnan]
    X = df_s_small_enc[Tnotisnan]
    X["T"] = T
    X["E"] = E
    ID = X["soascaseid"]
    index_train, index_test = train_test_split(range(len(X)), test_size=0.1)
    index_train, index_valid = train_test_split(index_train, test_size=0.1)
    split_dict = make_train_test_split(
        X, T, E, ID, index_train, index_valid, index_test
    )

    S_baseline = pd.DataFrame()
    for v in baseline_vars:
        print(v)
        s = compute_coxph(split_dict, [v])
        S_baseline = pd.concat([S_baseline, s])
    features = list(S_baseline[S_baseline.p_values < 0.1].variables)
    print(f"keeping {len(features)} out of {len(baseline_vars)} features")
    print(features)
    # fit baseline models: multivariate on all or restricted subset
    # param_grids = {"coxph": {"lr": [0.001, 0.01], "l2_reg": [1e-2,]},
    #                "CondRF": {"num_trees": [10,50,100], "max_depth": [5,15, 30]},
    #                "coxph_nn": {"lr": [0.001,], "num_epochs": [500], "l2_reg": [1e-2,], "dropout": [ 0.5], "batch_normalization": [True, False], "bn_and_dropout": [True, False]}}
    param_grids = {
        "linear_nn": {
            "structure": [
                [{"activation": "ReLU", "num_units": 150}],
                [
                    {"activation": "ReLU", "num_units": 150},
                    {"activation": "ReLU", "num_units": 150},
                ],
                [
                    {"activation": "ReLU", "num_units": 150},
                    {"activation": "ReLU", "num_units": 150},
                    {"activation": "ReLU", "num_units": 150},
                ],
            ],
            "lr": [0.001],
            "l2_reg": [1e-2, 1e-3, 1e-4],
            "num_epochs": [700],
            "dropout": [0.2, 0.5, 0.7],
            "batch_normalization": [True, False],
            "bn_and_dropout": [True, False],
        },
        "surv_machine": {
            "k": [3, 4, 6, 10],
            "distribution": ["LogNormal", "Weibull"],
            "learning_rate": [1e-4, 1e-3],
            "layers": [[], [100], [100, 100], [100, 100, 100]],
        },
        "linear": {"lr": [0.001], "l2_reg": [1e-2, 1e-3]},
        "coxph": {"lr": [0.001], "l2_reg": [1e-2, 1e-3]},
        "CondRF": {"num_trees": [20, 50, 100], "max_depth": [2, 5, 15, 30]},
        "coxph_nn": {
            "structure": [
                [{"activation": "ReLU", "num_units": 150}],
                [
                    {"activation": "ReLU", "num_units": 150},
                    {"activation": "ReLU", "num_units": 150},
                ],
                [
                    {"activation": "ReLU", "num_units": 150},
                    {"activation": "ReLU", "num_units": 150},
                    {"activation": "ReLU", "num_units": 150},
                ],
            ],
            "lr": [0.001],
            "num_epochs": [700],
            "l2_reg": [1e-2, 1e-3, 1e-4],
            "dropout": [0.2, 0.5, 0.7],
            "batch_normalization": [True, False],
            "bn_and_dropout": [True, False],
        },
    }
    # param_grids = {"linear_nn": {"structure": [ [{'activation': 'ReLU', 'num_units': 150}]], "lr": [0.001], "l2_reg": [1e-3], "num_epochs": [3], "dropout": [ 0.2], "batch_normalization": [True], "bn_and_dropout": [True]},
    #             "surv_machine": {'k' : [3],
    #           'distribution' : ['LogNormal'],
    #           'learning_rate' : [ 1e-4],
    #           'layers' : [  [100, 100] ]
    #          },
    #             "linear": {"lr": [0.001], "l2_reg": [1e-2]},
    #             "coxph": {"lr": [0.001], "l2_reg": [1e-2]},
    #             "CondRF": {"num_trees": [20], "max_depth": [2]},
    #             "coxph_nn": {"structure":[ [{'activation': 'ReLU', 'num_units': 150}]], "num_epochs": [3], "l2_reg": [1e-2], "dropout": [ 0.2], "batch_normalization": [False], "bn_and_dropout": [True]}}
    result_dfs = {"baseline": [], "features": []}
    horizons = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99, 1]
    tmp = X[["T", "E"]].reset_index(drop=True)
    eval_times = np.quantile(tmp["T"][tmp.E == 1], horizons).tolist()

    for model_name, model in zip(
        ["surv_machine", "linear_nn", "linear", "coxph_nn", "coxph", "CondRF"],
        [
            DeepSurvivalMachines(),
            NeuralMultiTaskModelMine(
                bins=100,
                auto_scaler=False,
                structure=param_grids["linear_nn"]["structure"][0],
            ),
            LinearMultiTaskModelMine(bins=100, auto_scaler=False),
            MineNonLinearCoxPHModel(auto_scaler=False),
            CoxPHModel(),
            ConditionalSurvivalForestModel(),
        ],
    ):
        print(model_name)
        for i, vars_ in zip(["baseline", "features"], [baseline_vars, features]):

            fold_dict, _ = get_folds(split_dict, vars_)
            result_dfs[i].append(
                fit_and_tune_bis(
                    model,
                    param_grids[model_name],
                    fold_dict,
                    eval_times,
                    model_name,
                    seed=0,
                )[0]
            )

        with open(save_path + "cv_results_23_12.pickle", "wb") as f:
            pickle.dump(result_dfs, f)
