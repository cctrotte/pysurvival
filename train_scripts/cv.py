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




if __name__ == "__main__":
    save_path = "/cluster/work/medinfmk/STCS_swiss_transplant/to_save_ct/"
    # with open(save_path + "rec_feat_select_results_17_01.pickle", "rb") as f:
    #     rec_feat_select = pickle.load(f)
    with open(save_path + "rec_feat_select_results_29_01_abmr.pickle", "rb") as f:
        rec_feat_select = pickle.load(f)
    set_seeds()
    df_s_small_enc, df_s_small, df_t = create_dfs()
    DF0 = T_E_outcomes(df_s_small)
    t_col = "T_ABMR"
    e_col = "E_ABMR"
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

    param_grids = {
        "surv_machine": {
            "k": [3, 4, 6, 10],
            "distribution": ["LogNormal", "Weibull"],
            "learning_rate": [1e-4, 1e-3],
            "layers": [[], [100], [100, 100], [100, 100, 100]],
        },
        "CondRF": {"num_trees": [20, 50, 100, 200], "max_depth": [2, 5, 15, 30, 50]},
    }
    # param_grids = {
    #     "surv_machine": {
    #         "k": [3, 4,],
    #         "distribution": ["LogNormal"],
    #         "learning_rate": [1e-4],
    #         "layers": [[100]],
    #     },
    #     "CondRF": {"num_trees": [20, 50], "max_depth": [2]},
    # }

    result_dfs = []
    horizons = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99, 1]
    tmp = X[["T", "E"]].reset_index(drop=True)
    #eval_times = np.quantile(tmp["T"][tmp.E == 1], horizons).tolist()
    eval_times = [0.02917808219178082, 0.25931506849315067, 0.6552054794520548, 1.5, 2.636849315068494, 4.069452054794517, 5.115260273972603, 6.753890410958904, 8.585342465753424, 10]
    for model_name, model in zip(
        ["surv_machine", "CondRF"],
        [
            DeepSurvivalMachines(),
            ConditionalSurvivalForestModel(),
        ],
    ):
        print(model_name)
        vars_ = rec_feat_select[model_name]["features"]

        fold_dict, _ = get_folds(split_dict, vars_)
        result_dfs.append(
            fit_and_tune_bis(
                model,
                param_grids[model_name],
                fold_dict,
                eval_times,
                model_name,
                seed=0,
            )[0]
        )

    with open(save_path + "test.pickle", "wb") as f:
        pickle.dump(result_dfs, f)