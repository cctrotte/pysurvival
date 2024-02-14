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
import numpy as np

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
# baseline_vars = [
#     "DQB1_MM",
#     "dbp",
# ]

param_grids_all = {"rejection": {
        
        "surv_machine": {
            "k": [6],
            "distribution": ["Weibull"],
            "learning_rate": [1e-3], 
            "layers":[ [100]]},
        "CondRF": {"num_trees": [100], "max_depth": [30]},
}, "tcmr":{
        
        "surv_machine": {
            "k": [10],
            "distribution": ["LogNormal"],
            "learning_rate": [1e-3], 
            "layers":[ [100, 100]]},
        "CondRF": {"num_trees": [20], "max_depth": [5]},
}, "failure":{
        
        "surv_machine": {
            "k": [3],
            "distribution": ["LogNormal"],
            "learning_rate": [1e-4], 
            "layers":[ [100, 100, 100]]},
        "CondRF": {"num_trees": [20], "max_depth": [2]},
},
"abmr":{
        
        "surv_machine": {
            "k": [10],
            "distribution": ["Weibull"],
            "learning_rate": [1e-3], 
            "layers":[ [100, 100, 100]]},
        "CondRF": {"num_trees": [100], "max_depth": [5]},
 }}
if __name__ == "__main__":
    save_path = "/cluster/work/medinfmk/STCS_swiss_transplant/to_save_ct/"
    set_seeds()
    df_s_small_enc, df_s_small, df_t = create_dfs()
    DF0 = T_E_outcomes(df_s_small)
    # t_col = "T_rejection"
    # e_col = "E_rejection"
    t_col = "T_TCMR"
    e_col = "E_TCMR"
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

    # fit baseline models: multivariate on all or restricted subset
    # param_grids = {"coxph": {"lr": [0.001, 0.01], "l2_reg": [1e-2,]},
    #                "CondRF": {"num_trees": [10,50,100], "max_depth": [5,15, 30]},
    #                "coxph_nn": {"lr": [0.001,], "num_epochs": [500], "l2_reg": [1e-2,], "dropout": [ 0.5], "batch_normalization": [True, False], "bn_and_dropout": [True, False]}}
#     param_grids = {
        
#         "surv_machine": {
#             "k": [6],
#             "distribution": ["Weibull"],
#             "learning_rate": [1e-3], 
#             "layers":[ [100]]},
#         "CondRF": {"num_trees": [100], "max_depth": [30]},
# }
    # param_grids = {
    #     "surv_machine": {
    #         "k": [3],
    #         "distribution": ["LogNormal"],
    #         "learning_rate": [1e-3], 
    #         "layers": [[100, 100]]}}
    param_grids = param_grids_all["tcmr"]
    
    result_dfs = {key : {"features": [], "c_index": []} for key in param_grids.keys()}
    # horizons = [0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99, 1]
    # tmp = X[["T", "E"]].reset_index(drop=True)
    # eval_times = np.quantile(tmp["T"][tmp.E == 1], horizons).tolist()
    eval_times = [0.02917808219178082, 0.25931506849315067, 0.6552054794520548, 1.5, 2.636849315068494, 4.069452054794517, 5.115260273972603, 6.753890410958904, 8.585342465753424, 10] 



    for model_name, model in zip(
        ["surv_machine", "CondRF"],
        [
            DeepSurvivalMachines(),
            ConditionalSurvivalForestModel(),
        ],
    ):
        print(model_name)
        best_c_index = 0
        remaining_vars = baseline_vars.copy()
        selected_vars = []
        stop = False
        while not stop and len(remaining_vars) > 0:
            best_var = None
            for var in remaining_vars:
                current_vars = selected_vars + [var]
                fold_dict, _ = get_folds(split_dict, current_vars)
                res = fit_and_tune_bis(
                        model,
                        param_grids[model_name],
                        fold_dict,
                        eval_times,
                        model_name,
                        seed=0,
                    )[0]
                if model_name == "surv_machine":
                    score = res["overall_ctd"].mean()
                else:
                    score = res["ctd"].mean()
                if score > best_c_index:
                    best_c_index = score
                    best_var = var
            if best_var is None:
                stop = True
            else:   
                print(selected_vars)          
                print(best_c_index)  
                selected_vars.append(best_var)
                remaining_vars.remove(best_var)
                result_dfs[model_name]["features"].append(best_var)
                result_dfs[model_name]["c_index"].append(best_c_index) 
            

    with open(save_path + "rec_feat_select_results_29_01_tcmr.pickle", "wb") as f:
        pickle.dump(result_dfs, f)
