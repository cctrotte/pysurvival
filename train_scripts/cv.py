import sys
sys.path.append("/cluster/work/medinfmk/STCS_swiss_transplant/code_ct/pysurvival_mine/")

 #### 1 - Importing packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pysurvival_mine.models.simulations import SimulationModel
from pysurvival_mine.models.coxph_mine import MineNonLinearCoxPHModel
from pysurvival_mine.utils.metrics import concordance_index
from pysurvival_mine.utils.metrics import concordance_index_mine
from sklearn.preprocessing import StandardScaler
from pysurvival_mine.utils.display import integrated_brier_score
from pysurvival_mine.utils.display import compare_to_actual

from utils_mine.preprocessing import *
from utils_mine.feature_selection import *
from utils_mine.ml_prepro import *

baseline_vars = ['DQB1_MM', 'dbp','donage', 'A_MM', 'C_MM',
       'DRB345_MM', 'MFI', 'height',
       'HLA_MM_Total', 'weight', 'ColdIschemia',
       'warmisch', 'DRB1_MM', 'HLA_MM_Class II',
       'DPB1_MM', 
       'HLA_MM_Class I', 'B_MM', 'DSAcount', 'sbp', 
       'DSAcumulativeMFI', 'age_at_transp', 'transp_num',
        'sex_Male', 'dontype_Living unrelated',
       'dontype_Living related', 'dontype_Brain dead', 'dontype_NHBD',
       'dontype_Unknown', 'HLARiskGroup_No HLA-DSA', 'HLARiskGroup_HLA-DSA',
       'HLARiskGroup_Other risk', 'LD_DD_DD', 'HLA_antibodies_yes',
       'simult_Double Tpx', 
       'donsex_Female', 'C/R_NoDSA',
       'C/R_ Current DSA', 'C/R_ Remote DSA',
       'DSAclass_NoDSA', 'DSAclass_II', 'DSAclass_I', 'DSAclass_I+II']

if __name__ == "__main__":
    set_seeds()
    df_s_small_enc, df_s_small, df_t = create_dfs()
    DF0 = T_E_outcomes(df_s_small)
    t_col = 'T_rejection'
    e_col = 'E_rejection'
    T = DF0[t_col].to_numpy()
    E = DF0[e_col].to_numpy()
    Tnotisnan = ~np.isnan(T)
    T = T[Tnotisnan]
    E = E[Tnotisnan]
    X = df_s_small_enc[Tnotisnan]
    X["T"] = T
    X["E"] = E
    ID = X['soascaseid']
    index_train, index_test = train_test_split( range(len(X)), test_size = 0.1)
    index_train, index_valid = train_test_split(index_train, test_size = 0.1)
    split_dict = make_train_test_split(X, T, E, ID, index_train, index_valid, index_test)
    S_baseline = pd.DataFrame()
    for v in baseline_vars:
        print(v)
        s = compute_coxph(split_dict, [v])
        S_baseline = pd.concat([S_baseline,s])
    features = list(S_baseline[S_baseline.p_values < 0.1].variables)
    print(f"keeping {len(features)} out of {len(baseline_vars)} features")
    print(features)


    
