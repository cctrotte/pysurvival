import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    OneHotEncoder,
)

features_to_keep = [
    "soascaseid",
    "patid",
    "sex",
    "weight",
    "height",
    "birthday",
    "sbp",
    "dbp",
    "donsex",
    "dontype",
    "donage",
    "LD_DD",
    "tpxdate",
    "DaysPostTRPLastVisit",
    "ColdIschemia",
    "warmisch",
    "A_MM",
    "B_MM",
    "C_MM",
    "DRB1_MM",
    "DRB345_MM",
    "DQB1_MM",
    "DPB1_MM",
    "HLA_MM_Class I",
    "HLA_MM_Class II",
    "HLA_MM_Total",
    "HLA_antibodies",
    "HLARiskGroup",
    "MFI",
    "C/R",
    "DSAcount",
    "DSAclass",
    "DSAcumulativeMFI",
    "simult",
]
outcomes_to_keep = [
    "soascaseid",
    "patid",
    "tpxdate",
    "dgf",
    "txFailureDate",
    "txF_DaysPost_tpx",
    "deathDate",
    "Death_DaysPost_tpx",
    "death",
    "AnyRejection",
    "FirstRejectionDate",
    "Rj_DaysPost_tpx",
    "Acute TCMR",
    "Acute/active ABMR",
    "Acute_Mixed ",
    "Chronic active ABMR",
    "Chronic active TCMR",
    "Chronic_Mixed ",
]


def count_transplants(df):
    df.transp_num = [i for i in range(1, len(df) + 1)]
    return df


def imputation(df, column_name):
    tmp = df.copy()
    if column_name in [
        "DSAclass",
        "C/R",
    ]:
        tmp[column_name] = tmp[column_name].fillna("NoDSA")
    elif column_name in ["DSAcount"]:
        tmp[column_name] = tmp[column_name].fillna(0)
    return tmp


def create_dfs():
    folder = "/cluster/work/medinfmk/STCS_swiss_transplant/data/original_data/Yun_Near fianl set/"
    data_path1 = folder + "1-STCS Sum_Yun.xlsx"
    data_path2 = folder + "2-STCS Creatine eGFR.xlsx"
    data_path3 = folder + "kidney_index2.xlsx"
    data_path4 = folder + "Donor_index.xlsx"

    df1 = pd.read_excel(data_path1, sheet_name=None)["Basic"]
    df2 = pd.read_excel(data_path2, sheet_name=None)["Creatine"]
    df3 = pd.read_excel(data_path3, index_col=0, sheet_name=None)["Sheet1"]
    df4 = pd.read_excel(data_path4, sheet_name=None)["Tabelle1"]
    df_s = df1.copy()
    df_t = df2.copy()

    df_s = df_s.merge(df4[["soascaseid", "donage"]], on="soascaseid")
    df_t = df_t.merge(df3[["patid", "assdate", "num_days"]], on=["patid", "assdate"])
    df_t = df_t[df_t.patid.isin(df_s.patid)]

    for col in [
        "tpxdate",
        "birthday",
        "txFailureDate",
        "deathDate",
        "FirstRejectionDate",
    ]:
        df_s[col] = pd.to_datetime(df_s[col], dayfirst=True)
    df_s_small = df_s[list(set(features_to_keep + outcomes_to_keep))]
    df_s_small["age_at_transp"] = (
        df_s_small["tpxdate"] - df_s_small["birthday"]
    ).apply(lambda x: x.days)
    df_s_small = df_s_small.sort_values(by="tpxdate")
    # multiple transplant
    df_s_small["transp_num"] = np.nan
    df_s_small = df_s_small.groupby("patid").apply(count_transplants)
    for col in ["DSAclass", "C/R", "DSAcount"]:
        df_s_small = imputation(df_s_small, col)
        encoders = {}
    df_s_small_enc = df_s_small.copy()
    for i in range(len(df_s_small.dtypes)):
        if df_s_small.dtypes[i] == "object" and (
            df_s_small.dtypes.index[i] not in ["soascaseid"]
        ):
            name = df_s_small.dtypes.index[i]
            labels = df_s_small[name].unique()
            labels = [[elem for elem in labels if elem == elem]]
            print(labels)
            encoders[name] = OneHotEncoder(
                sparse_output=False,
                drop="if_binary",
                categories=labels,
                handle_unknown="ignore",
            )
            encoders[name].fit(df_s_small_enc[[name]])
            arr = encoders[name].transform(df_s_small_enc[[name]])
            arr[df_s_small_enc[name].isnull(), :] = np.nan
            print(arr.shape)
            if arr.shape[1] > 1:
                df_s_small_enc[[name + "_" + elem for elem in labels[0]]] = arr
            elif name in [
                "Chronic active TCMR",
                "Chronic active ABMR",
                "Acute_Mixed ",
                "Chronic_Mixed ",
                "Acute/active ABMR",
            ]:
                df_s_small_enc[name + "_" + labels[0][0]] = arr
            else:
                df_s_small_enc[name + "_" + labels[0][1]] = arr
            df_s_small_enc = df_s_small_enc.drop(columns=name)

        elif df_s_small.dtypes[i] == "bool":
            name = df_s_small.dtypes.index[i]
            df_s_small_enc[name] = df_s_small[name].astype(int)

    return df_s_small_enc, df_s_small, df_t


def T_E_outcomes(DF0, divide=365):
    """
    updates the DF0 with columns of T and E pairs of several outcomes
    """

    # all available time variables in the data
    days_post = [
        "Rj_DaysPost_tpx",
        "txF_DaysPost_tpx",
        "Death_DaysPost_tpx",
        "DaysPostTRPLastVisit",
    ]

    # possible outcomes
    outcomes = ["failure", "death", "rejection", "ABMR", "TCMR"]

    var_names = {
        "failure": "txF_DaysPost_tpx",
        "death": "Death_DaysPost_tpx",
        "rejection": "Rj_DaysPost_tpx",
        "ABMR": "Rj_DaysPost_tpx",
        "TCMR": "Rj_DaysPost_tpx",
    }

    # time for censor (i.e. last evidence for patient)
    T_censor = np.max(DF0[days_post].fillna(0).to_numpy() / divide, 1)

    for j, out in enumerate(outcomes):

        # time for event
        T_event = DF0[var_names[out]].to_numpy() / divide

        # definition of event
        if out in ["failure", "death", "rejection"]:
            E = 1 - np.isnan(T_event)
        elif out == "ABMR":
            # both acute and chronic ABMR
            # DF0['Acute/active ABMR'].unique() = [nan, 'acute ABMR']
            # DF0['Chronic active ABMR'].unique() = [nan, 'Chronic ABMR']
            E = (~DF0["Acute/active ABMR"].isnull().to_numpy()) | (
                ~DF0["Chronic active ABMR"].isnull().to_numpy()
            )
            # E = DF0['Acute/active ABMR'].to_numpy()=='acute ABMR'
        elif out == "TCMR":
            # DF0['Acute TCMR'].unique() = [nan, 'TCMR IA', 'TCMR IIA', 'TCMR IIB', 'TCMR IB']
            # DF0['Chronic active TCMR'].unique() = [nan, 'Chronic TCMR']
            E = (~DF0["Acute TCMR"].isnull().to_numpy()) | (
                ~DF0["Chronic active TCMR"].isnull().to_numpy()
            )

        E = E * 1
        print("number of " + out + " = " + str(sum(E)))

        # censored time (time of event (if it happend) or last time we have notice)
        T = T_event.copy()
        T[E == 0] = T_censor[E == 0]

        # update DF0
        DF0["T_" + out] = T
        DF0["E_" + out] = E

    return DF0
