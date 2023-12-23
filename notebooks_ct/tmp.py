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
