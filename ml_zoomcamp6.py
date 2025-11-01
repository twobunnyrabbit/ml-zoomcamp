import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import dotenv
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from typing import List
    from dotenv import load_dotenv
    return load_dotenv, mo, np, os, pd, plt, sns


@app.cell
def _(load_dotenv, os):
    load_dotenv()

    DATASET_PATH = os.getenv("DATASET_PATH")
    filename = "CreditScoring.csv"
    return DATASET_PATH, filename


@app.cell
def _(DATASET_PATH, filename, pd):
    df = pd.read_csv(DATASET_PATH + "/" + filename)
    df.columns = df.columns.str.lower()
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.status.value_counts()
    return


@app.cell
def _(df):
    status_values = {1: "ok", 2: "default", 0: "unk"}

    df.status = df.status.map(status_values)
    return


@app.cell
def _(df):
    home_values = {
        1: "rent",
        2: "owner",
        3: "private",
        4: "ignore",
        5: "parents",
        6: "other",
        0: "unk",
    }

    df.home = df.home.map(home_values)

    marital_values = {
        1: "single",
        2: "married",
        3: "widow",
        4: "separated",
        5: "divorced",
        0: "unk",
    }

    df.marital = df.marital.map(marital_values)

    records_values = {1: "no", 2: "yes", 0: "unk"}

    df.records = df.records.map(records_values)

    job_values = {1: "fixed", 2: "partime", 3: "freelance", 4: "others", 0: "unk"}

    df.job = df.job.map(job_values)
    return


@app.cell
def _():
    # cat_vars = ['home', 'marital', 'records', 'job']
    # cat_values = [home_values, marital_values, records_values, job_values]

    # for i in range(len(cat_vars)):
    #     df[cat_vars[i]] = df[cat_vars[i]].map(cat_values[i])
    return


@app.cell
def _(df):
    df.describe().round()
    return


@app.cell
def _(df, np):
    for c in ["income", "assets", "debt"]:
        df[c] = df[c].replace(to_replace=99999999, value=np.nan)
    return


@app.cell
def _(df):
    df_2 = df[df.status != "unk"].reset_index(drop=True)
    return (df_2,)


@app.cell
def _(df_2):
    from sklearn.model_selection import train_test_split

    df_full_train, df_test = train_test_split(df_2, test_size=0.2, random_state=11)
    df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=11)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = (df_train.status == "default").astype(int).values
    y_val = (df_val.status == "default").astype(int).values
    y_test = (df_test.status == "default").astype(int).values

    del df_train["status"]
    del df_val["status"]
    del df_test["status"]
    return df_full_train, df_test, df_train, df_val, y_train, y_val


@app.cell
def _(df_train):
    df_train
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.3 Decision trees
    """)
    return


@app.function
def assess_risk(client):
    if client["records"] == "yes":
        if client["job"] == "parttime":
            return "default"
        else:
            return "ok"
    else:
        if client["assets"] > 6000:
            return "ok"
        else:
            return "default"


@app.cell
def _(df_train):
    xi = df_train.iloc[0].to_dict()
    assess_risk(xi)
    return


@app.cell
def _():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.feature_extraction import DictVectorizer
    return DecisionTreeClassifier, DictVectorizer


@app.cell
def _(df_train):
    train_dicts = df_train.fillna(0).to_dict(orient="records")
    return (train_dicts,)


@app.cell
def _(train_dicts):
    train_dicts[:5]
    return


@app.cell
def _(DictVectorizer, train_dicts):
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    return X_train, dv


@app.cell
def _(dv):
    dv.get_feature_names_out()
    return


@app.cell
def _(DecisionTreeClassifier, X_train, y_train):
    dt_test = DecisionTreeClassifier(max_depth=3)
    dt_test.fit(X_train, y_train)
    return (dt_test,)


@app.cell
def _(df_val, dv):
    val_dicts = df_val.fillna(0).to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    return (X_val,)


@app.cell
def _(X_val, dt_test):
    y_pred = dt_test.predict_proba(X_val)[:, 1]
    return (y_pred,)


@app.cell
def _(y_pred):
    y_pred
    return


@app.cell
def _(y_pred, y_val):
    from sklearn.metrics import roc_auc_score

    roc_auc_score(y_val, y_pred)
    return (roc_auc_score,)


@app.cell
def _(X_train, dt_test, roc_auc_score, y_train):
    # auc for training dataset
    y_pred_train = dt_test.predict_proba(X_train)[:, 1]
    roc_auc_score(y_train, y_pred_train)
    return


@app.cell
def _(mo):
    mo.md("""
    r"Validation AUC is 0.65 and training AUF is 1.0 - this represents overfitting."
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    X_train,
    X_val,
    mo,
    roc_auc_score,
    y_train,
    y_val,
):
    # restricting max_depth=3
    def _():
        dt = DecisionTreeClassifier(max_depth=3)
        dt.fit(X_train, y_train)
        y_pred_train = dt.predict_proba(X_train)[:, 1]
        auc_train = roc_auc_score(y_train, y_pred_train)
        y_pred_val = dt.predict_proba(X_val)[:, 1]
        auc_val = roc_auc_score(y_val, y_pred_val)
        return mo.md(f"""
                     With `max-depth=3`    
                     **AUC train:** {auc_train} <br> **AUC val:** {auc_val}
        """)

    _()
    return


@app.cell
def _():
    from sklearn.tree import export_text
    return (export_text,)


@app.cell
def _(dt_test, export_text):
    print(export_text(dt_test))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.4 - Decision tree learning algorithm

    This section is to find `assets > T`
    """)
    return


@app.cell
def _():
    data = [
        [8000, "default"],
        [2000, "default"],
        [0, "default"],
        [5000, "ok"],
        [5000, "ok"],
        [4000, "ok"],
        [19000, "ok"],
        [3000, "default"],
    ]
    return (data,)


@app.cell
def _(data, pd):
    df_example = pd.DataFrame(data, columns=["assets", "status"])
    return (df_example,)


@app.cell
def _(df_example):
    df_example
    return


@app.cell
def _(df_example):
    df_example.sort_values("assets")
    return


@app.cell
def _():
    # For each of these T, cut the dataset into LEFT or RIGHT
    Ts = [0, 2000, 3000, 4000, 5000, 8000]
    return (Ts,)


@app.cell
def _():
    from IPython.display import display
    return


@app.function
def get_impurity(df, target, impurity):
    dict = df[target].value_counts(normalize=True).to_dict()
    return dict.get(impurity, 0)


@app.function
# LEFT = 'default'
# RIGHT = 'ok'
def impurities_per_thresholds(Ts, df, feature, target, left_code, right_code):
    impurities = []
    for T in Ts:
        df_left = df[df[feature] <= T]
        # get impurity left
        # display(df_left.status.value_counts(normalize=True).to_dict())
        df_right = df[df[feature] > T]
        left = get_impurity(df_left, target, left_code)
        right = get_impurity(df_right, target, right_code)
        impurities.append({"T": T, "LEFT": left, "RIGHT": right})
    return impurities


@app.cell
def _(Ts, df_example):
    impurity_1 = impurities_per_thresholds(
        Ts, df_example, "assets", "status", "ok", "default"
    )
    return (impurity_1,)


@app.cell
def _(impurity_1, pd):
    df_impurity_1 = pd.DataFrame(impurity_1)
    df_impurity_1
    return (df_impurity_1,)


@app.cell
def _(df_impurity_1):
    df_impurity_1["AVG"] = df_impurity_1.iloc[:, 1:].mean(axis=1)
    return


@app.cell
def _(df_impurity_1):
    df_impurity_1
    return


@app.cell
def _():
    data_2 = [
        [8000, 3000, "default"],
        [2000, 1000, "default"],
        [0, 1000, "default"],
        [5000, 1000, "ok"],
        [5000, 1000, "ok"],
        [4000, 1000, "ok"],
        [19000, 500, "ok"],
        [3000, 2000, "default"],
    ]
    return (data_2,)


@app.cell
def _(data_2, pd):
    df_data_2 = pd.DataFrame(data_2, columns=["assets", "debt", "status"])
    df_data_2
    return (df_data_2,)


@app.cell
def _(df_data_2, np):
    # get thresholds for debt
    debt_thresholds = list(np.sort(df_data_2["debt"].unique()))[:-1]
    debt_thresholds
    return (debt_thresholds,)


@app.cell
def _(debt_thresholds, df_data_2):
    impurity_2 = impurities_per_thresholds(
        debt_thresholds, df_data_2, "debt", "status", "default", "ok"
    )
    impurity_2
    return (impurity_2,)


@app.cell
def _(impurity_2, pd):
    df_impurity_2 = pd.DataFrame(impurity_2)
    df_impurity_2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.5 Decision trees parameter tuning

    - selecting `max_depth`
    - selecting `min_samples_leaf`
    """)
    return


@app.cell
def _(DecisionTreeClassifier, X_train, X_val, roc_auc_score, y_train, y_val):
    def _():
        for d in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:
            dt = DecisionTreeClassifier(max_depth=d)
            dt.fit(X_train, y_train)

            y_pred = dt.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            print("%4s -> %.3f" % (d, auc))

    _()
    return


@app.cell
def _(
    DecisionTreeClassifier,
    X_train,
    X_val,
    pd,
    roc_auc_score,
    y_train,
    y_val,
):
    def _():
        scores = []
        for d in [4, 5, 6, 7, 10, 15, 20, None]:
            for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
                dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=s)
                dt.fit(X_train, y_train)

                y_pred = dt.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)
                # print('%4s, %3d -> %.3f' % (d, s, auc))
                scores.append({"max_depth": d, "min_leaf": s, "auc": auc})
        return pd.DataFrame(scores)

    df_tuning = _()
    return (df_tuning,)


@app.cell
def _(df_tuning):
    df_pivot = df_tuning.pivot(columns="max_depth", index="min_leaf", values="auc")
    return (df_pivot,)


@app.cell
def _(df_pivot):
    df_pivot.round(3)
    return


@app.cell
def _(df_pivot, sns):
    sns.heatmap(df_pivot, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5)
    return


@app.cell
def _(DecisionTreeClassifier, X_train, y_train):
    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20)
    dt.fit(X_train, y_train)
    return (dt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.6 Ensembles and random forest

    - board of experts
    - ensembling models
    - random forest - ensembling decision trees
    - tuning random forest
    """)
    return


@app.cell
def _():
    from sklearn.ensemble import RandomForestClassifier
    return (RandomForestClassifier,)


@app.cell
def _(RandomForestClassifier, X_train, y_train):
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(X_train, y_train)
    return (rf,)


@app.cell
def _(X_val, rf, roc_auc_score, y_val):
    def _():
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        print(auc)

    _()
    return


@app.cell
def _(
    RandomForestClassifier,
    X_train,
    X_val,
    np,
    roc_auc_score,
    y_train,
    y_val,
):
    def _():
        results = []
        for n in range(10, 210, 10):
            rf = RandomForestClassifier(n_estimators=n, random_state=1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            results.append({"n_est": n, "auc": np.round(auc, 4)})
        return results

    results = _()
    return (results,)


@app.cell
def _(pd, results):
    df_est = pd.DataFrame(results)
    return (df_est,)


@app.cell
def _(df_est):
    df_est
    return


@app.cell
def _(df_est, plt):
    plt.plot(df_est["n_est"], df_est["auc"])
    return


@app.cell
def _(
    RandomForestClassifier,
    X_train,
    X_val,
    np,
    roc_auc_score,
    y_train,
    y_val,
):
    def _():
        results = []
        for d in [5, 10, 15]:
            for n in range(10, 210, 10):
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
                rf.fit(X_train, y_train)
                y_pred = rf.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)
                results.append({"n_est": n, "depth": d, "auc": np.round(auc, 4)})
        return results

    res_depth_n_est = _()
    return (res_depth_n_est,)


@app.cell
def _(pd, res_depth_n_est):
    df_depth_n_est = pd.DataFrame(res_depth_n_est)
    return (df_depth_n_est,)


@app.cell
def _(df_depth_n_est):
    df_depth_n_est
    return


@app.cell
def _(
    RandomForestClassifier,
    X_train,
    X_val,
    np,
    roc_auc_score,
    y_train,
    y_val,
):
    def _():
        results = []
        for s in [1, 3, 5, 10, 50]:
            for n in range(10, 210, 10):
                rf = RandomForestClassifier(
                    n_estimators=n, max_depth=10, min_samples_leaf=s, random_state=1
                )
                rf.fit(X_train, y_train)
                y_pred = rf.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)
                results.append({"n_est": n, "leaf": s, "auc": np.round(auc, 4)})
        return results

    res_leaf_n_est = _()
    return (res_leaf_n_est,)


@app.cell
def _(pd, res_leaf_n_est):
    df_leaf = pd.DataFrame(res_leaf_n_est)
    return (df_leaf,)


@app.cell
def _(df_leaf):
    df_leaf
    return


@app.cell
def _(df_leaf, plt):
    for l in [1, 3, 5, 10, 50]:
        df_subset = df_leaf[df_leaf.leaf == l]
        plt.plot(df_subset.n_est, df_subset.auc, label="leaf=%d" % l)
    plt.legend()
    return


@app.cell
def _(RandomForestClassifier, X_train, y_train):
    rf_final = RandomForestClassifier(
        max_depth=10, min_samples_leaf=3, n_estimators=200, random_state=1
    )
    rf_final.fit(X_train, y_train)
    return (rf_final,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Gradient boosting and XGBoost
    - Gradient boosting vs random forest
    - Installing XGBoost
    - Training the first model
    - Performance monitoring
    - Parsing xgboot's monitoring output
    """)
    return


@app.cell
def _():
    import xgboost as xgb
    return (xgb,)


@app.cell
def _(X_train, X_val, dv, xgb, y_train, y_val):
    features = dv.get_feature_names_out().tolist()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    return dtrain, dval


@app.cell
def _(dtrain, dval, xgb):
    xgb_params = {
        "eta": 0.3,
        "max_depth": 6,
        "min_child_weight": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "nthread": 8,
        "seed": 1,
        "verbosity": 1,
    }

    watchlist = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(
        xgb_params, dtrain, evals=watchlist, verbose_eval=5, num_boost_round=200
    )
    return (model,)


@app.cell
def _():
    output = """
    [0]	train-auc:0.86357	val-auc:0.78606
    [5]	train-auc:0.92758	val-auc:0.81751
    [10]	train-auc:0.95403	val-auc:0.82117
    [15]	train-auc:0.96671	val-auc:0.82120
    [20]	train-auc:0.97513	val-auc:0.82477
    [25]	train-auc:0.98070	val-auc:0.82108
    [30]	train-auc:0.98582	val-auc:0.82050
    [35]	train-auc:0.98803	val-auc:0.82172
    [40]	train-auc:0.99160	val-auc:0.82195
    [45]	train-auc:0.99365	val-auc:0.82057
    [50]	train-auc:0.99552	val-auc:0.82064
    [55]	train-auc:0.99700	val-auc:0.81980
    [60]	train-auc:0.99760	val-auc:0.82052
    [65]	train-auc:0.99842	val-auc:0.81908
    [70]	train-auc:0.99904	val-auc:0.81786
    [75]	train-auc:0.99942	val-auc:0.81761
    [80]	train-auc:0.99970	val-auc:0.81761
    [85]	train-auc:0.99980	val-auc:0.81609
    [90]	train-auc:0.99993	val-auc:0.81515
    [95]	train-auc:0.99995	val-auc:0.81426
    [100]	train-auc:0.99997	val-auc:0.81465
    [105]	train-auc:0.99999	val-auc:0.81242
    [110]	train-auc:0.99999	val-auc:0.81072
    [115]	train-auc:1.00000	val-auc:0.81010
    [120]	train-auc:1.00000	val-auc:0.80812
    [125]	train-auc:1.00000	val-auc:0.80951
    [130]	train-auc:1.00000	val-auc:0.80836
    [135]	train-auc:1.00000	val-auc:0.80861
    [140]	train-auc:1.00000	val-auc:0.80838
    [145]	train-auc:1.00000	val-auc:0.80763
    [150]	train-auc:1.00000	val-auc:0.80771
    [155]	train-auc:1.00000	val-auc:0.80722
    [160]	train-auc:1.00000	val-auc:0.80573
    [165]	train-auc:1.00000	val-auc:0.80632
    [170]	train-auc:1.00000	val-auc:0.80595
    [175]	train-auc:1.00000	val-auc:0.80601
    [180]	train-auc:1.00000	val-auc:0.80598
    [185]	train-auc:1.00000	val-auc:0.80649
    [190]	train-auc:1.00000	val-auc:0.80589
    [195]	train-auc:1.00000	val-auc:0.80620
    [199]	train-auc:1.00000	val-auc:0.80613

    """
    return (output,)


@app.cell
def _():
    import re

    def replace_numbers(s):
        replaced_s = re.sub(r"\[.*?\]\s", "", s)
        # s_split= [x.split('\t') for x in replaced_s.split(' ')]
        # return {'train_auc': train, 'val_auc': va}
        # print(s_split)
        return replaced_s
    return re, replace_numbers


@app.function
def split_output(s):
    return s.lstrip().rstrip().split("\n")


@app.cell
def _(re):
    # get iterations
    def get_iterations(s):
        iteration = re.match(r"^\[(\d+)]", s)
        if iteration:
            return iteration[1]
    return (get_iterations,)


@app.function
def split_train_val_auc(s):
    train, val = s.split("\t")
    # print(f'{train} : {val}')
    return {
        "train_auc": float(train.split(":")[1]),
        "val_auc": float(val.split(":")[1]),
    }


@app.cell
def _(output):
    # get initial splits
    initial_splits = split_output(output)
    return (initial_splits,)


@app.cell
def _(get_iterations, initial_splits):
    # get the iteration numbers
    iteration_list = [int(get_iterations(x)) for x in initial_splits]
    return (iteration_list,)


@app.cell
def _(output, replace_numbers):
    train_val_auc = [replace_numbers(x) for x in split_output(output)]
    return (train_val_auc,)


@app.cell
def _(train_val_auc):
    final_splits = [split_train_val_auc(x) for x in train_val_auc]
    return (final_splits,)


@app.cell
def _(final_splits, iteration_list, pd):
    df_aucs = pd.DataFrame(final_splits)
    df_aucs["iterations"] = iteration_list
    return (df_aucs,)


@app.cell
def _(df_aucs):
    df_aucs
    return


@app.cell
def _(df_aucs, plt):
    plt.plot(df_aucs["iterations"], df_aucs["train_auc"], label="train")
    plt.plot(df_aucs["iterations"], df_aucs["val_auc"], label="val")
    plt.legend()
    return


@app.cell
def _(df_aucs, plt):
    plt.plot(df_aucs["iterations"], df_aucs["val_auc"], label="val")
    return


@app.cell
def _(dval, mo, model, roc_auc_score, y_val):
    def _():
        y_pred = model.predict(dval)
        auc = roc_auc_score(y_val, y_pred)
        return mo.md(f"AUC = {auc}")

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.8 XGBost parameter tuning

    ### Tuning order
    - eta: learning rate (size of each step)
    - max_depth
    - min_child_weight (similar to min_samples_leaf in random forest)

    Other useful parameters to tune:
    - subsample
    - colsample_bytree
    - lambda
    - alpha
    """)
    return


@app.cell
def _(dtrain, dval, xgb):
    def train_xgboost_with_captured_output(eta=0.3, max_depth=6, min_child_weight=1):
        xgb_params = {
            "eta": eta,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "nthread": 8,
            "seed": 1,
            "verbosity": 1,
        }
        evals_results = {}

        watchlist = [(dtrain, "train"), (dval, "val")]
        model = xgb.train(
            xgb_params,
            dtrain,
            evals=watchlist,
            verbose_eval=False,
            evals_result=evals_results,
            num_boost_round=200,
        )

        return evals_results

    # evals_result = train_xgboost_with_captured_output(0.3)
    return (train_xgboost_with_captured_output,)


@app.cell
def _(train_xgboost_with_captured_output):
    etas = [
        {"eta": eta, "xgb": train_xgboost_with_captured_output(eta)}
        for eta in [0.01, 0.05, 0.1, 0.3, 1.0]
    ]
    return (etas,)


@app.cell
def _(etas, plt):
    for e in etas:
        eta = e.get("eta")
        iters = list(range(len(e.get("xgb").get("train").get("auc"))))
        plt.plot(iters, e["xgb"]["val"]["auc"], label=eta)
    plt.legend()
    return


@app.cell
def _(train_xgboost_with_captured_output):
    max_depth = [
        {
            "max_depth": depth,
            "xgb": train_xgboost_with_captured_output(eta=0.1, max_depth=depth),
        }
        for depth in [6, 3, 4, 10]
    ]
    return (max_depth,)


@app.cell
def _(max_depth, plt):
    for d in max_depth:
        depth = d.get("max_depth")
        if depth != 10:
            iters_depth = list(range(len(d.get("xgb").get("val").get("auc"))))
            plt.plot(iters_depth, d["xgb"]["val"]["auc"], label=f"max-dept={depth}")
    plt.ylim(0.8, 0.84)
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(r"""
    `max-depth=3` is the better choice.
    """)
    return


@app.cell
def _(train_xgboost_with_captured_output):
    min_child = [
        {
            "min_child": child,
            "xgb": train_xgboost_with_captured_output(
                eta=0.1, max_depth=3, min_child_weight=child
            ),
        }
        for child in [1, 10, 30]
    ]
    return (min_child,)


@app.cell
def _(min_child):
    min_child
    return


@app.cell
def _(min_child, plt):
    for ch in min_child:
        child = ch.get("min_child")
        iters_child = list(range(len(ch.get("xgb").get("val").get("auc"))))
        plt.plot(iters_child, ch["xgb"]["val"]["auc"], label=f"min_child={child}")
    plt.ylim(0.81, 0.84)
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Train for 125 trees with min_child of 10
    """)
    return


@app.cell
def _(dtrain, xgb):
    xgb_params_final = {
        "eta": 0.1,
        "max_depth": 4,
        "min_child_weight": 10,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "nthread": 8,
        "seed": 1,
        "verbosity": 1,
    }
    model_final = xgb.train(xgb_params_final, dtrain, num_boost_round=125)
    return (model_final,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6.9 Selecting the final model
    - choosing between xgboost, random forest and decision tree
    - training the final model
    - saving the model
    """)
    return


@app.cell
def _(X_val, dt, dval, model_final, rf_final):
    y_pred_dt = dt.predict_proba(X_val)[:, 1]
    y_pred_rf = rf_final.predict_proba(X_val)[:, 1]
    y_pred_xgb = model_final.predict(dval)
    return y_pred_dt, y_pred_rf, y_pred_xgb


@app.cell
def _(roc_auc_score, y_pred_dt, y_pred_rf, y_pred_xgb, y_val):
    model_aucs = [
        {"model": x[0], "auc": roc_auc_score(y_val, x[1])}
        for x in zip(["dt", "rf", "xgb"], [y_pred_dt, y_pred_rf, y_pred_xgb])
    ]
    return (model_aucs,)


@app.cell
def _(model_aucs, pd):
    pd.DataFrame(model_aucs)
    return


@app.cell
def _(df_full_train):
    df_full_train_2 = df_full_train.reset_index(drop=True)
    return (df_full_train_2,)


@app.cell
def _(df_full_train_2):
    y_full_train = (df_full_train_2.status == "default").astype(int).values
    return (y_full_train,)


@app.cell
def _(df_full_train_2):
    del df_full_train_2["status"]
    return


@app.cell
def _(df_full_train_2):
    dict_full_train = df_full_train_2.to_dict(orient="records")
    return (dict_full_train,)


@app.cell
def _(DictVectorizer, dict_full_train):
    dv_2 = DictVectorizer(sparse=False)
    X_full_train = dv_2.fit_transform(dict_full_train)
    return X_full_train, dv_2


@app.cell
def _(df_test, dv_2):
    dict_test = df_test.to_dict(orient="records")
    X_test = dv_2.fit_transform(dict_test)
    return (X_test,)


@app.cell
def _(dv_2):
    len(dv_2.get_feature_names_out().tolist())
    return


@app.cell
def _(X_test):
    len(X_test[0])
    return


@app.cell
def _(df_full_train):
    df_full_train.shape
    return


@app.cell
def _(X_full_train, X_test, dv_2, xgb, y_full_train):
    dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv_2.get_feature_names_out().tolist())
    # X_test = dv_2.transform(dict_test)
    dtest = xgb.DMatrix(X_test, feature_names=dv_2.get_feature_names_out().tolist())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary
    - Decision tress learn if-then-else rules from data.
    - Finding the best split: select the least impure split. This algorithm can overfit, that's why we control it by limiting the max depth and the size of the group.
    - Random forest is a way of combining multiple decision trees. It should have a diverse set of models to make good predictions
    - Gradient boosing trains moedl sequentially: each model tries to fix errors of the previous model. XGBoost is an implementation of gradient boosting.
    -
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
