import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import dotenv
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from typing import List
    from dotenv import load_dotenv
    return load_dotenv, mo, np, os, pd


@app.cell
def _(load_dotenv, os, pd):
    load_dotenv()

    DATASET_PATH = os.getenv("DATASET_PATH")
    filename = "course_lead_scoring.csv"
    df = pd.read_csv(DATASET_PATH + "/" + filename)
    # df = df.dropna()
    return DATASET_PATH, df, filename


@app.cell
def _(DATASET_PATH, filename, pd):
    df0 = pd.read_csv(DATASET_PATH + "/" + filename)
    return


@app.cell
def _(mo):
    mo.md(r"""Target variable is `converted`""")
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""## Data preparation""")
    return


@app.cell
def _(mo):
    mo.md(r"""Check for missing data""")
    return


@app.cell
def _(df, pd):
    is_missing = df.isna().sum() > 0
    df_missing = pd.DataFrame({'num_missing': df.isna().sum()[is_missing], 'dtype': df.dtypes[is_missing]})
    df_missing
    return (df_missing,)


@app.cell
def _(df_missing):
    missing_cat = df_missing[df_missing.dtype == 'object'].index.to_list()
    missing_num = df_missing[df_missing.dtype != 'object'].index.to_list()
    return missing_cat, missing_num


@app.cell
def _(df, missing_cat, missing_num):
    # fill missing variables with NA or 0.0
    for c in missing_cat + missing_num:
        is_object = c in missing_cat
        replace_with = 'NA' if is_object else 0.0
        df.fillna({c: replace_with}, inplace=True)
    return


@app.cell
def _(df):
    # check if all missing variables are filled
    df.isna().sum()
    return


@app.cell
def _(mo):
    mo.md(r"""Split the data into 3 parts: train/validation/test with 60%/20%/20% distribution. Use `train_test_split` function for that with `random_state=1`""")
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    return (train_test_split,)


@app.cell
def _(df, train_test_split):
    # split the respective dataframes
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    # reset index
    df_full_train.reset_index(drop=True, inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    global_train_converted = df_train['converted'].mean()

    # initialise the target to their respective variables
    y_train, y_val, y_test = [x['converted'].values for x in [df_train, df_val, df_test]]

    # drop the target variable in the features dataframe
    df_train, df_val, df_test = [df.drop(columns=['converted']) for df in [df_train, df_val, df_test]]
    return df_full_train, df_test, df_train, df_val, y_train, y_val


@app.cell
def _(df, df_full_train, df_test, df_train, df_val, pd):
    # check size of splits
    df_names = ['df', 'df_full_train', 'df_train', 'df_val', 'df_test']
    df_list = [df, df_full_train, df_train, df_val, df_test]
    df_zip = zip(df_names, df_list)

    split_sizes = []
    for d in df_zip:
        split_sizes.append({'df_name': d[0], 'shape': d[1].shape})
    pd.DataFrame(split_sizes)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    - `df_train` has 876 records
    - `df_val` has 293 records
    - `df_test` has 293 records
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Question 1: ROC AUC feature importance""")
    return


@app.cell
def _(df_train):
    numerics = df_train.dtypes[df_train.dtypes != 'object'].index.to_list()
    numerics
    return (numerics,)


@app.cell
def _(df_train, numerics):
    df_train[numerics]
    return


@app.cell
def _():
    from sklearn.metrics import roc_auc_score
    return (roc_auc_score,)


@app.cell
def _():
    # roc_auc_score(df_train['converted'], df_train['number_of_courses_viewed'], multi_class='ovr')
    # df_train['number_of_courses_viewed'].values
    # df_train['converted'].values
    return


@app.cell
def _(df_train, numerics, roc_auc_score, y_train):
    roc_scores = []
    for f in numerics:
        if (f != 'converted'):
            s = roc_auc_score(y_train, df_train[f], multi_class='ovr')
            if s < 0.5:
                s = roc_auc_score(y_train, -df_train[f], multi_class='ovr')
            roc_scores.append({'feature': f, 'auc': s})
    return (roc_scores,)


@app.cell
def _(pd, roc_scores):
    pd.DataFrame(roc_scores)
    return


@app.cell
def _(mo):
    mo.md(r"""`number_of_courses_viewed` has the highest AUC at 0.764""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 2: Training the model

    Apply one-hot-encoding using DictVectorizer and train the logistic regression with these parameters:

    `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`
    """
    )
    return


@app.cell
def _():
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    return DictVectorizer, LogisticRegression


@app.cell
def _(DictVectorizer):
    dv = DictVectorizer(sparse=False)
    return (dv,)


@app.cell
def _(df_train, dv):
    train_dicts = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    return (X_train,)


@app.cell
def _(LogisticRegression, X_train, y_train):
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000).fit(X_train, y_train)
    return (model,)


@app.cell
def _(df_val, dv, model):
    X_val = dv.fit_transform(df_val.to_dict(orient='records'))
    y_pred = model.predict_proba(X_val)[:, 1]
    # y_pred
    return (y_pred,)


@app.cell
def _(roc_auc_score, y_pred, y_val):
    round(roc_auc_score(y_val, y_pred),3)
    return


@app.cell
def _(mo):
    mo.md(r"""AUC for validation dataset is 0.817""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 3: Precision and Recall

    Evaluate the model on all thresholds from 0.0 to 1.0 with step 0.01
    """
    )
    return


@app.cell
def _(np, y_pred, y_val):
    thresholds = np.arange(0.00001, 1.0, 0.01)
    precision_recall = []
    for t in thresholds:
        pred_pos = (y_pred >= t)
        pred_neg = (y_pred < t)
        true_pos = (y_val == 1)
        true_neg = (y_val == 0)
        tp = (pred_pos & true_pos).sum()
        tn = (pred_neg & true_neg).sum()
        fp = (pred_pos & true_neg).sum()
        fn = (pred_neg & true_pos).sum()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        # print(f'tp: {tp}\ttn: {tn}\t fp: {fp}\tfn: {fn}')
        # print(f'threshold: {t:.3f}\ttpr:{tpr:.3f}\tfpr:{fpr:.3f}\tprecision:{precision:.3f}\trecall:{recall:.3f}')
        precision_recall.append({'t': t, 'precision': precision, 'recall': recall})
    return (precision_recall,)


@app.cell
def _(pd, precision_recall):
    df_precision_recall = pd.DataFrame(precision_recall)
    df_precision_recall
    return (df_precision_recall,)


@app.cell
def _(df_precision_recall):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(df_precision_recall['t'], df_precision_recall['precision'], label='Precision', color='blue')
    plt.plot(df_precision_recall['t'], df_precision_recall['recall'], label='Recall', color='red')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 4: F1 score

    $$
    F_1 = \frac{P\cdot R}{P + R}
    $$

    At which threshold is $F_1$ maximal?
    """
    )
    return


@app.function
def get_f1(precision, recall):
    return 2*precision*recall / (precision + recall)


@app.cell
def _(precision_recall):
    f1_scores = [{'t': x.get('t'), 'f1': get_f1(x.get('precision'), x.get('recall'))} for x in precision_recall]
    sorted_f1_scores = sorted(f1_scores, key=lambda x: x['f1'], reverse=True)
    return (sorted_f1_scores,)


@app.cell
def _(sorted_f1_scores):
    sorted_f1_scores[0]
    return


@app.cell
def _(mo):
    mo.md(r"""Maximal $F_1$ is 0.812 with threshold of 0.57""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 5: 5-Fold CV

    Use the KFold class from Scikit-Learn to evaluate our model on 5 different folds:  
    `KFold(n_splits=5, shuffle=True, random_state=1)`

    - Iterate over different folds of df_full_train  
    - Split the data into train and validation  
    - Train the model on train with these parameters: `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`  
    - Use AUC to evaluate the model on validation
    """
    )
    return


@app.cell
def _():
    from sklearn.model_selection import KFold
    from tqdm.auto import tqdm
    return (KFold,)


@app.cell
def _(KFold):
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    return (kfold,)


@app.cell
def _(LogisticRegression, df_full_train, dv, kfold, roc_auc_score):
    def perform_kfold(C=1.0):
        scores = []
        for i, (train_idx, val_idx) in enumerate(kfold.split(df_full_train)):
            # get target
            y_train = df_full_train.converted[train_idx]
            y_val = df_full_train.converted[val_idx]

            # # get features and drop target
            df_t = df_full_train.iloc[train_idx]
            df_v = df_full_train.iloc[val_idx]
            df_t, df_v = [df.drop(columns=['converted']) for df in [df_t, df_v]]

            # # one-hot encode
            X_train = dv.fit_transform(df_t.to_dict(orient='records'))
            X_val = dv.fit_transform(df_v.to_dict(orient='records'))

            # # train the model
            model = LogisticRegression(solver='liblinear', C=C, max_iter=1000).fit(X_train, y_train)

            # use the model
            y_pred = model.predict_proba(X_val)[:, 1]
            auc_k = roc_auc_score(y_val, y_pred)
            scores.append({'fold': i, 'auc': auc_k})
        return scores

    result = perform_kfold()
    return perform_kfold, result


@app.cell
def _(pd, result):
    df_result = pd.DataFrame(result)
    df_result
    return (df_result,)


@app.cell
def _(df_result):
    df_result['auc'].std()
    return


@app.cell
def _(mo):
    mo.md(r"""Standard deviation is 0.04 across the 5 folds.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 6: Hyperparameter Tuning

    Now let's use 5-Fold cross-validation to find the best parameter C

    - Iterate over the following C values: `[0.000001, 0.001, 1]`  
    - Initialize KFold with the same parameters as previously
    - Use these parameters for the model: `LogisticRegression(solver='liblinear', C=C, max_iter=1000)`
    - Compute the mean score as well as the std (round the mean and std to 3 decimal digits)

    Which C leads to the best mean score?
    """
    )
    return


@app.cell
def _(perform_kfold):
    c_values = [0.0000001, 0.001, 1]

    hyper_p = []

    for c_v in c_values:
        hyper_p.append({'C': c_v, 'AUCs_per_fold' :perform_kfold(c_v)})
    return (hyper_p,)


@app.cell
def _(hyper_p):
    hyper_p
    return


@app.cell
def _(np):
    def get_mean_std(list_dict):
        """
        x is a list of dictionary with keys fold and auc
        [{'fold': 0, 'auc': 0.55}, ...]
        """
        data = np.array([x.get('auc') for x in list_dict])
        result = [f(data) for f in [np.mean, np.std]]
        return result
    return (get_mean_std,)


@app.cell
def _(get_mean_std, hyper_p):
    hyper_p_results = []
    for p in hyper_p:
        m_std = get_mean_std(p.get('AUCs_per_fold'))
        hyper_p_results.append({'C': p.get('C'), 'mean': m_std[0], 'std': m_std[1]})
    return (hyper_p_results,)


@app.cell
def _(hyper_p_results):
    hyper_p_results
    return


@app.cell
def _(hyper_p_results, pd):
    pd.DataFrame(hyper_p_results)
    return


@app.cell
def _(mo):
    mo.md(r"""C = 0.001 has the best mean score of 0.867""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
