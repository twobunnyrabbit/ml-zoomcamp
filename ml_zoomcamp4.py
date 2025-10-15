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
    import matplotlib.pyplot as plt
    from typing import List
    from dotenv import load_dotenv
    return load_dotenv, mo, np, os, pd, plt


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    return DictVectorizer, LogisticRegression, train_test_split


@app.cell
def _(load_dotenv, os, pd):
    load_dotenv()

    DATASET_PATH = os.getenv("DATASET_PATH")
    filename = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = pd.read_csv(DATASET_PATH + "/" + filename)
    return (df,)


@app.cell
def _(df, pd):
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)
    return


@app.cell
def _(df, train_test_split):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.churn.values
    y_val = df_val.churn.values
    y_test = df_test.churn.values

    del df_train['churn']
    del df_val['churn']
    del df_test['churn']
    return df_full_train, df_train, df_val, y_train, y_val


@app.cell
def _():
    numerical = ['tenure', 'monthlycharges', 'totalcharges']

    categorical = [
        'gender',
        'seniorcitizen',
        'partner',
        'dependents',
        'phoneservice',
        'multiplelines',
        'internetservice',
        'onlinesecurity',
        'onlinebackup',
        'deviceprotection',
        'techsupport',
        'streamingtv',
        'streamingmovies',
        'contract',
        'paperlessbilling',
        'paymentmethod',
    ]
    return categorical, numerical


@app.cell
def _(
    DictVectorizer,
    LogisticRegression,
    categorical,
    df_train,
    numerical,
    y_train,
):
    dv = DictVectorizer(sparse=False)

    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    return dv, model


@app.cell
def _(categorical, df_val, dv, model, numerical, y_val):
    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    y_pred = model.predict_proba(X_val)[:, 1]
    churn_decision = (y_pred >= 0.5)
    (y_val == churn_decision).mean()
    return churn_decision, y_pred


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Accuracy and Dummy model

    - `y_val` is the truth
    - `y_pred` is the soft prediction with a probability value
    - `churn_decision` is the calculated using `y_pred >= 0.5`
    """
    )
    return


@app.cell
def _(y_val):
    len(y_val)
    return


@app.cell
def _(y_val):
    y_val
    return


@app.cell
def _(y_val):
    y_val.sum()
    return


@app.cell
def _(churn_decision):
    churn_decision
    return


@app.cell
def _(churn_decision):
    churn_decision.sum()
    return


@app.cell
def _(churn_decision, y_val):
    (y_val == churn_decision).mean()
    return


@app.cell
def _(mo):
    mo.md(r"""The accuracy calculated above is based a on threshold of 0.5. What would the accuracy be like if the threshold changes?""")
    return


@app.cell
def _(np):
    thresholds = np.linspace(0, 1, 21)
    thresholds
    return (thresholds,)


@app.cell
def _(thresholds, y_pred, y_val):
    from sklearn.metrics import accuracy_score
    def _():
        scores = []
        for t in thresholds:
            score = accuracy_score(y_val, y_pred >= t)
            scores.append({'threshold': t, 'score': score})
        return scores

    scores = _()
    return (scores,)


@app.cell
def _(pd, scores):
    df_scores = pd.DataFrame(scores)
    df_scores
    return (df_scores,)


@app.cell
def _(df_scores, plt):
    plt.plot(df_scores.threshold, df_scores.score)
    return


@app.cell
def _(mo):
    mo.md(r"""0.5 seems to be the best threshold in terms of accuracy. However, accuracy does not tell us how good the model is.""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Confusion table""")
    return


@app.cell
def _(y_val):
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    return actual_negative, actual_positive


@app.cell
def _(y_pred):
    t = 0.5
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)
    return predict_negative, predict_positive


@app.cell
def _(actual_negative, actual_positive, predict_negative, predict_positive):
    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    return fn, fp, tn, tp


@app.cell
def _(fn, fp, tn, tp):
    {'tp': tp, 'fp': fp, 'tn': tn, 'fn' :fn}
    return


@app.cell
def _(fn, fp, np, tn, tp):
    confusion_matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    confusion_matrix
    return (confusion_matrix,)


@app.cell
def _(confusion_matrix):
    (confusion_matrix / confusion_matrix.sum()).round(2)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Precision and Recall

    - precision is the proporton of positive predictions that are correct
    - recall is the proportion of correctly predicted positive classes out of all actual positive classes


    |  | Predicted Negative  | Predicted Positive |
    |---|----|---|
    |Actual negative | True Negative (TN) | False Positive (FP)  |
    |Actual positive | False Negative (FN) | True Positive (TP)  |

    $\text{precision} = \frac{TP}{TP + FP}$

    $\text{recall} = \frac{TP}{TP + FN}$
    """
    )
    return


@app.cell
def _(fp, tp):
    precision = tp / (tp + fp)
    precision
    return


@app.cell
def _(fn, tp):
    recall = tp / (tp + fn)
    recall
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Recall - proportion of actual positive cases that was correctly predicted.

    Prediction = [P, P, P, N, P, N, N, P, N] 

    Actual     = [T, T, T, T, T, T, T, T, T]
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Precision - proportion of predicted positive cases that are actually positive.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Both recall and precision deal with positive states. For recall, the positive state is the actual positives. For precision, the positive state refers to all positive predictions or tests. Trying to remember:  

    - "RECALL" = "REAL"

    - "PRECISION" = "PREDICTION/TEST"

    The numerator is always the true positives.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## ROC Curves

    |  | Predicted Negative  | Predicted Positive |
    |---|----|---|
    |Actual negative | True Negative (TN) | False Positive (FP)  |
    |Actual positive | False Negative (FN) | True Positive (TP)  |

    $\text{False Positive Rate (FPR)} = \frac{FP}{TN + FP}$

    $\text{True Positive Rate (TPR)} = \frac{TP}{FN + FP}$

    Ideally, FPR should be as small as possible, TPR should be as large as possible.
    """
    )
    return


@app.cell
def _(fn, tp):
    tpr = tp / (tp + fn)
    tpr
    return


@app.cell
def _(fp, tn):
    fpr = fp / (fp + tn)
    fpr
    return


@app.cell
def _(actual_negative, actual_positive, np, y_pred):
    # calculate tpr and fpr for all thresholds

    def _():
        thresholds = np.linspace(0, 1, 101)
        scores = []
        for t in thresholds:
            predict_positive = (y_pred >= t)
            predict_negative = (y_pred < t)
            # actual_positive = (y_val == 1)
            # actual_negative = (y_val == 0)
            tp = (predict_positive & actual_positive).sum()
            tn = (predict_negative & actual_negative).sum()
            fp = (predict_positive & actual_negative).sum()
            fn = (predict_negative & actual_positive).sum()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            scores.append({'threshold': t, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'tpr': tpr, 'fpr': fpr})
        return scores
    tpr_fpr_thresholds = _()        
    return (tpr_fpr_thresholds,)


@app.cell
def _(pd, tpr_fpr_thresholds):
    df_tpr_fpr = pd.DataFrame(tpr_fpr_thresholds)
    df_tpr_fpr
    return (df_tpr_fpr,)


@app.cell
def _(df_tpr_fpr, plt):
    plt.plot(df_tpr_fpr.threshold, df_tpr_fpr.tpr, label='TPR')
    plt.plot(df_tpr_fpr.threshold, df_tpr_fpr.fpr, label='FPR')
    plt.legend()
    # plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""## Random model""")
    return


@app.cell
def _(np, y_val):
    np.random.seed(1)
    y_rand = np.random.uniform(0, 1, size=len(y_val))
    return (y_rand,)


@app.cell
def _(y_rand, y_val):
    ((y_rand >= 0.5) == y_val).mean()
    return


@app.cell
def _(np, pd):
    def tpr_fpr_df(y_val, y_pred):
        thresholds = np.linspace(0, 1, 101)
        scores = []
        for t in thresholds:
            predict_positive = (y_pred >= t)
            predict_negative = (y_pred < t)
            actual_positive = (y_val == 1)
            actual_negative = (y_val == 0)
            tp = (predict_positive & actual_positive).sum()
            tn = (predict_negative & actual_negative).sum()
            fp = (predict_positive & actual_negative).sum()
            fn = (predict_negative & actual_positive).sum()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            scores.append({'threshold': t, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'tpr': tpr, 'fpr': fpr})
        return pd.DataFrame(scores)
    return (tpr_fpr_df,)


@app.cell
def _(tpr_fpr_df, y_rand, y_val):
    df_rand = tpr_fpr_df(y_val, y_rand)
    df_rand[::10]
    return (df_rand,)


@app.cell
def _(df_rand, plt):
    plt.plot(df_rand.threshold, df_rand.tpr, label='TPR')
    plt.plot(df_rand.threshold, df_rand.fpr, label='FPR')
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(r"""## Ideal model""")
    return


@app.cell
def _(y_val):
    num_neg = (y_val == 0).sum()
    num_pos = (y_val == 1).sum()
    num_neg, num_pos
    return num_neg, num_pos


@app.cell
def _(np, num_neg, num_pos):
    y_ideal = np.repeat([0, 1], [num_neg, num_pos])
    y_ideal
    return (y_ideal,)


@app.cell
def _(np, y_val):
    y_ideal_pred = np.linspace(0, 1, len(y_val))
    y_ideal_pred
    return (y_ideal_pred,)


@app.cell
def _(y_ideal, y_ideal_pred):
    ((y_ideal_pred >= 0.726) == y_ideal).mean()
    return


@app.cell
def _(tpr_fpr_df, y_ideal, y_ideal_pred):
    df_ideal = tpr_fpr_df(y_ideal, y_ideal_pred)
    df_ideal[::10]
    return (df_ideal,)


@app.cell
def _(df_ideal, plt):
    plt.figure(figsize=(5,5))
    plt.plot(df_ideal.threshold, df_ideal.tpr, label='TPR')
    plt.plot(df_ideal.threshold, df_ideal.fpr, label='FPR')
    plt.legend()
    return


@app.cell
def _(df_tpr_fpr, plt):
    plt.figure(figsize=(5,5))
    plt.plot(df_tpr_fpr.fpr, df_tpr_fpr.tpr, label='model')
    plt.plot([0, 1], [0,1], label='random')
    # plt.plot(df_rand.fpr, df_rand.tpr, label='random')
    # plt.plot(df_ideal.fpr, df_ideal.tpr, label='ideal')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    return


@app.cell
def _():
    from sklearn.metrics import roc_curve
    return (roc_curve,)


@app.cell
def _(roc_curve, y_pred, y_val):
    fpr1, tpr1, thresholds_rc = roc_curve(y_val, y_pred)
    return fpr1, tpr1


@app.cell
def _(fpr1, plt, tpr1):
    plt.figure(figsize=(5,5))
    plt.plot(fpr1, tpr1, label='model')
    plt.plot([0, 1], [0,1], label='random')
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## ROC AUC

    - area under the ROC curve - useful metric
    - interpretation of AUC

    **AUC:** It is the probability of a randomly selected positive class that has a higher score than aa randomly selected negative class.
    """
    )
    return


@app.cell
def _():
    from sklearn.metrics import auc
    return (auc,)


@app.cell
def _(auc, df_tpr_fpr):
    auc(df_tpr_fpr['fpr'], df_tpr_fpr['tpr'])
    return


@app.cell
def _():
    from sklearn.metrics import roc_auc_score
    return (roc_auc_score,)


@app.cell
def _(roc_auc_score, y_pred, y_val):
    roc_auc_score(y_val, y_pred)
    return


@app.cell
def _(y_pred, y_val):
    # y_pred is the predicted probability from the model
    # selecting all the predictons where the actual class is negative
    neg = y_pred[y_val == 0]

    # selecting all the predictions where the actual class is positive
    pos = y_pred[y_val == 1]
    [neg, pos]
    return neg, pos


@app.cell
def _(neg, pos):
    [len(x) for x in [neg, pos]]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    There are 1023 actual negative classes and 386 actual positive classes. Within these classes are a range of scores for each observation.

    How likely is the predicted probability of a randomly selected actual positive class be higher compared to a random;y selected actual negative class? AUC gives us this probability. Approximately 84% of the time it will be correct.
    """
    )
    return


@app.cell
def _():
    import random
    return (random,)


@app.cell
def _(neg, pos, random):
    # randomly select a positive and a negative class
    # is the probability of the positive class going to be greater than the negative class?
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)
    return neg_ind, pos_ind


@app.cell
def _(neg, neg_ind, pos, pos_ind):
    pos[pos_ind] > neg[neg_ind]
    return


@app.cell
def _(neg, pos, random):
    def _():
        # now we perform this random process 10000 times
        n = 10000
        comparisons = []
        for i in range(n):
            pos_ind = random.randint(0, len(pos) - 1)
            neg_ind = random.randint(0, len(neg) - 1)
            is_bigger = pos[pos_ind] > neg[neg_ind]
            comparisons.append(is_bigger)
        return comparisons


    comparisons = _()
    return (comparisons,)


@app.cell
def _(comparisons, np):
    # after randomingly selecting a negative class and postive class,
    # the proportion of times that the score in a positive class is larger than the negative class is
    # 0.85
    np.array(comparisons).mean()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Cross-validation

    - evaluating the same model on different subsets of data
    - getting the average prediction and the spread within predictions
    """
    )
    return


@app.cell
def _(DictVectorizer, LogisticRegression, categorical, numerical):
    def train(df_train, y, C=1.0):
        dicts = df_train[categorical + numerical].to_dict(orient='records')
        dv = DictVectorizer(sparse=False)
        X_train = dv.fit_transform(dicts)

        model = LogisticRegression(C=C, max_iter=10000)
        model.fit(X_train, y)

        return dv, model
    return (train,)


@app.cell
def _(df_train, train, y_train):
    dv_train, model_train = train(df_train, y_train, C=0.001)
    return dv_train, model_train


@app.cell
def _(categorical, numerical):
    def predict(df, dv, model):
        dicts = df[categorical + numerical].to_dict(orient='records')
        X = dv.transform(dicts)
        y_pred = model.predict_proba(X)[:, 1]

        return y_pred
    
    return (predict,)


@app.cell
def _(df_val, dv_train, model_train, predict):
    y_pred2 = predict(df_val, dv_train, model_train )
    return


@app.cell
def _():
    from sklearn.model_selection import KFold
    return (KFold,)


@app.cell
def _():
    from tqdm.auto import tqdm
    return (tqdm,)


@app.cell
def _(KFold, df_full_train, predict, roc_auc_score, tqdm, train):
    n_splits = 5

    scores_c = []
    for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        scores_k = []
        for train_idx, val_idx in kfold.split(df_full_train):
            df_train_k = df_full_train.iloc[train_idx]
            df_val_k = df_full_train.iloc[val_idx]
    
            y_train_k = df_train_k.churn.values
            y_val_k = df_val_k.churn.values
        
            dv_train_k, model_train_k = train(df_train_k, y_train_k, C=C)
            y_pred3 = predict(df_val_k, dv_train_k, model_train_k)
            auc_k = roc_auc_score(y_val_k, y_pred3)
            scores_k.append(auc_k)
        scores_c.append({'C': C, 'fold': scores_k})
    return (scores_c,)


@app.cell
def _(scores_c):
    # [{'stats': f.__name__, 'val': round(f(scores_k), 3)} for f in [np.mean, np.std]]
    scores_c
    return


@app.cell
def _(np, pd, scores_c):
    # [{'mean': np.array(x['fold']).mean().round(3)} for x in scores_c]
    pd.DataFrame([{'C': x['C'], 'mean': np.array(x['fold']).mean().round(3), 'std': np.array(x['fold']).std().round(3)} for x in scores_c])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
