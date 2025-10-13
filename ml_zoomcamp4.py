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
    return load_dotenv, mo, np, os, pd


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
    return df_train, df_val, y_train, y_val


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
def _(df_scores):
    df_scores.plot()
    return


@app.cell
def _(mo):
    mo.md(r"""0.5 seems to be the best threshold in terms of accuracy.""")
    return


if __name__ == "__main__":
    app.run()
