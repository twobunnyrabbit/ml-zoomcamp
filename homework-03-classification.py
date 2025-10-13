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

    load_dotenv()

    DATASET_PATH = os.getenv("DATASET_PATH")
    filename = "course_lead_scoring.csv"
    return DATASET_PATH, List, filename, mo, pd


@app.cell
def _(DATASET_PATH, filename, pd):
    df = pd.read_csv(DATASET_PATH + "/" + filename)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Data preparation
    - target variable is `converted`
    """
    )
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    # check missing variables
    df.isna().sum()
    return


@app.cell
def _(df):
    var_na_list = df.isna().sum()[(df.isna().sum() > 0)].index.to_list()
    var_na_list
    return (var_na_list,)


@app.cell
def _(List, pd, var_na_list):
    def fill_nas(df: pd.DataFrame, v: List[str]):
        for v in var_na_list:
            is_object = df[v].dtype == 'object'
            replacement_value = 'NA' if is_object else 0.0
            df.fillna({v: replacement_value}, inplace=True)
    return (fill_nas,)


@app.cell
def _(df, fill_nas, var_na_list):
    fill_nas(df, var_na_list)
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell
def _(df):
    # get categorical and numerical variables
    is_object = (df.dtypes == 'object').values
    categorical = df.dtypes[is_object].index.to_list()
    numerical = [x for x in df.dtypes[~is_object].index.to_list() if x != 'converted']
    return categorical, numerical


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 1
    What is the most frequent observation (mode) for the column `industry`?
    """
    )
    return


@app.cell
def _(df):
    df['industry'].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""The most frequent observation for `industry` is `retail` at 203.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 2
    Create the correlation matrix for the numerical features of your dataset. In a correlation matrix, you compute the correlation coefficient between every pair of features.
    """
    )
    return


@app.cell
def _(df):
    numericals = df.dtypes[df.dtypes != 'object'].index.to_list()
    numericals = numericals[:len(numericals) - 1]
    numericals
    return (numericals,)


@app.cell
def _(df, numericals):
    df_corr = df[numericals].corr()
    df_corr
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    What are the two features that have the biggest correlation?

    `annual_income` vs `interaction_count` = 0.027
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Split the data into train/val/test : 60/20/20 using seed 42""")
    return


@app.cell
def _(df):
    # get features and target
    features = [x for x in df.columns if x != 'converted']
    X = df[features].values
    y = df['converted'].values
    return


@app.cell
def _(df):
    from sklearn.model_selection import train_test_split

    # np.random.seed(42)

    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

    y_train = df_train['converted'].values
    y_val = df_val['converted'].values
    y_test = df_test['converted'].values

    del df_train['converted']
    del df_val['converted']
    del df_test['converted']
    return df_test, df_train, df_val, y_train, y_val


@app.cell
def _(df, df_test, df_train, df_val):
    [x.shape for x in [df_train, df_val, df_test, df]]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 3
    - Calculate the mutual information score between `y` and other categorical variables in the dataset. Use the training set only.
    - Round the scores to 2 decimals using round(score, 2).

    """
    )
    return


@app.cell
def _():
    from sklearn.metrics import mutual_info_score
    return (mutual_info_score,)


@app.cell
def _(categorical, df_train, mutual_info_score, y_train):
    mi_scores = []
    for c in categorical:
        score = mutual_info_score(y_train, df_train[c].values)
        mi_scores.append({"feature": c, "score": score})
    return (mi_scores,)


@app.cell
def _(mi_scores, pd):
    pd.DataFrame(mi_scores)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Which of these variables has the biggest mutual information score?

    `lead_source` 
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 4

    - Now let's train a logistic regression.
    - Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
    - Fit the model on the training dataset.
    - To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters: `model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`
    - Calculate the accuracy on the validation dataset and round it to 2 decimal digits.
    """
    )
    return


@app.cell
def _():
    from sklearn.feature_extraction import DictVectorizer
    return (DictVectorizer,)


@app.cell
def _(DictVectorizer, categorical, df_train, df_val, numerical):
    dv = DictVectorizer(sparse=False)

    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    X_val = dv.fit_transform(val_dict)
    return X_train, X_val


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    return LogisticRegression, accuracy_score


@app.cell
def _(LogisticRegression):
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    return (model,)


@app.cell
def _(X_train, model, y_train):
    model.fit(X_train, y_train)
    return


@app.cell
def _(X_val, model):
    y_pred = model.predict_proba(X_val)[:, 1]
    return (y_pred,)


@app.cell
def _(y_pred):
    y_pred_converted = y_pred >= 0.5
    return (y_pred_converted,)


@app.cell
def _(pd, y_pred, y_pred_converted, y_val):
    df_pred = pd.DataFrame()
    df_pred['probability'] = y_pred
    df_pred['prediction'] = y_pred_converted.astype(int)
    df_pred['actual'] = y_val
    return (df_pred,)


@app.cell
def _(df_pred):
    df_pred['correct'] = df_pred.prediction == df_pred.actual
    return


@app.cell
def _(df_pred):
    df_pred['correct'].mean()
    return


@app.cell
def _(X_val, accuracy_score, model, y_val):
    acc_score_all_feaures = accuracy_score(y_val, model.predict(X_val))
    round(acc_score_all_feaures, 3)
    return (acc_score_all_feaures,)


@app.cell
def _(mo):
    mo.md(r"""Closest answer is 0.74""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 5
    - Let's find the least useful feature using the feature elimination technique.
    - Train a model using the same features and parameters as in Q4 (without rounding).
    - Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
    - For each feature, calculate the difference between the original accuracy and the accuracy without the feature.
    """
    )
    return


@app.cell
def _(categorical, numerical):
    all_features = categorical + numerical
    all_features
    return (all_features,)


@app.cell
def _(
    DictVectorizer,
    acc_score_all_feaures,
    accuracy_score,
    all_features,
    df_train,
    df_val,
    model,
    y_train,
    y_val,
):
    def _():
        score_elimination = []
        dv_elim = DictVectorizer(sparse=False)
        for f in all_features:
            # exclude from data frame
            some_features = [x for x in all_features if x != f]
            # print(some_features)

            train_dict = df_train[some_features].to_dict(orient='records')
            val_dict = df_val[some_features].to_dict(orient='records')
            X_train = dv_elim.fit_transform(train_dict)
            X_val = dv_elim.fit_transform(val_dict)
            acc_score = accuracy_score(y_val, model.fit(X_train, y_train).predict(X_val))
            score_elimination.append({"excluded": f, "accuracy": acc_score, "diff": (acc_score - acc_score_all_feaures)})

        return score_elimination
    score_elimination = _()
    return (score_elimination,)


@app.cell
def _(pd, score_elimination):
    pd.DataFrame(score_elimination)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Which of following feature has the smallest difference?

    `industry`
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Question 6

    - Now let's train a regularized logistic regression.
    - Let's try the following values of the parameter C: [0.01, 0.1, 1, 10, 100].
    - Train models using all the features as in Q4.
    - Calculate the accuracy on the validation dataset and round it to 3 decimal digits.
    """
    )
    return


@app.cell
def _(
    DictVectorizer,
    LogisticRegression,
    acc_score_all_feaures,
    accuracy_score,
    all_features,
    df_train,
    df_val,
    y_train,
    y_val,
):
    def _():
        score_elimination = []
        dv_elim = DictVectorizer(sparse=False)
        for c in [0.01, 0.1, 1, 10, 100]:
            model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
            train_dict = df_train[all_features].to_dict(orient='records')
            val_dict = df_val[all_features].to_dict(orient='records')
            X_train = dv_elim.fit_transform(train_dict)
            X_val = dv_elim.fit_transform(val_dict)
            acc_score = accuracy_score(y_val, model.fit(X_train, y_train).predict(X_val))
            score_elimination.append({"C": c, "accuracy": acc_score, "diff": (acc_score - acc_score_all_feaures)})

        return score_elimination
    score_regularized = _()
    return (score_regularized,)


@app.cell
def _(pd, score_regularized):
    pd.DataFrame(score_regularized)
    return


@app.cell
def _(mo):
    mo.md(r"""All have diff of 0. So the smallest C is 0.01""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
