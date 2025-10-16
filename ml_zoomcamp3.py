import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from dotenv import load_dotenv
    return load_dotenv, mo, np, os, pd, plt


@app.cell
def _(load_dotenv, os, pd):
    load_dotenv()

    DATASET_PATH = os.getenv("DATASET_PATH")

    filename = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(DATASET_PATH + "/" + filename)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return


@app.cell
def _(df):
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    return (categorical_columns,)


@app.cell
def _(categorical_columns):
    categorical_columns
    return


@app.cell
def _(df):
    df.dtypes
    return


@app.cell
def _(df, pd):
    tc = pd.to_numeric(df['totalcharges'], errors='coerce')
    return (tc,)


@app.cell
def _(tc):
    tc.isna().sum()
    return


@app.cell
def _(df, pd):
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    return


@app.cell
def _(df):
    df['totalcharges'] = df['totalcharges'].fillna(0)
    return


@app.cell
def _(df):
    df['totalcharges'].isna().sum()
    return


@app.cell
def _(df):
    df['churn'] = (df['churn'] == "Yes").astype(int)
    return


@app.cell
def _(df):
    df['churn']
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Setting up the validation framework
    - perform the train/validation/test split with Scikit-Learn
    """
    )
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    return (train_test_split,)


@app.cell
def _(df, train_test_split):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return df_full_train, df_test


@app.cell
def _(df_full_train, train_test_split):
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    return df_train, df_val


@app.cell
def _(df_test, df_train, df_val):
    [len(x) for x in [df_train, df_val, df_test]]
    return


@app.cell
def _(df_test, df_train, df_val):
    (y_train, y_val, y_test) = [x.values for x in [df_train['churn'], df_val['churn'], df_test['churn']]]
    return y_test, y_train, y_val


@app.cell
def _(df_test, df_train, df_val):
    del df_train['churn']
    del df_val['churn']
    del df_test['churn']
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## EDA
    - check missing values
    - look at target variable (churn)
    - look at numerical and categorical variables
    """
    )
    return


@app.cell
def _(df_full_train):
    df_full_train
    return


@app.cell
def _(df_full_train):
    # check for missing values
    df_full_train.isna().sum()
    return


@app.cell
def _(df_full_train):
    # check distribution of target variable churn
    df_full_train['churn'].value_counts(normalize=True)
    return


@app.cell
def _(df_full_train, pd):
    df_full_train['churn'].value_counts().pipe(lambda x: pd.DataFrame({'count': x, 'prop': x/x.sum()}))
    return


@app.cell
def _(df_full_train, pd):
    df_full_train['churn'].value_counts().pipe(lambda x: pd.DataFrame({
        'count': x,
        'prop': x/x.sum()
    }))
    return


@app.cell
def _(df_full_train):
    global_churn_rate = df_full_train['churn'].mean()
    round(global_churn_rate,2)
    return (global_churn_rate,)


@app.cell
def _(df_full_train):
    df_full_train.dtypes
    return


@app.cell
def _():
    numerical = ['tenure', 'monthlycharges', 'totalcharges']
    return (numerical,)


@app.cell
def _(df_full_train, numerical):
    exclude = ['customerid', 'tenure', 'churn'] + numerical
    categorical = [x for x in df_full_train.columns if x not in exclude]
    categorical
    return (categorical,)


@app.cell
def _(categorical, df_full_train):
    # examine number of unique categorical variables

    df_full_train[categorical].nunique()
    return


@app.cell
def _(mo):
    mo.md(r"""## Feature importance - churn rate and risk ratio""")
    return


@app.cell
def _(df_full_train, global_churn_rate, pd):
    # Examine churn rate within each group
    # df_full_train[df_full_train['gender'] == 'Female']['churn'].mean()
    # df_full_train.groupby('gender')['churn'].mean().to_dict()
    # df_full_train['gender']

    categorical_vars = ['gender', 'seniorcitizen', 'dependents', 'partner']
    group_bys = []
    # for var in categorical_vars:
    #     group_bys.append(df_full_train.groupby(var)['churn'].mean())

    for var in categorical_vars:
        churn_by_var = df_full_train.groupby(var)['churn'].mean()
        result = pd.DataFrame({
            'churn_rate': churn_by_var,
            'diff': global_churn_rate - churn_by_var,
            'risk_ratio': churn_by_var/global_churn_rate
        })
        group_bys.append(result)
    group_bys
    return


@app.cell
def _():
    from IPython.display import display
    return (display,)


@app.cell
def _(categorical, df_full_train, display, global_churn_rate):

    for c in categorical:
        df_group = df_full_train.groupby(c)['churn'].agg(['mean', 'count'])
        df_group['diff'] = df_group['mean'] - global_churn_rate
        df_group['risk'] = df_group['mean'] / global_churn_rate
        display(df_group)
    return


@app.cell
def _(mo):
    mo.md(r"""## Feature importance - mutual information""")
    return


@app.cell
def _():
    from sklearn.metrics import mutual_info_score
    return (mutual_info_score,)


@app.cell
def _(df_full_train, mutual_info_score):
    mutual_info_score(df_full_train['churn'], df_full_train['contract'])
    return


@app.cell
def _(categorical, df_full_train, mutual_info_score):
    mutual_info_list = [{'feature': x, 'score': mutual_info_score(df_full_train[x], df_full_train['churn'])} for x in categorical]
    mutual_info_list
    return (mutual_info_list,)


@app.cell
def _(mutual_info_list, pd):
    df_mutual_info = pd.DataFrame(mutual_info_list)
    df_mutual_info
    return


@app.cell
def _(categorical, df_full_train, mutual_info_score):
    df_mutual_info_2 = df_full_train[categorical].apply(lambda x: mutual_info_score(x, df_full_train['churn']))
    return (df_mutual_info_2,)


@app.cell
def _(df_mutual_info_2):
    df_mutual_info_2.sort_values(ascending=False)
    return


@app.cell
def _(mo):
    mo.md(r"""## Feature importance - correlation""")
    return


@app.cell
def _(df_full_train, numerical):
    df_full_train[numerical].corrwith(df_full_train['churn'])
    return


@app.cell
def _(df_full_train):
    df_full_train[df_full_train['tenure'] <= 2].churn.mean()
    return


@app.cell
def _(df_full_train):
    df_full_train[(df_full_train['tenure'] > 2) & (df_full_train['tenure'] <= 12)].churn.mean()
    return


@app.cell
def _(df_full_train):
    df_full_train[df_full_train['tenure'] > 12].churn.mean()
    return


@app.cell
def _(df_full_train):
    df_full_train.groupby('tenure')['churn'].agg('mean').plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## One-hot encoding
    Using Scikit-Leanr to encode categorical features
    """
    )
    return


@app.cell
def _():
    from sklearn.feature_extraction import DictVectorizer
    return (DictVectorizer,)


@app.cell
def _(df_train):
    dicts = df_train[['gender', 'contract', 'tenure']].iloc[:100].to_dict(orient='records')
    dicts
    return (dicts,)


@app.cell
def _(DictVectorizer):
    dv = DictVectorizer(sparse=False)
    return (dv,)


@app.cell
def _(dicts, dv):
    dv.fit(dicts)
    return


@app.cell
def _(dicts, dv):
    dv.transform(dicts)
    return


@app.cell
def _(dv):
    dv.get_feature_names_out()
    return


@app.cell
def _(categorical, df_train, numerical):
    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    train_dict[:10]
    return (train_dict,)


@app.cell
def _(DictVectorizer, train_dict):
    dv_train = DictVectorizer(sparse=False)
    dv_train.fit(train_dict)
    return (dv_train,)


@app.cell
def _(dv_train):
    dv_train.get_feature_names_out()
    return


@app.cell
def _(dv_train, train_dict):
    X_train = dv_train.transform(train_dict)
    return (X_train,)


@app.cell
def _(X_train):
    X_train
    return


@app.cell
def _(DictVectorizer, categorical, df_val, numerical):
    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    dv_val = DictVectorizer(sparse=False)
    X_val = dv_val.fit(val_dict).transform(val_dict)
    return (X_val,)


@app.cell
def _(DictVectorizer, categorical, df_test, numerical):
    test_dict = df_test[categorical + numerical].to_dict(orient='records')
    dv_test = DictVectorizer(sparse=False)
    X_test = dv_test.fit(test_dict).transform(test_dict)
    return (X_test,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Logistic regression

    A model that takes a set of features and outputs a probability of that record belonging to the positive class.
    """
    )
    return


@app.cell
def _(np):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    return (sigmoid,)


@app.cell
def _(np):
    z = np.linspace(-7, 7, 51)
    z
    return (z,)


@app.cell
def _(sigmoid, z):
    sigmoid(z)
    return


@app.cell
def _(plt, sigmoid, z):
    plt.plot(z, sigmoid(z))
    return


@app.cell
def _(mo):
    mo.md(r"""## Training logistic regression with scikit-learn""")
    return


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression
    return (LogisticRegression,)


@app.cell
def _(LogisticRegression, X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(model):
    model.coef_[0].round(3)
    return


@app.cell
def _(model):
    model.intercept_[0]
    return


@app.cell
def _(X_train, model):
    model.predict(X_train)
    return


@app.cell
def _(X_train, model):
    model.predict_proba(X_train)
    return


@app.cell
def _(X_val, model):
    y_pred_val = model.predict_proba(X_val)[:, 1]
    return (y_pred_val,)


@app.cell
def _(y_pred_val):
    churn_decision = (y_pred_val >= 0.5)
    churn_decision
    return (churn_decision,)


@app.cell
def _(churn_decision, df_val):
    df_val[churn_decision]
    return


@app.cell
def _(y_val):
    y_val
    return


@app.cell
def _(churn_decision):
    churn_decision.astype(int)
    return


@app.cell
def _(churn_decision, y_val):
    (y_val == churn_decision).mean()
    return


@app.cell
def _(churn_decision, pd, y_pred_val, y_val):
    df_pred = pd.DataFrame()
    df_pred['probablity'] = y_pred_val
    df_pred['prediction'] = churn_decision.astype(int)
    df_pred['actual'] = y_val
    df_pred['correct'] = df_pred['prediction'] == df_pred['actual']
    return (df_pred,)


@app.cell
def _(df_pred):
    df_pred
    return


@app.cell
def _(df_pred):
    df_pred['correct'].mean()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Model interpretation
    - Look at the coefficients
    - Train a smaller model with fewer features
    """
    )
    return


@app.cell
def _(dv_train, model):
    feature_weights = dict(zip(dv_train.get_feature_names_out(), model.coef_[0].round(3)))
    feature_weights
    return


@app.cell
def _(df_train, df_val):
    small = ['contract', 'tenure', 'monthlycharges']
    dicts_train_small = df_train[small].to_dict(orient='records')
    dicts_val_small = df_val[small].to_dict(orient='records')
    return (dicts_train_small,)


@app.cell
def _(DictVectorizer, dicts_train_small):
    dv_small = DictVectorizer(sparse=False)
    dv_small.fit(dicts_train_small)
    return (dv_small,)


@app.cell
def _(dv_small):
    dv_small.get_feature_names_out()
    return


@app.cell
def _(dicts_train_small, dv_small):
    X_train_small = dv_small.transform(dicts_train_small)
    return (X_train_small,)


@app.cell
def _(LogisticRegression, X_train_small, y_train):
    model_small = LogisticRegression()
    model_small.fit(X_train_small, y_train)
    return (model_small,)


@app.cell
def _(model_small):
    w0 = model_small.intercept_[0]
    w0
    return


@app.cell
def _(model_small):
    w = model_small.coef_[0]
    w.round(3)
    w
    return (w,)


@app.cell
def _(dv_small, w):
    dict(zip(dv_small.get_feature_names_out(), w.round(3)))
    return


@app.cell
def _(mo):
    mo.md(r"""## 3.12 Using the model""")
    return


@app.cell
def _(categorical, df_full_train, numerical):
    dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')
    return (dicts_full_train,)


@app.cell
def _(DictVectorizer, df_full_train, dicts_full_train):
    dv_full_train = DictVectorizer(sparse=False)
    X_full_train = dv_full_train.fit_transform(dicts_full_train)
    y_full_train = df_full_train['churn'].values
    return X_full_train, dv_full_train, y_full_train


@app.cell
def _(LogisticRegression, X_full_train, y_full_train):
    model_full_train = LogisticRegression()
    model_full_train.fit(X_full_train, y_full_train)
    return


@app.cell
def _(categorical, df_test, numerical):
    dicts_test = df_test[categorical + numerical].to_dict(orient='records')
    return (dicts_test,)


@app.cell
def _(dicts_test, dv_full_train):
    X_test_full_train = dv_full_train.transform(dicts_test)
    return


@app.cell
def _(X_test, model):
    y_pred_full_train = model.predict_proba(X_test)[:, 1]
    return (y_pred_full_train,)


@app.cell
def _(y_pred_full_train):
    churn_decision_full_train = (y_pred_full_train >= 0.5)
    return (churn_decision_full_train,)


@app.cell
def _(churn_decision_full_train, y_test):
    (churn_decision_full_train == y_test).mean()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
