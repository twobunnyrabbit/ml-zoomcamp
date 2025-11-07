import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""## Revision for zoomcamp3.py""")
    return


@app.cell
def _():
    import marimo as mo
    import os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from dotenv import load_dotenv

    return load_dotenv, mo, os, pd


@app.cell
def _(load_dotenv, os, pd):
    load_dotenv()

    DATASET_PATH = os.getenv("DATASET_PATH")

    filename = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(DATASET_PATH + "/" + filename)
    return (df,)


@app.cell
def _(df):
    # convert column names to lower case
    df.columns = df.columns.str.lower()
    return


@app.cell
def _(df, pd):
    # convert totalcharges to numeric
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    return


@app.cell
def _(df):
    # convert the target variable churn to numeric
    # df['churn'] = (df['churn'] == "Yes").astype('int')
    df["churn"]
    return


@app.cell
def _(df):
    # convert seniorcitizen to categorical variable
    df["seniorcitizen"] = df["seniorcitizen"].map({0: "No", 1: "Yes"})
    return


@app.cell
def _(df):
    # get all the categorical variables
    is_object = df.dtypes == "object"
    categoricals = df.dtypes[df.dtypes == "object"].index.to_list()
    # remove the id column and the target variable churn
    categoricals = [x for x in categoricals if x not in ["customerid", "churn"]]
    categoricals
    return categoricals, is_object


@app.cell
def _(df, is_object):
    # get numerical
    numericals = df.dtypes[~is_object].index.to_list()
    # numericals = [x for x in numericals if x != 'churn']
    numericals
    return (numericals,)


@app.cell
def _(categoricals, df, numericals):
    missing_values = df[categoricals + numericals].isna().sum()
    missing_values[missing_values > 0]
    return


@app.cell
def _(df):
    # examin totalcharges
    total_charges_na = df["totalcharges"].isna()
    df.loc[total_charges_na, :]
    return


@app.cell
def _(df):
    df[["tenure", "contract", "monthlycharges", "totalcharges"]]
    return


@app.cell
def _(mo):
    mo.md(
        r"""For missing `totalcharges` this can be approximated by multiplying `tenure` by `monthlycharges`"""
    )
    return


@app.cell
def _(df):
    df["totalcharges"] = df["totalcharges"].fillna(df["tenure"] * df["monthlycharges"])
    return


@app.cell
def _(df):
    # What is the churn rate?
    df["churn"].mean()
    return


@app.cell
def _(mo):
    mo.md(r"""## Set up validation framework""")
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split

    return (train_test_split,)


@app.cell
def _(categoricals, df, numericals, train_test_split):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_full_train, df_test = [
        x.reset_index(drop=True) for x in [df_full_train, df_test]
    ]

    # keep only relevant columns
    df_full_train = df_full_train[categoricals + numericals]
    df_test = df_test[categoricals + numericals]
    return df_full_train, df_test


@app.cell
def _(df_full_train, train_test_split):
    # define training and validation set
    # final splits is 60/40/40
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    return df_train, df_val


@app.function
# Alternative method using globals()
def get_variable_name_simple(variable):
    """Simple version using globals()"""
    return [name for name, var in globals().items() if var is variable][0]


@app.cell
def _(df, df_full_train, df_test, df_train, df_val):
    {
        get_variable_name_simple(x): x.shape
        for x in [df, df_full_train, df_train, df_val, df_test]
    }
    return


@app.cell
def _(df_test, df_train, df_val):
    # set up target values
    y_train, y_val, y_test = [x["churn"].values for x in [df_train, df_val, df_test]]
    # remove target variable for each dataframe
    # del df_full_train['churn']
    del df_train["churn"]
    del df_val["churn"]
    del df_test["churn"]
    return


@app.cell
def _(df_full_train):
    churn_global = df_full_train["churn"].mean()
    churn_global
    return


@app.cell
def _(mo):
    mo.md(r"""## Feature importance""")
    return


@app.cell
def _(categoricals):
    # What is the churn rate within each categorical variables?
    categoricals
    return


app._unparsable_cell(
    r"""
    feature_importance = []
    for c in categoricals
    df_full_train.groupby('gender')['churn'].mean().to_dict()
    """,
    name="_",
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
