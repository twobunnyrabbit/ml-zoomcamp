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

    from dotenv import load_dotenv

    load_dotenv()

    DATASET_PATH = os.getenv("DATASET_PATH")

    filename = "car_fuel_efficiency.csv"


    return DATASET_PATH, filename, mo, np, pd, sns


@app.cell
def _(DATASET_PATH, filename, pd):
    df0 = pd.read_csv(DATASET_PATH + "/" + filename)
    return (df0,)


@app.cell
def _(df0):
    variables = ['engine_displacement',
    'horsepower',
    'vehicle_weight',
    'model_year',
    'fuel_efficiency_mpg']
    df = df0[variables]
    return df, variables


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""EDA on fuel efficiency""")
    return


@app.cell
def _(df, sns):
    sns.histplot(df['fuel_efficiency_mpg'])
    return


@app.cell
def _(mo):
    mo.md(r"""The distribution of fuel_efficiency_mpg has a roughyl bell-shaped distribution.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Question 1:**

    There's one column with missing values. What is it?
    """
    )
    return


@app.cell
def _(df, variables):
    df[variables].isna().sum()
    return


@app.cell
def _(mo):
    mo.md(r"""`horsepower` has 708 missing values.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Question 2**

    What's the median (50% percentile) for variable 'horsepower'?
    """
    )
    return


@app.cell
def _(df):
    df['horsepower'].median()
    return


@app.cell
def _(mo):
    mo.md(r"""The median of `horsepower` is 149.0.""")
    return


@app.cell
def _(df, np):
    # prepare and split the dataset
    np.random.seed(42)

    n = len(df)

    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - (n_val + n_test)

    idx = np.arange(n)
    np.random.shuffle(idx)

    return idx, n_train, n_val


@app.cell
def _(df, idx, n_train, n_val):
    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train+n_val:]]

    return df_test, df_train, df_val


@app.cell
def _(df_test, df_train, df_val):
    (y_train, y_val, y_test) = [x.pop('fuel_efficiency_mpg').to_numpy() for x in [df_train, df_val, df_test]]
    # (y_train, y_val, y_test) = [x.to_numpy() x for x in [y_train, y_val, y_test]]
    return y_train, y_val


@app.cell
def _(df_test, df_train, df_val):
    X_train = df_train.values
    X_val = df_val.values
    X_test = df_test.values
    return


@app.cell
def _(np):
    def train_linear_regression(X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w_full = XTX_inv.dot(X.T).dot(y)

        return w_full[0], w_full[1:]
    return (train_linear_regression,)


@app.cell
def _(np):
    def train_linear_regression_reg(X, y, r=0.001):
            ones = np.ones(X.shape[0])
            X = np.column_stack([ones, X])

            XTX = X.T.dot(X)
            XTX = XTX + r * np.eye(XTX.shape[0])
            XTX_inv = np.linalg.inv(XTX)
            w_full = XTX_inv.dot(X.T).dot(y)

            return w_full[0], w_full[1:]
    return (train_linear_regression_reg,)


@app.cell
def _(np):
    def rmse(y, y_pred):
        error = y - y_pred
        se = error ** 2
        mse = se.mean()
        return np.sqrt(mse)
    return (rmse,)


@app.cell
def _(mo):
    mo.md(
        r"""
    **Question 3**

    - We need to deal with missing values for the column from Q1.
    - We have two options: fill it with 0 or with the mean of this variable.
    - Try both options. For each, train a linear regression model without regularization using the code from the lessons.
    - For computing the mean, use the training only!
    - Use the validation dataset to evaluate the models and compare the RMSE of each option.
    - Round the RMSE scores to 2 decimal digits using round(score, 2)
    - Which option gives better RMSE?
    """
    )
    return


@app.cell
def _(df_train):
    df_train_fill_0 = df_train.copy()
    df_train_fill_mean = df_train.copy()
    return df_train_fill_0, df_train_fill_mean


@app.cell
def _(df_train_fill_0, df_train_fill_mean):
    df_train_fill_0.fillna(0, inplace=True)
    df_train_fill_mean.fillna(df_train_fill_mean['horsepower'].mean(), inplace=True)
    return


@app.cell
def _(df_train_fill_0):
    df_train_fill_0.isna().sum()
    return


@app.cell
def _(df_train_fill_0):
    df_train_fill_0
    return


@app.cell
def _(df_train_fill_0, train_linear_regression, y_train):
    (w0_fill_0, w_fill_0) = train_linear_regression(df_train_fill_0.values, y_train)
    return w0_fill_0, w_fill_0


@app.cell
def _(w0_fill_0, w_fill_0):
    (w0_fill_0, w_fill_0)
    return


@app.cell
def _(df_val, rmse, w0_fill_0, w_fill_0, y_val):
    df_val_fill_0 = df_val.copy().fillna(0)  # Apply same zero-filling to validation set
    X_val_0 = df_val_fill_0.values  # Get feature matrix (1940 x 4)
    y_pred_0 = w0_fill_0 + X_val_0 @ w_fill_0  # Matrix multiplication

    # Calculate RMSE for zero-filling
    rmse_0 = rmse(y_val, y_pred_0)
    print(f"RMSE with zero-filling: {round(rmse_0, 2)}")
    return


@app.cell
def _(df_train, df_val, rmse, w0_fill_0, w_fill_0, y_val):
    df_val_fill_mean = df_val.copy().fillna(df_train['horsepower'].mean())  # Use training mean
    X_val_mean = df_val_fill_mean.values  # Get feature matrix (1940 x 4)
    y_pred_mean = w0_fill_0 + X_val_mean @ w_fill_0  # Matrix multiplication

    # Calculate RMSE for mean-filling
    rmse_mean = rmse(y_val, y_pred_mean)
    print(f"RMSE with mean-filling: {round(rmse_mean, 2)}")
    return


@app.cell
def _(mo):
    mo.md(r"""Mean has lower RMSE compared to filling in with 0s.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Question 4**

    - Now let's train a regularized linear regression.
    - For this question, fill the NAs with 0.
    - Try different values of r from this list: [0, 0.01, 0.1, 1, 5, 10, 100].
    - Use RMSE to evaluate the model on the validation dataset. 
    - Round the RMSE scores to 2 decimal digits.
    - Which r gives the best RMSE?
    """
    )
    return


@app.cell
def _(
    df_train_fill_0,
    df_val,
    rmse,
    train_linear_regression_reg,
    y_train,
    y_val,
):
    def _():
        scores = []
        df_val_fill_0 = df_val.copy().fillna(0).values 
        for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
            w0, w = train_linear_regression_reg(df_train_fill_0.values, y_train, r=r)
            # print(f'w0: {w0}, w: {w}')
            y_pred = w0 + df_val_fill_0.dot(w)
            score = rmse(y_val, y_pred)
            scores.append({'r': r, 'score': score, 'w0': w0})
            # print(score)
        return scores

    scores = _()
    print(scores)
    return (scores,)


@app.cell
def _(pd, scores):
    df_q3 = pd.DataFrame(scores)
    df_q3['score'] = df_q3['score'].round(2)
    df_q3
    return


@app.cell
def _(mo):
    mo.md(r"""r=0 gives the best RMSE - 0.52""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Question 5**

    - We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
    - Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
    - For each seed, do the train/validation/test split with 60%/20%/20% distribution.
    - Fill the missing values with 0 and train a model without regularization.
    - For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
    - What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
    - Round the result to 3 decimal digits (round(std, 3))
    """
    )
    return


@app.cell
def _(np):
    def split_with_seed(df, seed):
        # prepare and split the dataset
        np.random.seed(seed)
    
        n = len(df)
    
        n_val = int(n * 0.2)
        n_test = int(n * 0.2)
        n_train = n - (n_val + n_test)
    
        idx = np.arange(n)
    
        # shuffle the index
        np.random.shuffle(idx)
    
        # split the dataframes
        df0 = df.copy()
        df0.fillna({'horsepower': 0}, inplace=True)
        df_train = df0.iloc[idx[:n_train]]
        df_val = df0.iloc[idx[n_train:n_train+n_val]]
        df_test = df0.iloc[idx[n_train+n_val:]]

        (y_train, y_val, y_test) = [x.pop('fuel_efficiency_mpg').to_numpy() for x in [df_train, df_val, df_test]]
        X_train = df_train.values
        X_val = df_val.values
        X_test = df_test.values
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

    return (split_with_seed,)


@app.cell
def _(df, split_with_seed):
    seed_1 = split_with_seed(df, 0)
    return (seed_1,)


@app.cell
def _(seed_1, train_linear_regression):
    w0, w = train_linear_regression(seed_1['X_train'], seed_1['y_train'])
    return w, w0


@app.cell
def _(w, w0):
    w0, w
    return


@app.cell
def _(df, rmse, split_with_seed, train_linear_regression):
    def _():
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        score_seeds = []
        for i in range(len(seeds)):
            seed_dict = split_with_seed(df, seeds[i])
            w0, w = train_linear_regression(seed_dict['X_train'], seed_dict['y_train'])
            y_seed = w0 + seed_dict['X_val'].dot(w)
            score = rmse(seed_dict['y_val'], y_seed)
            score_seeds.append({'seed': i, 'rmse': score})

        return score_seeds


    score_seeds = _()
    return (score_seeds,)


@app.cell
def _(pd, score_seeds):
    df_score_seeds = pd.DataFrame(score_seeds)
    return (df_score_seeds,)


@app.cell
def _(df_score_seeds):
    df_score_seeds['rmse'].std()
    return


@app.cell
def _(mo):
    mo.md(r"""Standard deviation is 0.0074""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Question 6**

    Split the dataset like previously, use seed 9.
    Combine train and validation datasets.
    Fill the missing values with 0 and train a model with r=0.001.
    What's the RMSE on the test dataset?
    """
    )
    return


@app.cell
def _(df, split_with_seed):
    seed_9 = split_with_seed(df, 9)
    return (seed_9,)


@app.cell
def _(seed_9):
    seed_9
    return


@app.cell
def _():
    return


@app.cell
def _(np, seed_9, train_linear_regression_reg):
    X_train_9 = np.vstack([seed_9['X_train'], seed_9['X_val']])
    y_train_9 = np.concat([seed_9['y_train'], seed_9['y_val']])
    w0_9, w_9 = train_linear_regression_reg(X_train_9, y_train_9, r=0.001)

    return w0_9, w_9


@app.cell
def _(rmse, seed_9, w0_9, w_9):
    y_pred_9 = w0_9 + seed_9['X_test'].dot(w_9)
    rmse(y_pred_9, seed_9['y_test'])
    return


@app.cell
def _(mo):
    mo.md(r"""RMSE = 0.516""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
