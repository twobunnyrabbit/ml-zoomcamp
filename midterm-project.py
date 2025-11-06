import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    Choosing this dataset from [TidyTuesday](https://github.com/rfordatascience/tidytuesday/blob/main/data/2022/2022-11-01/readme.md)

    dataset link:
    `'https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2022/2022-11-01/horror_movies.csv'`
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    # import pickle
    import requests
    import os
    from pathlib import Path


    # Define the data directory and file path
    DATA_DIR = Path('data')
    CSV_FILE = DATA_DIR / 'horror_movies.csv'
    return CSV_FILE, DATA_DIR, np, pd, requests


@app.cell
def _(CSV_FILE, DATA_DIR, pd, requests):
    def load_horror_movies_data():
        """
        Load horror movies data from local CSV file.
        If the file doesn't exist locally, download it first.

        Returns:
            pd.DataFrame: DataFrame containing the horror movies data
        """
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(exist_ok=True)
        csv_location = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2022/2022-11-01/horror_movies.csv'

        # Check if file exists locally
        if not CSV_FILE.exists():
            print(f"File {CSV_FILE} not found locally. Downloading...")
            download_horror_movies_data(csv_location)

        # Load the data
        print(f"Loading data from {CSV_FILE}")
        df = pd.read_csv(CSV_FILE)
        return df

    def download_horror_movies_data(csv_location):
        """
        Download horror movies data from remote source.
        This is a placeholder function - you'll need to replace the URL
        with the actual source of your horror movies data.
        """
        # TODO: Replace this with the actual URL where you download the data
        url = csv_location # Placeholder URL

        try:
            # Download the data
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Save to local file
            with open(CSV_FILE, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded and saved data to {CSV_FILE}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            raise
        except Exception as e:
            print(f"Error saving data: {e}")
            raise
    return (load_horror_movies_data,)


@app.cell
def _(load_horror_movies_data):
    df = load_horror_movies_data()
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    # columns to keep
    cols_to_keep = ['original_language', 'release_date', 'popularity', 'vote_count', 'vote_average', 'revenue', 'runtime', 'status', 'adult', 'genre_names']
    return (cols_to_keep,)


@app.cell
def _(cols_to_keep, df):
    df_2 = df[cols_to_keep]
    return (df_2,)


@app.cell
def _(df_2):
    # check each variable
    df_2.iloc[:, 0].value_counts()
    return


@app.cell
def _(df_2):
    # limit language to en
    df_3 = df_2[df_2['original_language'] == 'en']
    return (df_3,)


@app.cell
def _(df_3):
    df_3.shape
    return


@app.cell
def _(df_3):
    df_3
    return


@app.cell
def _(df_3):
    # feature engineer genre_names
    df_3['genre_names']
    return


@app.cell
def _(df_3):
    genre_set = set()
    genre_raw_list = df_3['genre_names'].to_list()
    return genre_raw_list, genre_set


@app.function
def process_list(x):
    """
    x is a string e.g. Horror, or Drama, Horror
    check if comma is present, if it is split it and convert to lower case
    if not, convert to lower case
    return is a list of string(s)
    """
    if "," in x:
        # print("has comma")
        return [y.strip().lower() for y in x.split(',')]
    else:
        # print("no comma")
        return [x.lower()]


@app.cell
def _(genre_set):
    def add_to_set(x):
        """
        x is a list of string
        loop through this to add it to pre-defined set
        """
        for i in range(len(x)):
            genre_set.add(x[i])
    return (add_to_set,)


@app.cell
def _(add_to_set, genre_raw_list):
    # loop through original list and create unique list
    for i in range(len(genre_raw_list)):
        processed = process_list(genre_raw_list[i])
        add_to_set(processed)
    return


@app.cell
def _(genre_set):
    genre_list = [x for x in genre_set]
    return (genre_list,)


@app.cell
def _(genre_list):
    genre_list
    return


@app.cell
def _(df_3, genre_set, np):
    # create a zero-based numpy array with row(df_3) x col(genre_set)
    mtx_row = df_3.shape[0]
    mtx_col = len(genre_set)
    genre_mtx = np.zeros((mtx_row, mtx_col))
    return genre_mtx, mtx_row


@app.cell
def _(genre_mtx):
    genre_mtx
    return


@app.cell
def _(genre_list, genre_mtx):
    def update_mtx(row, genre):
        idx = genre_list.index(genre)
        genre_mtx[row, idx] = 1

    return (update_mtx,)


@app.cell
def _(genre_raw_list, mtx_row, update_mtx):
    for r in range(mtx_row):
        raw_genre = [x.strip().lower() for x in genre_raw_list[r].split(",")]
        # print(raw_genre)
        for iii in range(len(raw_genre)):
            update_mtx(r, raw_genre[iii])
    return


@app.cell
def _(genre_list, genre_mtx, pd):
    df_genre = pd.DataFrame(genre_mtx, columns=genre_list)
    return (df_genre,)


@app.cell
def _(df_3, df_genre, pd):
    df_4 = pd.concat([df_3, df_genre], axis=1)
    return (df_4,)


@app.cell
def _(df_4):
    df_4
    return


@app.cell
def _(df_4, pd):
    # Convert string to datetime and calculate days since today
    df_4['days_since'] = pd.to_datetime(pd.Timestamp.now().floor('D')) - pd.to_datetime(df_4['release_date'])
    df_4['days_since'] = df_4['days_since'].dt.days
    return


@app.cell
def _(df_4):
    df_4[['release_date', 'days_since']]
    return


@app.cell
def _(df_4):
    # drop release_date, genre_names and adult
    # drop original_language column
    del df_4['original_language']
    del df_4['release_date']
    del df_4['genre_names']
    del df_4['adult']
    return


@app.cell
def _(df_4):
    df_4
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return (plt,)


@app.cell
def _(df, plt):
    plt.figure()
    df['popularity'].plot.kde()
    plt.show()
    return


@app.cell
def _(df):
    df['popularity'].describe()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
