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
    import pickle
    import requests
    import os
    from pathlib import Path


    # Define the data directory and file path
    DATA_DIR = Path('data')
    CSV_FILE = DATA_DIR / 'horror_movies.csv'

    return CSV_FILE, DATA_DIR, pd, requests


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
def _():
    return


if __name__ == "__main__":
    app.run()
