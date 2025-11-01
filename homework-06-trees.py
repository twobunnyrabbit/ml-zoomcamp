import os
import pandas as pd
import numpy as np
import seaborn as sns

from dotenv import load_dotenv

load_dotenv()

# https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2025/06-trees/homework.md


DATASET_PATH = os.getenv("DATASET_PATH")

filename = "car_fuel_efficiency.csv"

df0 = pd.read_csv(DATASET_PATH + "/" + filename)
df0
df0.isna().sum()
df0.info()
df0.columns

target = "fuel_efficiency_mpg"
has_missing_values = (df0.isna().sum() > 0).values.tolist()

missing_val_col_names = df0.loc[:, has_missing_values].columns.to_list()

# for each column with missing value, what is the type?

for c in missing_val_col_names:
    print(df0[c].describe())
    print("\n")


df0["num_doors"].value_counts()
