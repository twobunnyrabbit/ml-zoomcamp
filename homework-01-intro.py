import os
import dotenv
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")

filename = "car_fuel_efficiency.csv"

df = pd.read_csv(DATASET_PATH + "/" + filename)

# Q1. Pandas version
# What's the version of Pandas that you installed?

pd.__version__
# 2.3.2

# Q2. Records count
# How many records are in the dataset?

df.shape
# (9704, 11)

# Q3. Fuel types
# How many fuel types are presented in the dataset?

len(df["fuel_type"].unique())

# 2

# Q4. Missing values
# How many columns in the dataset have missing values?
np.sum(df.isna().sum() > 0)

# 4

# Q5. Max fuel efficiency
# What's the maximum fuel efficiency of cars from Asia?

is_asia = df["origin"] == "Asia"
df[is_asia]["fuel_efficiency_mpg"].max()

# 23.76

# Q6. Median value of horsepower
# Find the median value of horsepower column in the dataset.
df["horsepower"].median()
# 149.0

# Next, calculate the most frequent value of the same horsepower column.
df["horsepower"].mode()
# 152.0

# Use fillna method to fill the missing values in horsepower column with the
# most frequent value from the previous step.

df.fillna({"horsepower": 152.0}, inplace=True)

# Now, calculate the median value of horsepower once again.
df["horsepower"].mode()
# 152.0

# Has it changed?
# Yes, it increased


# Q7. Sum of weights
# Select all the cars from Asia
df_asia = df[is_asia]

# Select only columns vehicle_weight and model_year

df_asia[["vehicle_weight", "model_year"]]

# Select the first 7 values
df_asia[["vehicle_weight", "model_year"]][0:7]

# Get the underlying NumPy array. Let's call it X.
X = df_asia[["vehicle_weight", "model_year"]][0:7].to_numpy()

# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
XTX = X.T.dot(X)

# Invert XTX.
XTX_inv = np.linalg.inv(XTX)

# Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
y = [1100, 1300, 800, 900, 1000, 1100, 1200]

# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.

w = XTX_inv.dot(X.T).dot(y)

# What's the sum of all the elements of the result?

sum(w)
# 0.52
