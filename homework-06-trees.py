import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv

load_dotenv()

# https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2025/06-trees/homework.md


DATASET_PATH = os.getenv("DATASET_PATH")

filename = "car_fuel_efficiency.csv"

df = pd.read_csv(DATASET_PATH + "/" + filename)

target = "fuel_efficiency_mpg"

# Fill missing values with zeros.
df.fillna(0, inplace=True)

# Do train/validation/test split with 60%/20%/20% distribution. 
# Use the train_test_split function and set the random_state parameter to 1.
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

print(f"Train shape: {df_train.shape}")
print(f"Val shape: {df_val.shape}")
print(f"Test shape: {df_test.shape}")

# get the target values for each dataset
y_train, y_val, y_test = [df['fuel_efficiency_mpg'].values for df in (df_train, df_val, df_test)]
y_full_train = df_full_train['fuel_efficiency_mpg']

# remove the target variables from the dataframes
del df_full_train['fuel_efficiency_mpg']
del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']

# Use DictVectorizer(sparse=True) to turn the dataframes into matrices.


dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(df_full_train.to_dict(orient='records'))
X_train = dv.fit_transform(df_train.to_dict(orient='records'))
X_val = dv.fit_transform(df_val.to_dict(orient='records'))
X_test = dv.fit_transform(df_test.to_dict(orient='records'))

"""
Question 1
Let's train a decision tree regressor to predict the fuel_efficiency_mpg variable.

Train a model with max_depth=1.
Which feature is used for splitting the data?

'vehicle_weight'
'model_year'
'origin'
'fuel_type'
"""

from sklearn.tree import DecisionTreeRegressor

# Create and train the model
model = DecisionTreeRegressor(max_depth=1, random_state=1)
model.fit(X_train, y_train)

features_name = dv.get_feature_names_out().tolist()
features_importance = (model.feature_importances_ == 1).tolist()

for item in zip(features_importance, features_name):
    if item[0] == True:
        print(f"{item[0]}: {item[1]}")

"""
Answer: vehicle_weight
"""

###################

"""
Question 2
Train a random forest regressor with these parameters:

n_estimators=10
random_state=1
n_jobs=-1 (optional - to make training faster)
What's the RMSE of this model on the validation data?

0.045
0.45
4.5
45.0
"""



# Create and train the model
model_rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
model_rf.fit(X_train, y_train)



# Make predictions on the test set
y_pred_rf = model_rf.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"RMSE: {rmse}")

"""
Answer: 0.45 (0.4494370104563073)

"""

###################

"""
Question 3

Now let's experiment with the n_estimators parameter

Try different values of this parameter from 10 to 200 with step 10.
Set random_state to 1.
Evaluate the model on the validation dataset.
After which value of n_estimators does RMSE stop improving? Consider 3 decimal places for calculating the answer.

10
25
80
200
If it doesn't stop improving, use the latest iteration number in your answer.
"""

def get_rmse(n_estimators, max_depth=None):
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

results = []

for n in range(10, 201, 10):
    rmse = get_rmse(n)
    results.append({'n_est': n, 'rmse': rmse})

best_n = min(results, key=lambda x: x['rmse'])['n_est']
print(best_n)


df_results = pd.DataFrame(results)
df_results['rmse_2'] = df_results['rmse'].round(3)
df_results['diff'] = df_results['rmse_2'].diff()
df_results.sort_values(by='rmse_2')

"""
Answer: 80
"""

###################

"""
Question 4
Let's select the best max_depth:

Try different values of max_depth: [10, 15, 20, 25]
For each of these values,
try different values of n_estimators from 10 till 200 (with step 10)
calculate the mean RMSE
Fix the random seed: random_state=1
What's the best max_depth, using the mean RMSE?

10
15
20
25

"""

results_2 = []
for max_depth in [10, 15, 20, 25]:
    # n_estimators_results = []
    for n_estimators in range(10, 201, 10):
        rmse = get_rmse(n_estimators=n_estimators, max_depth=max_depth)
        # n_estimators_results.append({'n_est': n_estimators, 'rmse': rmse})
        results_2.append({'max_depth': max_depth, 'n_est': n_estimators, 'rmse': rmse})
        
df_results_2 = pd.DataFrame(results_2)
df_results_2



plt.figure(figsize=(10, 6))
for max_depth in df_results_2['max_depth'].unique():
    subset = df_results_2[df_results_2['max_depth'] == max_depth]
    plt.plot(subset['n_est'], subset['rmse'], label=f'max_depth={max_depth}')
plt.xlabel('n_est')
plt.ylabel('rmse')
plt.legend()
plt.grid(True)
plt.show()

"""
Answer: max_depth = 10 has the best overall rmse
"""

##########################

"""
Question 5
We can extract feature importance information from tree-based models.

At each step of the decision tree learning algorithm, it finds the best split. 
When doing it, we can calculate "gain" - the reduction in impurity before and after the split. 
This gain is quite useful in understanding what are the important features for tree-based models.

In Scikit-Learn, tree-based models contain this information in the feature_importances_ field.

For this homework question, we'll find the most important feature:

Train the model with these parameters:
n_estimators=10,
max_depth=20,
random_state=1,
n_jobs=-1 (optional)
Get the feature importance information from this model
What's the most important feature (among these 4)?

vehicle_weight
horsepower
acceleration
engine_displacement
"""

def feature_importance():
    model = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)
    return model.feature_importances_.tolist()

f_importance_values = feature_importance()

f_importance_names = dv.get_feature_names_out().tolist()

df_f_importance = pd.DataFrame({'name':f_importance_names, 'values': f_importance_values})
df_f_importance.sort_values(by='values', ascending=False)

"""
Answer: vechicle_weight
"""

##########################


"""
Question 6
Now let's train an XGBoost model! For this question, we'll tune the eta parameter:

Install XGBoost
Create DMatrix for train and validation
Create a watchlist
Train a model with these parameters for 100 rounds:
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
Now change eta from 0.3 to 0.1.

Which eta leads to the best RMSE score on the validation dataset?

0.3
0.1
Both give equal value
"""

features = dv.get_feature_names_out().tolist()

# Create DMatrix for training data
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

# Create DMatrix for validation data
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)


def xgb_rmse(eta):
    xgb_params = {
        'eta': eta, 
        'max_depth': 6,
        'min_child_weight': 1,
        
        'objective': 'reg:squarederror',
        'nthread': 8,
        
        'seed': 1,
        'verbosity': 1,
    }

    watchlist = [(dtrain, "train"), (dval, "val")]
    eval_results = {}

    model_xgb = xgb.train(
        xgb_params, dtrain, 
        evals=watchlist, 
        evals_result=eval_results,
        verbose_eval=5, num_boost_round=100
    )
    return eval_results
    # y_pred = model_xgb.predict(X_val)
    # rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    # return rmse

eta_list = [{'eta': eta, 'evals': xgb_rmse(eta)} for eta in [0.3, 1]]

eta_list[0]['eta']
eta_list[0]['evals'].keys()

n = len(eta_list[0]['evals']['val']['rmse'])
"""
eta
    evals ->
            train -> rmse
            val -> rmse
"""

def process_results(eta_list):
    eta = eta_list['eta']
    train_rmse = eta_list['evals']['train']['rmse']
    val_rmse = eta_list['evals']['val']['rmse'] 
    return {'eta': eta, 'train_rmse': train_rmse, 'val_rmse': val_rmse}


eta_results = [process_results(eta) for eta in eta_list]

df_etas = [pd.DataFrame(e) for e in eta_results]
df_etas[0]['n_est'] = np.arange(1, 101)
df_etas[1]['n_est'] = np.arange(1, 101)

df_etas_final = pd.concat([df_etas[0], df_etas[1]])
df_etas_final.shape

plt.figure(figsize=(10, 6))
for eta in df_etas_final['eta'].unique():
    subset = df_etas_final[df_etas_final['eta'] == eta]
    plt.plot(subset['n_est'], subset['val_rmse'], label=f'eta={eta}')
plt.xlabel('n_est')
plt.ylabel('val_rmse')
plt.legend()
plt.grid(True)
plt.title('RMSE vs n_est grouped by eta')
plt.show()

"""
Answer: eta=0.3 gives the best rmse for validation set
"""