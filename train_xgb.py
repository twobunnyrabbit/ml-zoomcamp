import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")
DATA_FILENAME = "insurance.csv"
MODEL_FILENAME = "xgboost_insurance_model.bin"
TARGET_COLUMN = 'charges_log'
FEATURES = ['age', 'sex', 'bmi', 'children', 'smoker', 'region'] # 'region' was excluded based on analysis

# --- Data Loading and Preprocessing ---

def load_and_preprocess_data(dataset_path, filename, target_column, features):
    """
    Load data, handle missing values, and apply log transformation to the target.
    """
    file_path = os.path.join(dataset_path, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    
    # Drop rows with missing values
    df_clean = df.dropna().copy()

    # Log transform the target variable
    df_clean.loc[:, target_column] = np.log1p(df_clean['charges'])
    
    # Select features and target
    df_final = df_clean[features + [target_column]]
    
    return df_final

def split_data(df, target_column, test_size=0.2, val_size=0.25, random_state=42):
    """
    Split data into train, validation, and test sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    # First split: separate test set
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: separate train and validation from full train
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train, y_full_train, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_features(X_train, X_val, X_test):
    """
    Convert dataframes to dictionaries, apply one-hot encoding, and scale features.
    """
    # Convert to dictionaries
    train_dict = X_train.to_dict(orient="records")
    val_dict = X_val.to_dict(orient="records")
    test_dict = X_test.to_dict(orient="records")

    # Initialize and fit DictVectorizer
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)

    # Transform the data
    X_train_encoded = dv.transform(train_dict)
    X_val_encoded = dv.transform(val_dict)
    X_test_encoded = dv.transform(test_dict)

    # Initialize and fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_val_scaled = scaler.transform(X_val_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    return X_train_scaled, X_val_scaled, X_test_scaled, dv, scaler

# --- Model Training and Evaluation ---

def train_xgboost_model(X_train, y_train):
    """
    Train an XGBoost model with hyperparameter tuning using GridSearchCV.
    """
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validated RMSE: {-grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on the validation set.
    """
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    print(f"Validation RMSE: {rmse:.4f}")
    return rmse

# --- Model Saving ---

def save_model_and_preprocessors(model, dv, scaler, filename):
    """
    Save the trained model, DictVectorizer, and StandardScaler to a file.
    """
    with open(filename, "wb") as f_out:
        pickle.dump((model, dv, scaler), f_out)
    print(f"Model, vectorizer, and scaler saved to {filename}")

# --- Main Execution Script ---

def main():
    """
    Main function to execute the training pipeline.
    """
    print("Starting XGBoost training pipeline...")

    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df_processed = load_and_preprocess_data(DATASET_PATH, DATA_FILENAME, TARGET_COLUMN, FEATURES)
    
    # 2. Split data
    print("Splitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_processed, TARGET_COLUMN)
    
    # 3. Prepare features (vectorize and scale)
    print("Preparing features (vectorizing and scaling)...")
    X_train_scaled, X_val_scaled, X_test_scaled, dv, scaler = prepare_features(X_train, X_val, X_test)

    # 4. Train XGBoost model
    print("Training XGBoost model with hyperparameter tuning...")
    best_model = train_xgboost_model(X_train_scaled, y_train)

    # 5. Evaluate the best model on the validation set
    print("\nEvaluating the best model on the validation set...")
    evaluate_model(best_model, X_val_scaled, y_val)

    # 6. Save the final model and preprocessors
    print(f"\nSaving the final model and preprocessors to {MODEL_FILENAME}...")
    save_model_and_preprocessors(best_model, dv, scaler, MODEL_FILENAME)
    
    print("\nTraining pipeline completed successfully.")

if __name__ == "__main__":
    main()