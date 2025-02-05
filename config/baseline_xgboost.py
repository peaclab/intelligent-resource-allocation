"""
    -This script provides the second baseline [Menear et al., PEARC'23] for the resource allocation problem we solve.
    -The baseline model is an XGBoost-based model that predicts the execution time for a given workload.
    
    -The model architecture is based on the following paper:
    "Menear, K., Nag, A., Perr-Sauer, J., Lunacek, M., Potter, K., & Duplyakin, D. (2023). 
    Mastering HPC Runtime Prediction: From Observing Patterns to a Methodological Approach. 
    In Practice and Experience in Advanced Research Computing (pp. 75-85)."
    - GitHub: https://github.com/NREL/eagle-jobs/tree/master

"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from plot_functions import plot_raw_results, plot_kde_results

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def train_eagle_xgboost(df, df_name, target_feature, train_features, requested_feature, filepath, random_state=42):
    if target_feature not in df.columns:
        raise ValueError(f"Target feature '{target_feature}' not found in the DataFrame.")

    df.loc[:, 'lagged_target'] = df[target_feature].shift(1)  # Currently shifting by 1
    df = df.dropna()

    features_with_lag = train_features + ['lagged_target']
    X = df[features_with_lag]
    y = df[target_feature]
    requested_values = df[requested_feature]

    if len(X) < 2:  # If there's only one sample or fewer, splitting is not possible
        print(f"Warning: Not enough data in {df_name} to perform train-test split. Returning NaN for RMSE.")
        return None, None, None, None, None, None

    test_size = 0.2
    X_train, X_test, y_train, y_test, req_train, req_test = train_test_split(
        X, y, requested_values, test_size=test_size, random_state=random_state
    )

    mse1 = mean_squared_error(y_test, req_test)
    rmse1 = np.sqrt(mse1)
    mae1 = mean_absolute_error(y_test, req_test)
    r21 = r2_score(y_test, req_test)
    print(f"RMSE of user requested execution time: {rmse1}")
    print(f"MAE of user requested execution time: {mae1}")
    print(f"R^2 of user requested execution time: {r21}")

    params = { 
        'n_estimators': 168,
        'max_depth': 7,
        'learning_rate': 0.3968571956999504,
        'gamma': 0.640232768439118,
        'subsample': 0.747747407403972,
        'colsample_bytree': 0.6280085182287491
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Some metrics for comparison
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    #plot_raw_results(y_pred, y_test, req_test, target_feature, df_name, filepath)

    # Overestimation factor
    requested_over_target = req_test / y_test
    predicted_over_target = y_pred / y_test
    plot_kde_results(requested_over_target, predicted_over_target, target_feature, df_name, filepath)

    return rmse, mae, r2, y_test, y_pred, req_test
