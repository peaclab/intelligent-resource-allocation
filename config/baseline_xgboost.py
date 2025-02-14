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


def train_eagle_xgboost(train_df, test_df, train_features, target_feature, filename):
    if target_feature not in df.columns:
        raise ValueError(f"Target feature '{target_feature}' not found in the DataFrame.")

    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    X_train = train_df[train_features]
    X_test = test_df[train_features]
    y_train = train_df[target_feature]
    y_test = test_df[target_feature]

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

    pred_vs_act_df = pd.DataFrame(columns=['pred', 'act'])
    pred_vs_act_df = pd.DataFrame({'pred': y_pred, 'act': y_test})

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f'r2: {r2:.3f}, rmse: {rmse:.0f}')
     
    pred_vs_act_df.to_pickle(filename)

def test_eagle_xgboost(test_df, model, biases, bias, train_features, target_feature, user_req_feature, file_path):
    
    test_df = test_df.dropna()
    X_test = test_df[train_features]
    y_test = test_df[target_feature]

    y_pred = model.predict(X_test)

    if bias != 'none':
        last_bias = biases[-1]
        if bias == 'mean':
            bias = last_bias['mean']
        elif bias == 'mad':
            bias = last_bias['mad']
        elif bias == 'std_dev':
            bias = last_bias['std_dev']
        elif bias == 'two_sigma':
            bias = last_bias['two_sigma']
    else:
        bias = 0.0       
    
    y_pred += bias

    y_pred = np.array(y_pred).flatten()
    y_test = np.array(y_test).flatten()

    if user_req_feature is not None:
        user_req = test_df[user_req_feature]
        user_req = np.array(user_req).flatten()
        pred_vs_act_df = pd.DataFrame(columns=['pred', 'act', 'req'])
        pred_vs_act_df = pd.DataFrame({'pred': y_pred, 'act': y_test, 'req': user_req})
    else:
        pred_vs_act_df = pd.DataFrame(columns=['pred', 'act'])
        pred_vs_act_df = pd.DataFrame({'pred': y_pred, 'act': y_test})

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f'r2: {r2:.3f}, rmse: {rmse:.0f}')
     
    pred_vs_act_df.to_pickle(file_path)