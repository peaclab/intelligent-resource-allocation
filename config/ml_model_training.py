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


# Custom loss function
def custom_loss(y_true, y_pred):
    residuals = y_pred - y_true

    grad = np.where(residuals < 0, -residuals * 0.5, residuals)  # Underpredictions have a penalty factor of 0.5, overpredictions keep the original residuals
    hess = np.where(residuals < 0, np.ones_like(residuals) * 0.5, np.ones_like(residuals))  # Hessian adjusted for underpredictions
    
    return grad, hess

def train_xgboost(df, df_name, target_feature, train_features, requested_feature, filepath, random_state=42):

    if target_feature not in df.columns:
        raise ValueError(f"Target feature '{target_feature}' not found in the DataFrame.")
    
    df.loc[:, 'lagged_target'] = df[target_feature].shift(1) #currently shifting by 1
    df = df.dropna() 
    
    features_with_lag = train_features + ['lagged_target']    
    X = df[features_with_lag]
    y = df[target_feature]
    requested_values = df[requested_feature]

    if len(X) < 2:  # If there's only one sample or fewer, splitting is not possible
        print(f"Warning: Not enough data in {df_name} to perform train-test split. Returning NaN for RMSE.")
        return None, None, None, None, None, None
    
    test_size = 0.2
    X_train, X_test, y_train, y_test, req_train, req_test = train_test_split(X, y, requested_values, test_size=test_size, random_state=random_state)

    mse1 = mean_squared_error(y_test, req_test)
    rmse1 = np.sqrt(mse1)
    mae1 = mean_absolute_error(y_test, req_test)
    r21 = r2_score(y_test, req_test)
    print(f"RMSE of user requested execution time: {rmse1}")
    print(f"MAE of user requested execution time: {mae1}")
    print(f"R^2 of user requested execution time: {r21}")

    model = XGBRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #Some metrics for comparison
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    plot_raw_results(y_pred, y_test, req_test, target_feature, df_name, filepath)

    # Overestimation factor
    requested_over_target = req_test / y_test
    predicted_over_target = y_pred / y_test
    plot_kde_results(requested_over_target, predicted_over_target, target_feature, df_name)


    return rmse, mae, r2, y_test, y_pred, req_test



def train_bayesian_nn(df, df_name, target_feature, train_features, requested_feature, filepath, random_state=42):
    if target_feature not in df.columns:
        raise ValueError(f"Target feature '{target_feature}' not found in the DataFrame.")

    df.loc[:, 'lagged_target'] = df[target_feature].shift(1)
    df = df.dropna()

    features_with_lag = train_features + ['lagged_target']
    X = df[features_with_lag]
    y = df[target_feature]
    requested_values = df[requested_feature]

    if len(X) < 2:
        print(f"Warning: Not enough data in {df_name} to perform train-test split. Returning NaN for RMSE.")
        return None, None, None, None, None, None

    test_size = 0.2
    X_train, X_test, y_train, y_test, req_train, req_test = train_test_split(X, y, requested_values, test_size=test_size, random_state=random_state)

    mse1 = mean_squared_error(y_test, req_test)
    rmse1 = np.sqrt(mse1)
    mae1 = mean_absolute_error(y_test, req_test)
    r21 = r2_score(y_test, req_test)
    print(f"RMSE of user requested execution time: {rmse1}")
    print(f"MAE of user requested execution time: {mae1}")
    print(f"R^2 of user requested execution time: {r21}")

    input_dim = X_train.shape[1]
    model = Sequential([
        Dense(128, input_dim=input_dim, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(64, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    y_pred = model.predict(X_test).flatten()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    plot_raw_results(y_pred, y_test, req_test, target_feature, df_name, filepath)
    
    return rmse, mae, r2, y_test, y_pred, req_test

def train_lstm(df, df_name, target_feature, train_features, requested_feature, filepath, random_state=42):
    if target_feature not in df.columns:
        raise ValueError(f"Target feature '{target_feature}' not found in the DataFrame.")

    df.loc[:, 'lagged_target'] = df[target_feature].shift(1)
    df = df.dropna()

    features_with_lag = train_features + ['lagged_target']
    X = df[features_with_lag].values
    y = df[target_feature].values
    requested_values = df[requested_feature].values

    if len(X) < 2:
        print(f"Warning: Not enough data in {df_name} to perform train-test split. Returning NaN for RMSE.")
        return None, None, None, None, None, None

    test_size = 0.2
    X_train, X_test, y_train, y_test, req_train, req_test = train_test_split(X, y, requested_values, test_size=test_size, random_state=random_state)


    mse1 = mean_squared_error(y_test, req_test)
    rmse1 = np.sqrt(mse1)
    mae1 = mean_absolute_error(y_test, req_test)
    r21 = r2_score(y_test, req_test)
    print(f"RMSE of user requested execution time: {rmse1}")
    print(f"MAE of user requested execution time: {mae1}")
    print(f"R^2 of user requested execution time: {r21}")

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = (X_train.shape[1], 1)
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
        Dropout(0.2),
        BatchNormalization(),
        
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.2),
        BatchNormalization(),
        
        Dense(64),
        Activation('relu'),
        Dropout(0.2),
        
        Dense(1, activation='linear')  # Linear activation for regression tasks
    ])

    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred = model.predict(X_test).flatten()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    plot_raw_results(y_pred, y_test, req_test, target_feature, df_name, filepath)

    return rmse, mae, r2, y_test, y_pred, req_test
