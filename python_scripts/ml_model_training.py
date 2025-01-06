import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from plot_functions import plot_raw_results, plot_kde_results

# Custom loss function
def custom_loss(y_true, y_pred):
    residuals = y_pred - y_true

    grad = np.where(residuals < 0, -residuals * 0.5, residuals)  # Underpredictions have a penalty factor of 0.5, overpredictions keep the original residuals
    hess = np.where(residuals < 0, np.ones_like(residuals) * 0.5, np.ones_like(residuals))  # Hessian adjusted for underpredictions
    
    return grad, hess

def train_xgboost(df, df_name, target_feature, train_features, requested_feature, random_state=42):

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

    #print(f"Train size (X_train): {len(X_train)}")
    #print(f"Test size (X_test): {len(X_test)}")
    #print(f"Train size (y_train): {len(y_train)}")
    #print(f"Test size (y_test): {len(y_test)}")
    model = XGBRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #Some metrics for comparison
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    plot_raw_results(y_pred, y_test, req_test, target_feature, df_name)

    # Overestimation factor
    requested_over_target = req_test / y_test
    predicted_over_target = y_pred / y_test
    plot_kde_results(requested_over_target, predicted_over_target, target_feature, df_name)

    return rmse, mae, r2, y_test, y_pred, req_test

