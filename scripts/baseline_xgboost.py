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
import matplotlib.pyplot as plt
import seaborn as sns


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


def train_eagle_xgboost(train_df, train_features, target_feature):

    biases = []

    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X_train, X_val, y_train, y_val = train_test_split(train_df[train_features], train_df[target_feature], test_size=0.4, random_state=42)

    
    params = { 
        'n_estimators': 168,
        'max_depth': 7,
        'learning_rate': 0.3968571956999504,
        'gamma': 0.640232768439118,
        'subsample': 0.747747407403972,
        'colsample_bytree': 0.6280085182287491,
        
    }

    model = XGBRegressor(**params)
    #model = XGBRegressor(objective='reg:gamma', **params)
    #model = XGBRegressor(objective='count:poisson')
    #model = XGBRegressor()
    y_train_log = np.log(y_train)
    model.fit(X_train, y_train_log)

    # Finding bias from underpredictions
    y_pred_log = model.predict(X_val)
    y_pred = np.exp(y_pred_log)

    y_val = np.asarray(y_val).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Calculate underprediction amount
    underprediction_mask = y_val > y_pred
    underprediction_amount = y_val[underprediction_mask] - y_pred[underprediction_mask]

    # Calculate bias
    mean_val = np.mean(underprediction_amount)
    median = np.median(underprediction_amount)
    mad = np.median(np.abs(underprediction_amount - median))
    std_dev = np.std(underprediction_amount)
    two_sigma = 2 * std_dev

    biases.append({
        'mean': mean_val,
        'mad': mad,
        'std_dev': std_dev,
        'two_sigma': two_sigma
    })

    return model, biases


def test_eagle_xgboost(test_df, model, biases, bias_type, train_features, target_feature, user_req_feature, file_path):
    
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()

    X_test = test_df[train_features]
    y_test = test_df[target_feature]

    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)

    if bias_type != 'none':
        last_bias = biases[-1]
        if bias_type == 'mean':
            bias = last_bias['mean']
        elif bias_type == 'mad':
            bias = last_bias['mad']
        elif bias_type == 'std_dev':
            bias = last_bias['std_dev']
        elif bias_type == 'two_sigma':
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

    '''Some additional plots to visualize the results'''
    '''
    print(f'Predictions vs. Actuals for bias = {bias}')

    plt.figure(figsize=(10, 6))
    plt.scatter(pred_vs_act_df['act'], pred_vs_act_df['pred'], alpha=0.5)
    plt.plot([pred_vs_act_df['act'].min(), pred_vs_act_df['act'].max()], 
            [pred_vs_act_df['act'].min(), pred_vs_act_df['act'].max()], 
            color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs. Actuals for {target_feature}')
    plt.show()

    overestimation_factor = y_pred / y_test

    plt.figure(figsize=(10, 6))
    sns.kdeplot(overestimation_factor, shade=True, log_scale=True, fill=True, alpha=0.5)
    plt.axvline(x=1, color='red', linestyle='--')  # Add red dashed line for perfect prediction
    plt.grid(True)
    plt.xlabel('Overestimation Factor (y_pred / y_test)')
    plt.title('KDE Plot for Overestimation Factor')
    plt.show()
    '''

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f'r2: {r2:.3f}, rmse: {rmse:.0f}')
     
    pred_vs_act_df.to_pickle(file_path)