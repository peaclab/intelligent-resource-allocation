# Data Manipulation and File I/O
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# Machine Learning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

# Utilities
import itertools

# Custom Functions
from ml_model_training import train_xgboost
from kmeans_clustering import cluster_create_sub_dataframes
from plot_functions import plot_everything
from feature_selection import (
    xgboost_feature_selection, 
    random_forest_feature_importance, 
    correlation_feature_selection
)

directory = '/projectnb/peaclab-mon/boztop/resource-allocation/datasets/fugaku/24_04.parquet'

# Load the dataset
df = pd.read_parquet(directory)

# Convert the 'adt', 'sdt', and 'edt' columns to datetime
df['adt'] = pd.to_datetime(df['adt'])
df['sdt'] = pd.to_datetime(df['sdt'])
df['edt'] = pd.to_datetime(df['edt'])

# Calculate the wait time and run time in seconds
df['wait_time'] = (df['sdt'] - df['adt']).dt.total_seconds()
df['run_time'] = (df['edt'] - df['sdt']).dt.total_seconds()

sorted_df = df.sort_values(by='edt')
df_success = sorted_df[sorted_df['exit state'] == 'completed']
df_failure = sorted_df[sorted_df['exit state'] == 'failed']  


numerical_submission_features = ['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req']
categorical_submission_features = ['jid', 'usr', 'jnam', 'jobenv_req']
label_encoders = {col: LabelEncoder() for col in categorical_submission_features}
for col in categorical_submission_features:
    df_success[col] = label_encoders[col].fit_transform(df_success[col])

# *********************************************************************************************************************
# Execution Time Prediction - No Feature Selection
# *********************************************************************************************************************
print('User requested values:\n')
rmse = np.sqrt(((df_success['duration'] - df_success['elpl']) ** 2).mean())
print(f"RMSE of user requested execution time: {rmse}")

mae = np.mean(np.abs(df_success['duration'] - df_success['elpl']))
print(f"MAE of user requested execution time: {mae}")

r2 = r2_score(df_success['duration'], df_success['elpl'])
print(f"R^2 of user requested execution time: {r2} \n")

# Single Model Training
print('Single Model Training - No Feature Selection:\n')
train_features = ['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req','jid', 'usr', 'jnam', 'jobenv_req' ]
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'Fugaku Dataset', 'duration', train_features , 'elpl')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - No Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df_success, train_features , ['duration'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'Fugaku Dataset', 'duration', train_features, 'elpl')
    if rmse is not None:
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        total_jobs = len(y_test)
        requested_gt_target = (req_test > y_test).sum()
        predicted_lt_actual = (y_pred < y_test).sum()
        predicted_gt_actual = (y_pred > y_test).sum()
        op1 = y_pred > y_test
        op2 = req_test > y_test
        sum_predicted_minus_actual = (y_pred[op1] - y_test[op1]).sum()
        sum_requested_minus_actual = (req_test[op2] - y_test[op2]).sum()
        metrics_data.append([i + 1, total_jobs, requested_gt_target, predicted_lt_actual, predicted_gt_actual])
        overprediction_sums.append([sum_predicted_minus_actual, sum_requested_minus_actual])

rmse = np.mean(rmse_list)
print(f"Average RMSE for Clustering: {rmse:.4f}")
mae = np.mean(mae_list)
print(f"Average MAE for Clustering: {mae:.4f}")
r2 = np.mean(r2_list)
print(f"Average R^2 for Clustering: {r2:.4f}\n")

plot_everything(metrics_data, overprediction_sums, 
                'Execution Time Experiment on Fugaku Dataset: Over and Under Prediction Analysis ', 
                'Fugaku Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (Seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/fugaku_plots/no_feature_selection')

# *********************************************************************************************************************

# *********************************************************************************************************************
# Execution Time Prediction - XGBoost Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - XGBoost Feature Selection:\n')
train_features = xgboost_feature_selection(['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req','jid', 'usr', 'jnam', 'jobenv_req' ], 'duration', df_success,'RMSE Values for Each Input Feature Combination', 'RMSE (seconds)', 5, 5)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'Fugaku Dataset', 'duration', train_features_list , 'elpl')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - XGBoost Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df_success, ['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req','jid', 'usr', 'jnam', 'jobenv_req' ] , ['duration'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = xgboost_feature_selection(['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req','jid', 'usr', 'jnam', 'jobenv_req' ], 'duration', sub_df,'RMSE Values for Each Input Feature Combination', 'RMSE (seconds)', 5, 5)
    train_features_list = list(train_features)
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'Fugaku Dataset', 'duration', train_features_list, 'elpl')
    if rmse is not None:
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        total_jobs = len(y_test)
        requested_gt_target = (req_test > y_test).sum()
        predicted_lt_actual = (y_pred < y_test).sum()
        predicted_gt_actual = (y_pred > y_test).sum()
        op1 = y_pred > y_test
        op2 = req_test > y_test
        sum_predicted_minus_actual = (y_pred[op1] - y_test[op1]).sum()
        sum_requested_minus_actual = (req_test[op2] - y_test[op2]).sum()
        metrics_data.append([i + 1, total_jobs, requested_gt_target, predicted_lt_actual, predicted_gt_actual])
        overprediction_sums.append([sum_predicted_minus_actual, sum_requested_minus_actual])

rmse = np.mean(rmse_list)
print(f"Average RMSE for Clustering: {rmse:.4f}")
mae = np.mean(mae_list)
print(f"Average MAE for Clustering: {mae:.4f}")
r2 = np.mean(r2_list)
print(f"Average R^2 for Clustering: {r2:.4f}\n")

plot_everything(metrics_data, overprediction_sums, 
                'Execution Time Experiment on Fugaku Dataset: Over and Under Prediction Analysis ', 
                'Fugaku Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (Seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/fugaku_plots/xgboost_feature_selection')

#*********************************************************************************************************************

# *********************************************************************************************************************
# Execution Time Prediction - Random Forest Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - Random Forest Feature Selection:\n')
train_features = random_forest_feature_importance(['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req','jid', 'usr', 'jnam', 'jobenv_req' ], 'duration', df_success,top_k=5)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'Fugaku Dataset', 'duration', train_features_list , 'elpl')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - Random Forest Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df_success, ['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req','jid', 'usr', 'jnam', 'jobenv_req' ] , ['duration'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = random_forest_feature_importance(['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req','jid', 'usr', 'jnam', 'jobenv_req' ], 'duration', sub_df,top_k=5)
    train_features_list = list(train_features)
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'Fugaku Dataset', 'duration', train_features_list, 'elpl')
    if rmse is not None:
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        total_jobs = len(y_test)
        requested_gt_target = (req_test > y_test).sum()
        predicted_lt_actual = (y_pred < y_test).sum()
        predicted_gt_actual = (y_pred > y_test).sum()
        op1 = y_pred > y_test
        op2 = req_test > y_test
        sum_predicted_minus_actual = (y_pred[op1] - y_test[op1]).sum()
        sum_requested_minus_actual = (req_test[op2] - y_test[op2]).sum()
        metrics_data.append([i + 1, total_jobs, requested_gt_target, predicted_lt_actual, predicted_gt_actual])
        overprediction_sums.append([sum_predicted_minus_actual, sum_requested_minus_actual])

rmse = np.mean(rmse_list)
print(f"Average RMSE for Clustering: {rmse:.4f}")
mae = np.mean(mae_list)
print(f"Average MAE for Clustering: {mae:.4f}")
r2 = np.mean(r2_list)
print(f"Average R^2 for Clustering: {r2:.4f}\n")

plot_everything(metrics_data, overprediction_sums, 
                'Execution Time Experiment on Fugaku Dataset: Over and Under Prediction Analysis ', 
                'Fugaku Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (Seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/fugaku_plots/random_forest_feature_selection')

#*********************************************************************************************************************


# *********************************************************************************************************************
# Execution Time Prediction - Pearson Correlation Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - Pearson Correlation Feature Selection:\n')
train_features = correlation_feature_selection(['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req','jid', 'usr', 'jnam', 'jobenv_req' ], 'duration', df_success,method='pearson', top_n=5)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'Fugaku Dataset', 'duration', train_features_list , 'elpl')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - Pearson Correlation Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df_success, ['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req','jid', 'usr', 'jnam', 'jobenv_req' ] , ['duration'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = correlation_feature_selection(['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req','jid', 'usr', 'jnam', 'jobenv_req' ], 'duration', sub_df, method='pearson', top_n=5)
    train_features_list = list(train_features)
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'Fugaku Dataset', 'duration', train_features_list, 'elpl')
    if rmse is not None:
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        total_jobs = len(y_test)
        requested_gt_target = (req_test > y_test).sum()
        predicted_lt_actual = (y_pred < y_test).sum()
        predicted_gt_actual = (y_pred > y_test).sum()
        op1 = y_pred > y_test
        op2 = req_test > y_test
        sum_predicted_minus_actual = (y_pred[op1] - y_test[op1]).sum()
        sum_requested_minus_actual = (req_test[op2] - y_test[op2]).sum()
        metrics_data.append([i + 1, total_jobs, requested_gt_target, predicted_lt_actual, predicted_gt_actual])
        overprediction_sums.append([sum_predicted_minus_actual, sum_requested_minus_actual])

rmse = np.mean(rmse_list)
print(f"Average RMSE for Clustering: {rmse:.4f}")
mae = np.mean(mae_list)
print(f"Average MAE for Clustering: {mae:.4f}")
r2 = np.mean(r2_list)
print(f"Average R^2 for Clustering: {r2:.4f}\n")

plot_everything(metrics_data, overprediction_sums, 
                'Execution Time Experiment on Fugaku Dataset: Over and Under Prediction Analysis ', 
                'Fugaku Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (Seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/fugaku_plots/pearson_feature_selection')

#*********************************************************************************************************************

