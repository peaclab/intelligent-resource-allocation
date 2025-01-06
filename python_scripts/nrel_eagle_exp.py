# Standard Libraries
from datetime import datetime
import os
import bz2
import itertools

# Data Manipulation and File I/O
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# Machine Learning and Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Custom Functions
from ml_model_training import train_xgboost
from kmeans_clustering import cluster_create_sub_dataframes
from plot_functions import plot_everything
from feature_selection import (
    xgboost_feature_selection, 
    random_forest_feature_importance, 
    correlation_feature_selection
)

file_path = "/projectnb/peaclab-mon/boztop/resource-allocation/datasets/nrel-eagle/eagle_data.csv.bz2"
df = pd.read_csv(file_path, compression='bz2')

df_success = df[df['state'] == 'COMPLETED']
df_failure = df[df['state'] == 'TIMEOUT'] 
df_success['start_time'] = pd.to_datetime(df_success['start_time'])
df_success['submit_time'] = pd.to_datetime(df_success['submit_time'])
df_success['wait_time'] = (df_success['start_time'] - df_success['submit_time']).dt.total_seconds()

for col in ['job_id', 'user', 'account', 'partition', 'qos', 'name', 'work_dir','submit_line']:
    print(f'Encoding the feature {col}')
    le = LabelEncoder()
    df_success[col] = le.fit_transform(df_success[col].astype(str))


# *********************************************************************************************************************
# Execution Time Prediction - No Feature Selection
# *********************************************************************************************************************
 
rmse = np.sqrt(((df_success['run_time'] - df_success['wallclock_req']) ** 2).mean())
print(f"RMSE of user requested execution time: {rmse}")

mae = np.mean(np.abs(df_success['run_time'] - df_success['wallclock_req']))
print(f"MAE of user requested execution time: {mae}")

df_success_clean = df_success.dropna(subset=['run_time', 'wallclock_req'])
r2 = r2_score(df_success_clean['run_time'], df_success_clean['wallclock_req'])
print(f"R^2 of user requested execution time: {r2}")


# Single Model Training
train_features = ['job_id', 'user', 'account', 'partition', 'qos', 'wallclock_req','nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'name', 'work_dir','submit_line', 'wait_time'] 
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'NREL Eagle Dataset', 'run_time', train_features, 'wallclock_req',)
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}")

# Training clusters
sub_dfs = cluster_create_sub_dataframes(df_success, train_features , ['run_time'],10)

rmse_list = []
mae_list = []
r2_list = []

metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'NREL Eagle Dataset', 'run_time', ['wallclock_req','mem_req','processors_req','user','partition'], 'wallclock_req')
    
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
print(f"Average R^2 for Clustering: {r2:.4f}")


plot_everything(metrics_data, overprediction_sums, 
                'Execution Time Experiment on NREL Eagle Dataset: Over and Under Prediction Analysis ', 
                'NREL Eagle Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/eagle_plots/no_feature_selection')

#*********************************************************************************************************************



# *********************************************************************************************************************
# Execution Time Prediction - XGBoost Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - XGBoost Feature Selection:\n')
train_features = xgboost_feature_selection(['job_id', 'user', 'account', 'partition', 'qos', 'wallclock_req','nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'name', 'work_dir','submit_line', 'wait_time'], 'run_time', df_success,'RMSE Values for Each Input Feature Combination', 'RMSE (seconds)', 5, 5)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'NREL Eagle Dataset', 'run_time', train_features_list , 'wallclock_req')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - XGBoost Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df_success, [ 'job_id', 'user', 'account', 'partition', 'qos', 'wallclock_req','nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'name', 'work_dir','submit_line', 'wait_time' ] , ['run_time'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = xgboost_feature_selection(['job_id', 'user', 'account', 'partition', 'qos', 'wallclock_req','nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'name', 'work_dir','submit_line', 'wait_time'], 'run_time', sub_df,'RMSE Values for Each Input Feature Combination', 'RMSE (seconds)', 5, 5)
    train_features_list = list(train_features)
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'NREL Eagle Dataset', 'run_time', train_features_list, 'wallclock_req')
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
                'Execution Time Experiment on NREL Eagle Dataset: Over and Under Prediction Analysis ', 
                'NREL Eagle Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/eagle_plots/xgboost_feature_selection')

#*********************************************************************************************************************

# *********************************************************************************************************************
# Execution Time Prediction - Random Forest Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - Random Forest Feature Selection:\n')
train_features = random_forest_feature_importance(['job_id', 'user', 'account', 'partition', 'qos', 'wallclock_req','nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'name', 'work_dir','submit_line', 'wait_time'], 'run_time', df_success,top_k=5)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'NREL Eagle Dataset', 'run_time', train_features_list , 'wallclock_req')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - Random Forest Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df_success, ['job_id', 'user', 'account', 'partition', 'qos', 'wallclock_req','nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'name', 'work_dir','submit_line', 'wait_time'] , ['run_time'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = random_forest_feature_importance(['job_id', 'user', 'account', 'partition', 'qos', 'wallclock_req','nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'name', 'work_dir','submit_line', 'wait_time'], 'run_time', sub_df,top_k=5)
    train_features_list = list(train_features)
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'NREL Eagle Dataset', 'run_time', train_features_list, 'wallclock_req')
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
                'Execution Time Experiment on NREL Eagle Dataset: Over and Under Prediction Analysis ', 
                'NREL Eagle Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/eagle_plots/random_forest_feature_selection')

#*********************************************************************************************************************


# *********************************************************************************************************************
# Execution Time Prediction - Pearson Correlation Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - Pearson Correlation Feature Selection:\n')
train_features = correlation_feature_selection(['job_id', 'user', 'account', 'partition', 'qos', 'wallclock_req','nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'name', 'work_dir','submit_line', 'wait_time'], 'run_time', df_success,method='pearson', top_n=5)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'NREL Eagle Dataset', 'run_time', train_features_list , 'wallclock_req')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - Pearson Correlation Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df_success, ['job_id', 'user', 'account', 'partition', 'qos', 'wallclock_req','nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'name', 'work_dir','submit_line', 'wait_time' ] , ['run_time'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = correlation_feature_selection(['job_id', 'user', 'account', 'partition', 'qos', 'wallclock_req','nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'name', 'work_dir','submit_line', 'wait_time'], 'run_time', sub_df, method='pearson', top_n=5)
    train_features_list = list(train_features)
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'NREL Eagle Dataset', 'run_time', train_features_list, 'wallclock_req')
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
                'Execution Time Experiment on NREL Eagle Dataset: Over and Under Prediction Analysis ', 
                'NREL Eagle Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/eagle_plots/pearson_feature_selection')

#*********************************************************************************************************************

