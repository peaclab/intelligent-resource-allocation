# Data Handling
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# Date and Time Manipulation
from datetime import timedelta

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

# Machine Learning and Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# XGBoost
import xgboost as xgb
from xgboost import XGBRegressor

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

def time_limit_to_seconds(time_limit_str):
    if '-' in time_limit_str:
        days, time_part = time_limit_str.split('-')
        time_parts = time_part.split(':')
        
        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
        elif len(time_parts) == 2:
            hours, minutes = map(int, time_parts)
            seconds = 0
        else:
            raise ValueError(f"Invalid time format: {time_limit_str}")
        
        total_seconds = int(days) * 86400 + hours * 3600 + minutes * 60 + seconds
    else:
        time_parts = time_limit_str.split(':')
        
        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
        elif len(time_parts) == 2:
            hours, minutes = map(int, time_parts)
            seconds = 0
        else:
            raise ValueError(f"Invalid time format: {time_limit_str}")
        
        total_seconds = hours * 3600 + minutes * 60 + seconds
    
    return total_seconds


directory = '/projectnb/peaclab-mon/boztop/resource-allocation/datasets/m100/21_12_job_table/metric=job_info_marconi100/a_0.parquet'
df = pd.read_parquet(directory)

df['submit_time'] = pd.to_datetime(df['submit_time'], format='%Y:%m:%d %H')
df['start_time'] = pd.to_datetime(df['start_time'], format='%Y:%m:%d %H')
df['end_time'] = pd.to_datetime(df['end_time'], format='%Y:%m:%d %H')
df['wait_time'] = (df['start_time'] - df['submit_time']).dt.total_seconds()
df['execution_time'] = (df['end_time'] - df['start_time']).dt.total_seconds()
df['time_limit_sec'] = df['time_limit_str'].apply(time_limit_to_seconds)

success_df = df[df['job_state'] == 'COMPLETED']
cleaned_df = success_df[['array_job_id','cpus_alloc_layout',
        'end_time', 'group_id', 'job_id',  'nodes',   'num_cpus', 'num_nodes',
       'partition', 'priority', 'qos', 'start_time', 'submit_time', 'time_limit_sec','user_id', 'wait_time', 'execution_time']]
df_success = cleaned_df.sort_values(by=['start_time']).reset_index(drop=True)

df_success['partition'] = df_success['partition'].astype(int)
df_success['qos'] = df_success['qos'].astype(int)

numerical_submission_features = ['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec']



# *********************************************************************************************************************
# Execution Time Prediction - No Feature Selection
# *********************************************************************************************************************
 
rmse = np.sqrt(((df_success['execution_time'] - df_success['time_limit_sec']) ** 2).mean())
print(f"RMSE of user requested execution time: {rmse}")

mae = np.mean(np.abs(df_success['execution_time'] - df_success['time_limit_sec']))
print(f"MAE of user requested execution time: {mae}")

r2 = r2_score(df_success['execution_time'], df_success['time_limit_sec'])
print(f"R^2 of user requested execution time: {r2}")

# Single Model Training
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'M100 Dataset', 'execution_time', ['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec'], 'time_limit_sec')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}")

# Training clusters
sub_dfs = cluster_create_sub_dataframes(df_success,['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec'], 'execution_time', 10)

rmse_list = []
mae_list = []
r2_list = []

metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'M100 Dataset', 'execution_time', ['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec'], 'time_limit_sec')
    
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
                'Execution Time Experiment on M100 CINECA Dataset: Over and Under Prediction Analysis ', 
                'M100 CINECA Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/m100_plots/no_feature_selection')

#*********************************************************************************************************************


# *********************************************************************************************************************
# Execution Time Prediction - XGBoost Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - XGBoost Feature Selection:\n')
train_features = xgboost_feature_selection(['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec'], 'execution_time', df_success,'RMSE Values for Each Input Feature Combination', 'RMSE (seconds)', 4, 6)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'M100 Dataset', 'execution_time', train_features_list , 'time_limit_sec')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - XGBoost Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df_success, ['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec' ] , ['execution_time'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = xgboost_feature_selection(['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec'], 'execution_time', sub_df,'RMSE Values for Each Input Feature Combination', 'RMSE (seconds)', 4, 6)
    train_features_list = list(train_features) #'group_id', 'priority', 'user_id','time_limit_sec'
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'M100 Dataset', 'execution_time', train_features_list, 'time_limit_sec')
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
                'Execution Time Experiment on M100 CINECA Dataset: Over and Under Prediction Analysis ', 
                'M100 CINECA Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/m100_plots/xgboost_feature_selection')

#*********************************************************************************************************************

# *********************************************************************************************************************
# Execution Time Prediction - Random Forest Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - Random Forest Feature Selection:\n')
train_features = random_forest_feature_importance(['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec'], 'execution_time', df_success,top_k=6)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'M100 Dataset', 'execution_time', train_features_list , 'time_limit_sec')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - Random Forest Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df_success, ['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec'] , ['execution_time'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = random_forest_feature_importance(['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec'], 'execution_time', sub_df,top_k=6)
    train_features_list = list(train_features)
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'M100 Dataset', 'execution_time', train_features_list, 'time_limit_sec')
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
                'Execution Time Experiment on M100 CINECA Dataset: Over and Under Prediction Analysis ', 
                'M100 CINECA Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/m100_plots/random_forest_feature_selection')

#*********************************************************************************************************************


# *********************************************************************************************************************
# Execution Time Prediction - Pearson Correlation Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - Pearson Correlation Feature Selection:\n')
train_features = correlation_feature_selection(['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec'], 'execution_time', df_success,method='pearson', top_n=6)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df_success, 'M100 Dataset', 'execution_time', train_features_list , 'time_limit_sec')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - Pearson Correlation Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df_success, ['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec' ] , ['execution_time'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = correlation_feature_selection(['array_job_id', 'group_id', 'job_id', 'partition', 'priority','qos', 'user_id','time_limit_sec' ], 'execution_time', sub_df,method='pearson', top_n=6)
    train_features_list = list(train_features)
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'M100 Dataset', 'execution_time', train_features_list, 'time_limit_sec')
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
                'Execution Time Experiment on M100 CINECA Dataset: Over and Under Prediction Analysis ', 
                'M100 CINECA Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/m100_plots/pearson_feature_selection')

#*********************************************************************************************************************

