# Data Manipulation and Numerical Operations
import pandas as pd
import numpy as np
import re
from pathlib import Path
import datetime

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning and Model Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Utilities
import itertools

# XGBoost
import xgboost as xgb

# Custom Functions
from ml_model_training import train_xgboost
from kmeans_clustering import cluster_create_sub_dataframes
from plot_functions import plot_everything
from feature_selection import (
    xgboost_feature_selection, 
    random_forest_feature_importance, 
    correlation_feature_selection
)

col_headers=["qname", "hostname", "group", "owner", "job_name", "job_number", "account", "priority", 
             "submission_time", "start_time", "end_time", "failed", "exit_status", "ru_wallclock", 
             "ru_utime", "ru_stime", "ru_maxrss", "ru_ixrss", "ru_ismrss", "ru_idrss", "ru_isrss", 
             "ru_minflt", "ru_majflt", "ru_nswap", "ru_inblock", "ru_oublock", "ru_msgsnd", 
             "ru_msgrcv", "ru_nsignals", "ru_nvcsw", "ru_nivcsw", "project", "department", "granted_pe", 
             "slots", "task_number", "cpu", "mem", "io", "category", "iow", "pe_taskid", "maxvmem", 
             "arid", "ar_submission_time"]

file_path = "/projectnb/peaclab-mon/boztop/resource-allocation/datasets/bu_scc/bu.accounting.2023" 
df = pd.read_csv(file_path, encoding='utf-8', skiprows=4, 
                 names=col_headers, sep=':')

df = df[df['ru_wallclock'] != 0]
df = df[df['failed'] == 0]
df = df[df['exit_status'] == 0]
df['lag_time'] = df['start_time'] - df['submission_time']
df = df[df['lag_time'] > 0]
df = df.dropna()


df['execution_time'] = df['end_time'] - df['start_time']
df['total_cpu_time'] = df['ru_utime'] + df['ru_stime']
df['ncpu'] = np.ceil(df['total_cpu_time'] / df['ru_wallclock'])
df['cpu_waste'] = df['slots'] - df['ncpu']
df['submission_time'] = df['submission_time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
df = df.drop(['account','job_number','failed','exit_status','iow','pe_taskid','arid','ar_submission_time'], axis=1)

print("Parsing \n")
def parse_h_rt_flag(column):
    parsed_data = []
    for entry in column:
        # Find all parts starting with '-l'
        matches = re.findall(r'-l ([^\s-]+)', entry)
        h_rt_value = None  # Default to None if h_rt is not present
        for match in matches:
            for item in match.split(','):
                if item.startswith('h_rt='):
                    _, value = item.split('=', 1)
                    h_rt_value = int(value)  # Cast the value to int
        parsed_data.append({'h_rt': h_rt_value})
    return parsed_data

parsed_flags = parse_h_rt_flag(df['category'])
parsed_df = pd.DataFrame(parsed_flags)
df = pd.concat([df, parsed_df], axis=1)
# Default hard time limit is 12 hours
df['h_rt'] = df['h_rt'].fillna(12 * 60 * 60).astype(int)
print("Parsing done!\n")

print("Encoding \n")
numerical_submission_features = ['slots']
categorical_submission_features = ['group', 'owner', 'job_name', 'department', 'granted_pe']
label_encoders = {col: LabelEncoder() for col in categorical_submission_features}
for col in categorical_submission_features:
    df[col] = label_encoders[col].fit_transform(df[col])

print("Encoding done!\n")

# *********************************************************************************************************************
# Execution Time Prediction - No Feature Selection
# *********************************************************************************************************************

rmse = np.sqrt(((df['execution_time'] - df['h_rt']) ** 2).mean())
print(f"RMSE: {rmse}") 

mae = np.mean(np.abs(df['execution_time'] - df['h_rt']))
print(f"MAE of user requested execution time: {mae}")

df_clean = df.dropna(subset=['execution_time', 'h_rt'])
r2 = r2_score(df_clean['execution_time'], df_clean['h_rt'])
print(f"R^2 of user requested execution time: {r2}")

# Single Model Training
train_features = ['group', 'owner', 'job_name', 'department', 'granted_pe','slots']
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df, 'BU SCC Dataset', 'execution_time', train_features , 'h_rt')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}")

# Training clusters
sub_dfs = cluster_create_sub_dataframes(df, train_features, ['execution_time'], 10)

rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'BU SCC Dataset', 'execution_time', train_features , 'h_rt')
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
print(f"Average R^2 for Clustering: {r2:.4f} \n")


plot_everything(metrics_data, overprediction_sums, 
                'Execution Time Experiment on BU SCC Dataset: Over and Under Prediction Analysis ', 
                'BU SCC Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (Seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/bu_scc_plots/no_feature_selection')

# *********************************************************************************************************************


# *********************************************************************************************************************
# Execution Time Prediction - XGBoost Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - XGBoost Feature Selection:\n')
train_features = xgboost_feature_selection(['group', 'owner', 'job_name', 'department', 'granted_pe','slots'], 'execution_time', df,'RMSE Values for Each Input Feature Combination', 'RMSE (seconds)', 5, 5)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df, 'BU SCC Dataset', 'execution_time', train_features_list , 'h_rt')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - XGBoost Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df, [ 'group', 'owner', 'job_name', 'department', 'granted_pe','slots'] , ['execution_time'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = xgboost_feature_selection(['group', 'owner', 'job_name', 'department', 'granted_pe','slots'], 'execution_time', sub_df,'RMSE Values for Each Input Feature Combination', 'RMSE (seconds)', 5, 5)
    train_features_list = list(train_features)
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'BU SCC Dataset', 'execution_time', train_features_list, 'h_rt')
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
                'Execution Time Experiment on BU SCC Dataset: Over and Under Prediction Analysis ', 
                'BU SCC Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/bu_scc_plots/xgboost_feature_selection')

#*********************************************************************************************************************

# *********************************************************************************************************************
# Execution Time Prediction - Random Forest Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - Random Forest Feature Selection:\n')
train_features = random_forest_feature_importance(['group', 'owner', 'job_name', 'department', 'granted_pe','slots'], 'execution_time', df,top_k=5)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df, 'BU SCC Dataset', 'execution_time', train_features_list , 'h_rt')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - Random Forest Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df, ['group', 'owner', 'job_name', 'department', 'granted_pe','slots'] , ['execution_time'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = random_forest_feature_importance(['group', 'owner', 'job_name', 'department', 'granted_pe','slots'], 'execution_time', sub_df,top_k=5)
    train_features_list = list(train_features)
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'BU SCC Dataset', 'execution_time', train_features_list, 'h_rt')
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
                'Execution Time Experiment on BU SCC Dataset: Over and Under Prediction Analysis ', 
                'BU SCC Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/bu_scc_plots/random_forest_feature_selection')

#*********************************************************************************************************************


# *********************************************************************************************************************
# Execution Time Prediction - Pearson Correlation Feature Selection
# *********************************************************************************************************************

# Single Model Training
print('Single Model Training - Pearson Correlation Feature Selection:\n')
train_features = correlation_feature_selection(['group', 'owner', 'job_name', 'department', 'granted_pe','slots'], 'execution_time', df,method='pearson', top_n=5)
train_features_list = list(train_features)
rmse, mae, r2, y_test, y_pred, req_test = train_xgboost(df, 'BU SCC Dataset', 'execution_time', train_features_list , 'h_rt')
print(f"RMSE for single model training: {rmse:.4f}")
print(f"MAE for single model training: {mae:.4f}")
print(f"R^2 for single model training: {r2:.4f}\n")

# Training clusters
print('Training Clusters - Pearson Correlation Feature Selection:\n')
sub_dfs = cluster_create_sub_dataframes(df, ['group', 'owner', 'job_name', 'department', 'granted_pe','slots'] , ['execution_time'], 10)
rmse_list = []
mae_list = []
r2_list = []
metrics_data = []
overprediction_sums = []

for i, sub_df in enumerate(sub_dfs):
    print(f"Processing cluster {i+1}/{len(sub_dfs)}")
    train_features = correlation_feature_selection(['group', 'owner', 'job_name', 'department', 'granted_pe','slots'], 'execution_time', sub_df, method='pearson', top_n=5)
    train_features_list = list(train_features)
    rmse, mae, r2, y_test, y_pred, req_test= train_xgboost(sub_df, 'BU SCC Dataset', 'execution_time', train_features_list, 'h_rt')
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
                'Execution Time Experiment on BU SCC Dataset: Over and Under Prediction Analysis ', 
                'BU SCC Dataset: Sum of Differences in Overprediction Cases Across Clusters',
                'Execution Time (seconds)','/projectnb/peaclab-mon/boztop/resource-allocation/plots/bu_scc_plots/pearson_feature_selection')

#*********************************************************************************************************************


