import sys
sys.path.append('../scripts')

from fugaku_data_preprocessing import preprocess_data
from ml_model_training import train_model_per_cluster, test_model_per_cluster
from baseline_xgboost import train_eagle_xgboost, test_eagle_xgboost
from kmeans_clustering import create_sub_dataframes

# Standard Libraries
import os
import sys
import datetime
from datetime import timedelta

# Data manipulation and numerical operations
import pandas as pd
import numpy as np

# Machine Learning and Model Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Plotting and Visualization
import matplotlib.pyplot as plt


if __name__ == "__main__":

    directory = '../data/fugaku/24_04.parquet' # hard coded, change if needed
    df_success, df_failure, numerical_features = preprocess_data(directory)
    print("Fugaku dataset - preprocessing complete.")

    start_time = '2024-04-05' # Based on the chosen .parquet file for Fugaku 
    end_time = '2024-04-30'
    window_size = 3 # The size of the window we make predictions on; change if you want
    update_frequency = '3D' # Update frequency for the sliding window; change if you want


    start_dates = pd.date_range(start=start_time, end=end_time, freq=update_frequency)
    end_dates = start_dates + pd.DateOffset(days=window_size)
    date_pairs = list(zip(start_dates, end_dates))

    train_features = ['usr', 'jnam', 'jid', 'cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req'] #'jid' improves prediction accuracy on Fugaku 
    target_feature = ['duration']  # can be 'cunumut', 'duration', or 'mmszu'
    user_req_feature = 'elpl' # can be 'cnumr', 'elpl', or 'mszl'
    bias_types = ['none', 'two_sigma'] 

    df = df_success.copy()
    df['edt'] = pd.to_datetime(df['edt']).dt.tz_convert(None)
    df = df.sort_values(by='adt').reset_index(drop=True)

    for start_date, end_date in date_pairs:
    
        df_slice = df[(df['edt'] >= start_date) & (df['edt'] < end_date)]
        train_df, test_df = train_test_split(df_slice, test_size=0.2, random_state=33)

        print("[INFO] Creating sub-dataframes and clustering...")
        sub_dataframes, cluster_centers = create_sub_dataframes(
            df=train_df, 
            selected_features=train_features, 
            n_clusters=4 # For Fugaku, we select 4 clusters based on the dataset characteristics
        )


        print("[INFO] Training plain models (XGBoost and RandomForest)...")
        xgb_plain_models, xgb_plain_biases = train_model_per_cluster(
            sub_dataframes, train_features, target_feature, 'xgboost')
        rf_plain_models, rf_plain_biases = train_model_per_cluster(
            sub_dataframes, train_features, target_feature, 'rf')
        print("[INFO] Plain models training completed.")


        for bias_type in bias_types:
            print(f"[INFO] Evaluating with bias type: {bias_type}...")

            test_model_per_cluster(
                test_df, train_features, target_feature, user_req_feature, cluster_centers,
                xgb_plain_models, xgb_plain_biases, bias_type, 'xgboost',
                f'../results/fugaku/fugaku_clustering_xgb_bias_{bias_type}_{target_feature}_window_{window_size}_days_{start_date}.pkl'
            ) # output directory hardcoded for now; can be changed later

            test_model_per_cluster(
                test_df, train_features, target_feature, user_req_feature, cluster_centers,
                rf_plain_models, rf_plain_biases, bias_type, 'rf',
                f'../results/fugaku/fugaku_clustering_rf_bias_{bias_type}_{target_feature}_window_{window_size}_days_{start_date}.pkl'
            ) # output directory hardcoded for now; can be changed later


        print("[INFO] Training resampled models (XGBoost and RandomForest)...")
        xgb_resampled_models, xgb_resampled_biases = train_model_per_cluster(
            sub_dataframes, train_features, target_feature, 'xgboost',resampling=True)
        rf_resampled_models, rf_resampled_biases = train_model_per_cluster(
            sub_dataframes, train_features, target_feature, 'rf',resampling=True)
        print("[INFO] Resampled models training completed.")
        
        for bias_type in bias_types:
            print(f"[INFO] Evaluating with bias type: {bias_type}...")

            test_model_per_cluster(
                test_df, train_features, target_feature, user_req_feature, cluster_centers,
                xgb_resampled_models, xgb_resampled_biases, bias_type, 'xgboost',
                f'../results/fugaku/fugaku_resampled_xgb_bias_{bias_type}_{target_feature}_window_{window_size}_days_{start_date}.pkl'
            ) # output directory hardcoded for now; can be changed later

            test_model_per_cluster(
                test_df, train_features, target_feature, user_req_feature, cluster_centers,
                rf_resampled_models, rf_resampled_biases, bias_type, 'rf',
                f'../results/fugaku/fugaku_resampled_rf_bias_{bias_type}_{target_feature}_window_{window_size}_days_{start_date}.pkl'
            )  # output directory hardcoded for now; can be changed later
            
           
            
            