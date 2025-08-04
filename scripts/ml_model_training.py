import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.utils import resample
from xgboost import XGBRegressor


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def train_model_per_cluster(sub_dataframes, train_features, target_feature, model_type, resampling=False):
    """Train individidual machine learning models per each cluster.

    Args:
        sub_dataframes (list of DataFrame): List of sub-dataframes, one for each cluster.
        train_features (list): List of feature column names for training.
        target_feature (str): Name of the target column.
        model_type (str): Type of model to train, e.g., 'lstm', 'bnn', 'xgboost', 'rf'.
        alpha (int): Scaling factor for bias adjustment.
    """

    models = []
    biases = []

    for i, sub_df in enumerate(sub_dataframes):
        sub_df = sub_df.dropna()
        

        num_samples = sub_df.shape[0]
        if num_samples < 2:
            print(f"Skipping cluster {i}: Not enough samples for training.")
            continue
        
        X_train, X_val, y_train, y_val = train_test_split(sub_df[train_features], sub_df[target_feature], test_size=0.4, random_state=42)


        if resampling:
            # Ensure target_feature is a 1-dimensional array
            target_values = sub_df[target_feature].values.flatten()

            # Create bins for resource usage (e.g., dividing target into 10 bins)
            bins = np.linspace(target_values.min(), target_values.max(), 11)  # 10 equal bins
            bin_indices = np.digitize(target_values, bins) - 1  # Assign each sample to a bin

            resampled_samples = []
            max_bin_size = max(np.bincount(bin_indices))

            # For each bin, resample to make sure each bin has the same size as the largest bin
            for bin_idx in np.unique(bin_indices):
                bin_samples = sub_df[bin_indices == bin_idx]
                # Upsample the bin to match the size of the largest bin
                resampled_bin_samples = resample(
                    bin_samples,
                    replace=True,
                    n_samples=max_bin_size,
                    random_state=42
                )
                resampled_samples.append(resampled_bin_samples)

            # Combine all resampled bins into a balanced dataset
            balanced_df = pd.concat(resampled_samples)

            X_train = balanced_df[train_features]
            y_train = balanced_df[target_feature]

        y_train = np.asarray(y_train).ravel()
        y_val = np.asarray(y_val).ravel()


        if X_train.shape[0] == 0:
            print(f"Skipping cluster {i}: No training data available.")
            continue

        if model_type.lower() == 'lstm':
            X_train_reshaped = X_train.reshape((X_train.shape[0], 2, X_train.shape[1])) # timesteps = 2 for now

            model = Sequential([
                Bidirectional(LSTM(128, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))),

                Dropout(0.2),
                BatchNormalization(),

                Bidirectional(LSTM(64, return_sequences=False)),
                Dropout(0.2),
                BatchNormalization(),

                Dense(64),
                Activation('relu'),
                Dropout(0.2),

                Dense(1, activation='linear')
            ])

            model.compile(optimizer=Adam(), loss='mean_squared_error')
            model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)
            y_pred = model.predict(X_val)

        elif model_type.lower() == 'bnn':
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

            epochs = 10
            batch_size = 32
            model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            y_pred = model.predict(X_val)
        
        elif model_type.lower() == 'xgboost':
            y_train_log = np.log(y_train)
            #model = XGBRegressor(objective='reg:gamma',random_state=42) # objective function changed from reg:squarederror to reg:gamma ; count:poission does not work with memory predictions
            model = XGBRegressor(random_state=42)
            model.fit(X_train, y_train_log) 
            y_pred_log = model.predict(X_val)
            y_pred = np.exp(y_pred_log)

        elif model_type.lower() == 'rf':
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train) 
            y_pred = model.predict(X_val)
        else:
            raise ValueError(f"Model type '{model_type}' not recognized. Choose from 'lstm', 'bnn', 'xgboost', 'rf'.")
        
        # Finding bias from underpredictions
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

        models.append(model)
        #print(f"Model trained for Cluster {i} with bias {bias:.3f}")

    return models, biases

def test_model_per_cluster(test_df, train_features, target_feature, user_req_feature, cluster_centers, models, biases, bias_type, model_type, filename):

    """
    Predict target values for test data using cluster-specific models.

    Args:
        test_df (DataFrame): Test dataset containing the selected features.
        models (list): List of trained models for each cluster.
        biases (list): List of bias values for each cluster.
        cluster_centers (ndarray): Cluster centers from the training phase.
        selected_features (list): List of feature column names used for training.
        model_name (str): Name of the model, e.g., 'lstm'.

    Returns:
        DataFrame: Test DataFrame with added 'cluster' and 'predictions' columns.
    """

    test_features = test_df[train_features].values
    distances = np.linalg.norm(cluster_centers[:, np.newaxis] - test_features, axis=2)
    
    test_df.loc[:, 'cluster'] = np.argmin(distances, axis=0)

    if user_req_feature is not None:
        pred_vs_act_df = pd.DataFrame(columns=['pred', 'act', 'req'])
    else:
        pred_vs_act_df = pd.DataFrame(columns=['pred', 'act'])


    for cluster_id, (model, bias_dict) in enumerate(zip(models, biases)):
        cluster_data = test_df[test_df['cluster'] == cluster_id]
        cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan).dropna()
        

        if not cluster_data.empty:
            X_test = cluster_data[train_features].values
            y_test = cluster_data[target_feature].values

            if model_type.lower() == 'xgboost':
                y_pred_log = model.predict(X_test)
                y_pred = np.exp(y_pred_log)
            else:
                y_pred = model.predict(X_test)

            if bias_type != 'none':
                # Select the appropriate bias based on the specified bias type
                bias = bias_dict.get(bias_type, 0.0) 
                y_pred = y_pred + bias               
            else:
                y_pred = y_pred
            
            y_pred = np.array(y_pred).flatten()
            y_test = np.array(y_test).flatten()

            if user_req_feature is not None:
                user_req = cluster_data[user_req_feature].values
                user_req = np.array(user_req).flatten()
                new_df = pd.DataFrame({
                    'pred': y_pred,
                    'act': y_test,
                    'req': user_req
                })
            else:
                new_df = pd.DataFrame({
                    'pred': y_pred,
                    'act': y_test,
                })

            pred_vs_act_df = pd.concat([pred_vs_act_df, new_df], ignore_index=True)

    pred_vs_act_df.to_pickle(filename)





