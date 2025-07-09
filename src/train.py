#!/share/pkg.8/python3/3.12.4/install/bin/python3

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.utils import resample

from load import LoadFilterData
from group import KMeansClustering

import os
import getpass



class ModelTrainer:
    def __init__(self, model_type, resampling=False):
        self.model_type = model_type
        self.resampling = resampling

    def train_model_per_cluster(self, sub_dataframes, train_features, target_feature):
        """Train individual machine learning models for each cluster.

        Args:
            sub_dataframes (list of DataFrame): List of sub-dataframes, one for each cluster.
            train_features (list): List of feature column names for training.
            target_feature (str): Name of the target column.
        """

        models = []
        biases = []

        for i, sub_df in enumerate(sub_dataframes):
            sub_df = sub_df.dropna()

            num_samples = sub_df.shape[0]
            if num_samples < 2:
                print(f"Skipping cluster {i}: Not enough samples for training.")
                continue

            # 40% for validation
            X_train, X_val, y_train, y_val = train_test_split(sub_df[train_features], sub_df[target_feature], test_size=0.4, random_state=42)

            if self.resampling:
                target_values = sub_df[target_feature].values.flatten()

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

            if self.model_type.lower() == 'xgboost':
                y_train_log = np.log(y_train) # Apply log transform to prevent negative values, specific to XGBoost
                model = XGBRegressor(random_state=42)
                model.fit(X_train, y_train_log)
                y_pred_log = model.predict(X_val)
                y_pred = np.exp(y_pred_log)

            elif self.model_type.lower() == 'rf':
                model = RandomForestRegressor(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            else:
                raise ValueError(f"Model type '{self.model_type}' not recognized. Choose from 'xgboost', 'rf'.")

            # Finding bias from underpredictions
            y_val = np.asarray(y_val).flatten()
            y_pred = np.asarray(y_pred).flatten()

            # Calculate underprediction amount
            underprediction_mask = y_val > y_pred
            underprediction_amount = y_val[underprediction_mask] - y_pred[underprediction_mask]

            # Calculate different bias values for further analysis
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

        return models, biases


class UserBasedModelTrainer:
    def __init__(self, user_name, resampling=True):
        self.user_name = user_name
        self.resampling = resampling
        self.train_features = ['project', 'owner', 'job_name', 'slots','h_rt']
        self.target_feature = 'ncpu'


    def train_model_per_user(self, df):
        biases = []

        # Filter the dataframe for the specific user
        user_df = df[df['owner'] == self.user_name]
        num_samples = user_df.shape[0]
        print(f"Number of samples for user {self.user_name}: {num_samples}")
        if num_samples < 2:
            print(f"Not enough samples for training for user {self.user_name}.")
            return None, None

        categorical_features = ['project', 'owner', 'job_name']
        label_encoders = {}
        for feature in categorical_features:
            le = LabelEncoder()
            user_df.loc[:, feature] = le.fit_transform(user_df[feature])
            label_encoders[feature] = le

        X_train, X_val, y_train, y_val = train_test_split(user_df[self.train_features], user_df[self.target_feature], test_size=0.4, random_state=42)

        if self.resampling:
            target_values = user_df[self.target_feature].values.flatten()
            bins = np.linspace(target_values.min(), target_values.max(), 11)  # 10 equal bins
            bin_indices = np.digitize(target_values, bins) - 1  # Assign each sample to a bin
            resampled_samples = []
            max_bin_size = max(np.bincount(bin_indices))
            # For each bin, resample to make sure each bin has the same size as the largest bin
            for bin_idx in np.unique(bin_indices):
                bin_samples = user_df[bin_indices == bin_idx]
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
            X_train = balanced_df[self.train_features]
            y_train = balanced_df[self.target_feature]
        y_train = np.asarray(y_train).ravel()
        y_val = np.asarray(y_val).ravel()

        if X_train.shape[0] == 0:
            print(f"Skipping user {self.user_name}: No training data available.")
            return None, None           

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on validation set
        y_pred = model.predict(X_val)

        y_val = np.asarray(y_val).flatten()
        y_pred = np.asarray(y_pred).flatten()

        underprediction_mask = y_val > y_pred
        underprediction_amount = y_val[underprediction_mask] - y_pred[underprediction_mask]
        
        if len(underprediction_amount) > 0:
            # Calculate bias metrics
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

        else:
            biases.append({
                'mean': 0.0,
                'mad': 0.0,
                'std_dev': 0.0,
                'two_sigma': 0.0
            })
        # Store the model and bias
        return label_encoders, model, biases


if __name__ == "__main__":

    data = pd.read_pickle('/projectnb/peaclab-mon/boztop/module_files/data/accounting_2025.pkl')
    print("Data loaded.")

    # Save the label encoders into the user's home directory under the .config folder
    home_dir = os.path.expanduser("~")
    config_dir = os.path.join(home_dir, ".config", "myrec_models")
    os.makedirs(config_dir, exist_ok=True)

    # This file contains clustering vs user-based training decision for each user. 
    # We need different automated_decision.csv for each HPC system.
    automated_df = pd.read_csv('/projectnb/peaclab-mon/boztop/module_files/data/automated_decision.csv')

    user_name = getpass.getuser()
    
    # Check if the user has a label in the automated_df
    label = automated_df.loc[automated_df['user'] == user_name, 'label'].values[0]

    if label == 0:
        print(f"User {user_name} has label 0. Clustering + Resampling model training.")

        train_features = ['project', 'owner', 'job_name', 'slots','h_rt']
        target_feature = 'ncpu'

        categorical_features = ['project', 'owner', 'job_name']
        label_encoders = {}
        for feature in categorical_features:
            le = LabelEncoder()
            data[feature] = le.fit_transform(data[feature])
            label_encoders[feature] = le
            label_encoders_path = os.path.join(config_dir, "label_encoders.pkl")
        
        with open(label_encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        print(f"Label encoders saved successfully to {label_encoders_path}.")

        grouper = KMeansClustering(n_clusters=10) # Adjust the number of clusters with elbow method.
        sub_dataframes, cluster_centers = grouper.create_sub_dataframes(data, train_features, days_to_train=90)
        print("Sub-dataframes created. Training models...")
            
        cluster_centers = np.array(cluster_centers, dtype=float)
        cluster_centers_path = os.path.join(config_dir, "cluster_centers.npy")
        np.save(cluster_centers_path, cluster_centers)
        print(f"Cluster centers saved successfully to {cluster_centers_path}.")


        trainer = ModelTrainer(model_type='rf', resampling=True)
        models, biases = trainer.train_model_per_cluster(sub_dataframes, train_features, target_feature)
        print(f"Models for {target_feature} trained successfully.")

        # Save the trained models and biases using numpy.savez
        model_data = {'models': models, 'biases': biases}
        models_path = os.path.join(config_dir, "trained_models.pkl")
        with open(models_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Models and biases saved successfully as a pickle file to {models_path}.")
    
    else:
        trainer = UserBasedModelTrainer(user_name=user_name, resampling=True)
        label_encoders, model, biases = trainer.train_model_per_user(data)

        model_path = os.path.join(config_dir, f"user_based_model_{user_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved successfully to {model_path}.")
        # Save the biases using numpy.savez
        biases_path = os.path.join(config_dir, f"user_based_biases_{user_name}.npz")
        np.savez(biases_path, biases=biases)
        print(f"Biases saved successfully to {biases_path}.")
        # Save the label encoders using pickle  
        label_encoders_path = os.path.join(config_dir, f"user_based_label_encoders_{user_name}.pkl")
        with open(label_encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        print(f"Label encoders saved successfully to {label_encoders_path}.")
         
    

    

    


