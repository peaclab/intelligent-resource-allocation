import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

class ModelTrainer:
    def __init__(self, model_type, resampling=False):
        self.model_type = model_type
        self.resampling = resampling

    def train_model_per_cluster(self, sub_dataframes, train_features, target_feature):
        """Train individual machine learning models per each cluster.

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
                y_train_log = np.log(y_train) # Apply log transform to prevent negative values
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

            # Calculate different bias values
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
