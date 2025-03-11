import numpy as np
import pandas as pd

class ModelTester:
    def __init__(self, models, biases, cluster_centers, bias_type, model_type):
        self.models = models
        self.biases = biases
        self.cluster_centers = cluster_centers
        self.bias_type = bias_type
        self.model_type = model_type

    def test_single_input(self, input_data, train_features, target_feature):
        """
        Predict target value for a single input using the closest cluster-specific model.

        Args:
            input_data (dict): Dictionary containing the input features.
            train_features (list): List of feature column names used for training.
            target_feature (str): Name of the target feature column.
            user_req_feature (str): Name of the user request feature column (optional).

        Returns:
            dict: Dictionary with predicted target value and optionally user request value.
        """

        input_features = np.array([input_data[feature] for feature in train_features], dtype=float).reshape(1, -1)
        cluster_centers = np.array(self.cluster_centers, dtype=float)
        distances = np.linalg.norm(cluster_centers - input_features, axis=1)
        closest_cluster = np.argmin(distances)

        model = self.models[closest_cluster]
        bias_dict = self.biases[closest_cluster]

        if self.model_type.lower() == 'xgboost':
            y_pred_log = model.predict(input_features)
            y_pred = np.exp(y_pred_log)
        else:
            y_pred = model.predict(input_features)

        if self.bias_type != 'none':
            bias = bias_dict.get(self.bias_type, 0.0)
            y_pred = y_pred + bias

        y_pred = y_pred.flatten()[0]

        result = {'predicted_' + target_feature: y_pred}
        
        return result
