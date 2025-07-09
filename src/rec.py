#!/share/pkg.8/python3/3.12.4/install/bin/python3

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
import getpass
import os
import grp
import sys


class ModelTester:
    def __init__(self, models, biases, cluster_centers, bias_type, model_type):
        self.models = models
        self.biases = biases
        self.cluster_centers = cluster_centers
        self.bias_type = bias_type
        self.model_type = model_type

    def test_single_input(self, input_data, train_features, target_feature):
        """
        Predict resource request value for a single input using the closest cluster's model.

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

class UserBasedModelTester:
    def __init__(self, model, biases, label_encoder, bias_type):
        self.model = model
        self.biases = biases
        self.label_encoder = label_encoder
        self.bias_type = bias_type

    def test_single_user_input(self, input_data, train_features, target_feature):
        """
        Predict target value for a single input in a user-based model.

        Args:
            input_data (dict): Dictionary containing the input features.
            train_features (list): List of feature column names used for training.
            target_feature (str): Name of the target feature column.

        Returns:
            dict: Dictionary with predicted target value.
        """
        # Transform input features using the label encoder
        for feature in train_features:
            if feature in self.label_encoder and input_data[feature] in self.label_encoder[feature].classes_:
                input_data[feature] = self.label_encoder[feature].transform([input_data[feature]])[0]
            elif feature in self.label_encoder:
                raise ValueError(f"Unknown value '{input_data[feature]}' for feature '{feature}'.")

        input_features = pd.DataFrame([[input_data[feature] for feature in train_features]], columns=train_features)

        y_pred = self.model.predict(input_features)

        if self.bias_type != 'none':
            bias = self.biases.get(self.bias_type, 0.0)
            print(f"Bias for {self.bias_type}: {bias}")
            print(f"Predicted value before bias: {y_pred}")
            y_pred = y_pred + bias
            print(f"Predicted value after bias: {y_pred}")

        y_pred = y_pred.flatten()[0]

        result = {'predicted_' + target_feature: y_pred}
        return result




class OperateBatchJobFile:
    def __init__(self, batch_job_path, rec_job_path):
        self.batch_job_path = batch_job_path
        self.rec_job_path = rec_job_path
        self.job_name = None
        self.h_rt = None
        self.slots = None
        

    def open_batch_job_file(self):
        with open(batch_job_path, 'r') as file:
            for line in file:
                if line.startswith('#$ -N'):
                    self.job_name = line.split()[-1].strip()
                elif line.startswith('#$ -l h_rt'):
                    h_rt_str = line.split('=')[-1].strip()
                    h, m, s = map(int, h_rt_str.split(':'))
                    self.h_rt = h * 3600 + m * 60 + s
                elif line.startswith('#$ -pe'):
                    self.slots = int(line.split()[-1].strip())

    def get_job_name(self):
        if self.job_name is None:
            raise ValueError("Job name not found in the batch job file.")
        return self.job_name
    def get_h_rt(self): 
        if self.h_rt is None:
            raise ValueError("Hard wallclock time limit not found in the batch job file.")
        return self.h_rt
    def get_slots(self):
        if self.slots is None:
            raise ValueError("Slots not found in the batch job file.")
        return self.slots


    def write_new_batch_job_file(self, target_feature, prediction):
        # The file content will be the same as the original batch job script except the predicted value
        with open(self.batch_job_path, 'r') as file:
            lines = file.readlines()
        with open(self.rec_job_path, 'w') as file:   
            for line in lines:
                if line.startswith('#$ -N'):
                    original_job_name = line.split()[-1].strip()
                    rec_job_name = f"rec_{original_job_name}"
                    line = f"#$ -N {rec_job_name}\n"
                elif line.startswith('#$ -l h_rt') and target_feature == 'execution_time':
                    predicted_seconds = int(prediction[f'predicted_{target_feature}'])
                    hours, remainder = divmod(predicted_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    predicted_value = f"{hours:02}:{minutes:02}:{seconds:02}"
                    line = f"#$ -l h_rt={predicted_value}\n"
                elif line.startswith('#$ -pe') and target_feature == 'ncpu':
                    predicted_ncpu = int(np.ceil(prediction[f'predicted_{target_feature}']))
                    original_pe_name = line.split()[2].strip()
                    line = f"#$ -pe {original_pe_name} {predicted_ncpu}\n"
                file.write(line)


def get_user_and_project():
    owner = getpass.getuser()
    user_groups = os.getgrouplist(owner, os.getgid())
    project = grp.getgrgid(user_groups[0]).gr_name
    return owner, project

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_npz_biases(path):
    with np.load(path, allow_pickle=True) as data:
        return data['biases'].item()

def encode_input(input_data, features, label_encoder):
    for feature in features:
        if feature in label_encoder and input_data[feature] in label_encoder[feature].classes_:
            input_data[feature] = label_encoder[feature].transform([input_data[feature]])[0]
    return input_data

def predict_and_write_batch(tester, input_data, features, target_feature, batch_job_file, rec_job_name):
    prediction = tester.test_single_input(input_data, features, target_feature) \
        if isinstance(tester, ModelTester) \
        else tester.test_single_user_input(input_data, features, target_feature)
    batch_job_file.write_new_batch_job_file(target_feature, prediction)
    print(f"New batch job file '{rec_job_name}' created successfully.")

if __name__ == "__main__":
    home_dir = os.path.expanduser("~")
    config_dir = os.path.join(home_dir, ".config", "myrec_models")

    batch_job_path = input("Batch job script path: ").strip()
    rec_job_name = f"rec_{os.path.basename(batch_job_path)}"
    rec_job_path = os.path.join(os.path.dirname(batch_job_path), rec_job_name)
    batch_job_file = OperateBatchJobFile(batch_job_path, rec_job_path)
    batch_job_file.open_batch_job_file()
    job_name = batch_job_file.get_job_name()
    h_rt = batch_job_file.get_h_rt()
    slots = batch_job_file.get_slots()
    print(f"Job Name: {job_name}, h_rt: {h_rt}, slots: {slots}")

    target_feature = 'ncpu'
    features = ['project', 'owner', 'job_name', 'slots', 'h_rt']

    owner, project = get_user_and_project()
    input_data = dict(project=project, owner=owner, job_name=job_name, slots=slots, h_rt=h_rt)


    automated_df = pd.read_csv('/projectnb/peaclab-mon/boztop/module_files/data/automated_decision.csv')
    label = automated_df.loc[automated_df['user'] == owner, 'label'].values[0]
    if label == 0:
        print(f"User {owner} label 0. K-means model.")
        label_encoder = load_pickle(os.path.join(config_dir, "label_encoders.pkl"))
        model_data = load_pickle(os.path.join(config_dir, "trained_models.pkl"))
        models, biases = model_data['models'], model_data['biases']
        cluster_centers = np.load(os.path.join(config_dir, "cluster_centers.npy"))
        input_data = encode_input(input_data, ['project', 'owner', 'job_name'], label_encoder)
        tester = ModelTester(models, biases, cluster_centers, bias_type='two_sigma', model_type='rf')
        predict_and_write_batch(tester, input_data, features, target_feature, batch_job_file, rec_job_name)
    else:
        print(f"User {owner} label 1. User-based model.")
        label_encoder = load_pickle(os.path.join(config_dir, f"user_based_label_encoders_{owner}.pkl"))
        model = load_pickle(os.path.join(config_dir, f"user_based_model_{owner}.pkl"))
        biases = load_npz_biases(os.path.join(config_dir, f"user_based_biases_{owner}.npz"))
        tester = UserBasedModelTester(model, biases, label_encoder, bias_type='none')
        predict_and_write_batch(tester, input_data, features, target_feature, batch_job_file, rec_job_name)
