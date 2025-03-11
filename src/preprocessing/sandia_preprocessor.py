import pandas as pd
from sklearn.preprocessing import LabelEncoder

class SandiaPreprocessor:
    def __init__(self, drop_columns=['State', 'Cluster', 'Elapsed.1'], categorical_features=['User', 'JobName', 'WorkDir'], file_path = '/projectnb/peaclab-mon/boztop/resource-allocation/datasets/sandia/may-october-2024-scorecard.csv_anom'):
        self.file_path = file_path
        self.drop_columns = drop_columns
        self.categorical_features = categorical_features
        self.train_features = ['User','JobName','WorkDir']
        self.df = None
        self.label_encoders = {col: LabelEncoder() for col in categorical_features}

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df = self.df.drop(self.drop_columns, axis=1)

    def encode_categorical_features(self):
        for col in self.categorical_features:
            self.df[col] = self.label_encoders[col].fit_transform(self.df[col])

    @staticmethod
    def time_to_seconds(time_str):
        if '-' in time_str:
            days, time_part = time_str.split('-')
            h, m, s = map(int, time_part.split(':'))
            return int(days) * 86400 + h * 3600 + m * 60 + s
        else:
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s

    def preprocess_data(self):
        self.load_data()
        self.encode_categorical_features()
        self.df['execution_time'] = self.df['Elapsed'].apply(self.time_to_seconds)
        self.df = self.df[self.df['execution_time'] > 0]
        self.df['start_time'] = pd.to_datetime(self.df['Start'], unit='s')
        return self.df