import pandas as pd
from sklearn.preprocessing import LabelEncoder

class EaglePreprocessor:
    def __init__(self, file_path='/projectnb/peaclab-mon/boztop/resource-allocation/datasets/nrel-eagle/eagle_data.csv.bz2'):
        self.file_path = file_path
        self.train_features = ['wallclock_req','processors_req','mem_req','nodes_req','gpus_req','user','partition','name','account']
        self.target_feature = 'run_time'
        self.df = None
        self.df_success = None
        self.df_failure = None

    def load_data(self):
        """
        This function loads the Eagle dataset from the given file path.

        Returns:
            DataFrame : The loaded dataset as a pandas DataFrame.
        """
        self.df = pd.read_csv(self.file_path, compression='bz2')

    def filter_data(self):
        """
        Filter the dataset into successful and failed jobs.

        Returns:
            df_success, df_failure (DataFrame, DataFrame): The filtered datasets for successful and failed jobs.
        """
        df_success0 = self.df[self.df['state'] == 'COMPLETED']
        self.df_success = df_success0[df_success0['run_time'] != 0]
        self.df_failure = self.df[self.df['state'] == 'TIMEOUT']

    def calculate_wait_time(self):
        """
        This function calculates the wait time for successful jobs.

        Returns:
            df (DataFrame): The dataset with the wait time calculated.
        """
        self.df_success['start_time'] = pd.to_datetime(self.df_success['start_time'])
        self.df_success['submit_time'] = pd.to_datetime(self.df_success['submit_time'])
        self.df_success['wait_time'] = (self.df_success['start_time'] - self.df_success['submit_time']).dt.total_seconds()

    def encode_categorical_features(self, categorical_features):
        """
        This function encodes the categorical features in the dataset

        Args:
            categorical_features (list): The list of categorical features to be encoded.

        Returns:
            df (dataFrame): The dataset with the categorical features encoded.
        """
        label_encoders = {}
        for col in categorical_features:
            print(f'Encoding the feature {col}')
            le = LabelEncoder()
            self.df_success[col] = le.fit_transform(self.df_success[col].astype(str))
            label_encoders[col] = le

    def preprocess_data(self):
        """
        This function preprocesses the Eagle dataset.

        Returns:
            df_success, df_failure (DataFrame, DataFrame): The preprocessed datasets for successful and failed jobs.
        """
        print("Loading data...\n")
        self.load_data()
        
        print("Filtering data...\n")
        self.filter_data()
        
        print("Calculating wait time for successful jobs...\n")
        self.calculate_wait_time()
        
        print("Encoding categorical features...\n")
        categorical_features = ['job_id', 'user', 'account', 'partition', 'qos', 'name', 'work_dir', 'submit_line']
        self.encode_categorical_features(categorical_features)
        
        print("Preprocessing complete!\n")
        return self.df_success
