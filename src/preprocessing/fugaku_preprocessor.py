import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class FugakuPreprocessor:
    """
    A class to preprocess the Fugaku dataset.
    """
    def __init__(self, directory='/projectnb/peaclab-mon/boztop/resource-allocation/datasets/fugaku/24_04.parquet'):
        self.directory = directory
        self.train_features = ['usr', 'jnam', 'cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req']
        self.df = self.load_dataset()
    
    def load_dataset(self):
        """
        This method loads the dataset from the specified Parquet file.
        """
        return pd.read_parquet(self.directory)

    def convert_to_datetime(self, datetime_columns):
        """
        This method converts the specified columns to datetime format.

        Args:
            datetime_columns (list): List of column names to convert.

        Returns:
           DataFrame: The DataFrame with updated datetime columns.
        """
        for col in datetime_columns:
            self.df[col] = pd.to_datetime(self.df[col])
        return self.df

    def calculate_time_metrics(self):
        """
        This method calculates the wait time and run time in seconds.

        Returns:
            DataFrame: The DataFrame with calculated time metrics.
        """
        self.df['wait_time'] = (self.df['sdt'] - self.df['adt']).dt.total_seconds()
        self.df['run_time'] = (self.df['edt'] - self.df['sdt']).dt.total_seconds()
        return self.df

    def split_by_exit_state(self):
        """
        This method splits the dataframe into successful and failed jobs based on the exit state.

        Returns:
            df_success, df_failure (DataFrame, DataFrame): Two DataFrames, one for successful jobs and one for failed jobs.
        """
        df_success0 = self.df[self.df['exit state'] == 'completed']
        df_success = df_success0[df_success0['run_time'] != 0]
        df_failure = self.df[self.df['exit state'] == 'failed']
        return df_success, df_failure

    def sort_by_column(self, column_name):
        """
        This method sorts the DataFrame by a specified column.

        Args:
            column_name (str): The column to sort by.

        Returns:
            DataFrame: The sorted DataFrame.
        """
        return self.df.sort_values(by=column_name)

    def encode_categorical_features(self, df, categorical_features):
        """
        This method encodes specified categorical features using Label Encoding.
        
        Args:
            df (DataFrame): The input DataFrame.
            categorical_features (list): List of categorical feature column names.

        Returns:
           DataFrame: The DataFrame with encoded features.
        """
        label_encoders = {col: LabelEncoder() for col in categorical_features}
        for col in categorical_features:
            df[col] = label_encoders[col].fit_transform(df[col])
        return df

    def preprocess_data(self):
        """
        The main data preprocessing method for Fugaku dataset.

        Returns:
            df_success, df_failure, numerical_submission_features (DataFrame, DataFrame, list): Preprocessed successful jobs DataFrame, failed jobs DataFrame, numerical features list.
        """
        self.df = self.convert_to_datetime(['adt', 'sdt', 'edt'])
        self.df = self.calculate_time_metrics()
        sorted_df = self.sort_by_column('edt')
        df_success, df_failure = self.split_by_exit_state()

        numerical_submission_features = ['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req']
        categorical_submission_features = ['jid', 'usr', 'jnam', 'jobenv_req']

        df_success = self.encode_categorical_features(df_success, categorical_submission_features)

        return df_success

    def convert_fugaku_to_swf(self, df):
        """
        This method converts the Fugaku dataset to the Standard Workload Format (SWF).

        Args:
            df (DataFrame): The input DataFrame containing Fugaku job data.

        Returns:
            DataFrame: The DataFrame converted to SWF format.
        """
        swf_df = pd.DataFrame()
        
        df = df.sort_values(by='adt').reset_index(drop=True)
        swf_df['Job Number'] = range(1, len(df) + 1)

        df['adt'] = pd.to_datetime(df['adt'], errors='coerce') 
        df['adt'] = df['adt'].astype('int64') // 10**9  
        min_submit_time = df['adt'].min()
        swf_df['Submit Time'] = df['adt'] - min_submit_time

        swf_df['Wait Time'] = df['wait_time'].fillna(-1)
        swf_df['Run Time'] = df['run_time'].fillna(-1)

        swf_df['Number of Allocated Processors'] = df['cnumat'].fillna(-1)

        swf_df['Average CPU Time Used'] = (df['usctmut'].fillna(0) / df['cnumat'].replace(0, np.nan)).fillna(-1)
        
        swf_df['Used Memory'] = df['mmszu'].fillna(-1)
        swf_df['Requested Number of Processors'] = df['cnumr'].fillna(-1)
        swf_df['Requested Time'] = df['elpl'].fillna(-1)
        swf_df['Requested Memory'] = df['mszl'].fillna(-1)
        
        status_mapping = {'completed': 1, 'FAILED': 0, 'CANCELLED': 5}
        swf_df['Status'] = df['exit state'].map(status_mapping).fillna(-1)
        
        user_mapping = {user: idx + 1 for idx, user in enumerate(df['usr'].unique())}
        swf_df['User ID'] = df['usr'].map(user_mapping).fillna(-1)
        
        swf_df['Group ID'] = -1
        
        app_mapping = {app: idx + 1 for idx, app in enumerate(df['jnam'].unique())}
        swf_df['Executable (Application) Number'] = df['jnam'].map(app_mapping).fillna(-1)
        
        swf_df['Queue Number'] = -1  # No info
        swf_df['Partition Number'] = -1  # No info
        swf_df['Preceding Job Number'] = -1  # No info
        swf_df['Think Time from Preceding Job'] = -1  # No info
        
        return swf_df
